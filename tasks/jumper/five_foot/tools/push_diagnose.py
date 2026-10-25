#!/usr/bin/env python3
"""Per-control-step diagnostics for the fixed-pose push validation.

The trial protocol is copied step for step from ``push_objects.py`` -- the same
reset, the same arm walk, the same box placement, the same commands, the same
stability waits and the same success checks -- with one addition: every control
step is recorded into a per-trial NPZ and a JSON summary, for diagnosing why a
trial fails to establish sustained contact.

Nothing about the trial itself changes: the policy, the physics and the checks
are exactly the ones ``push_objects.py`` runs, so for the same seed and trial
index the trajectory is bit-identical to the validator's.  The validator itself
is untouched; this is a separate opt-in entry point whose outputs are
structured files, not per-step terminal output.

Usage mirrors the validator::

    python tasks/jumper/five_foot/tools/push_diagnose.py \\
        --checkpoint <path>/model_87198.pt --seed 0 --trials 5 \\
        --distance-m 0.08 --speed 0.35 --initial-gap-mm 0 \\
        --shoulder-offset-rad 0.15 --backoff-m 0.05 \\
        --backoff-lateral-speed -0.15

Recorded per control step (in the per-trial NPZ): the episode step and trial
phase; base position, quaternion, yaw/pitch/roll and linear/angular velocity;
the twist and body-pose commands; the policy action (mean, max-abs, first-step
changes); the nearest claw-to-box geom distance (negative while penetrating);
push-contact found/force; per-foot contact states and net forces; the top
non-foot ground contact; and the box pose, velocity and displacement.  Reset
samples (base pose, all 21 observed joint positions, foot friction) are in the
JSON summary per trial.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from tasks.jumper.five_foot.tools.grasp_objects import (
    FALL_TILT_RAD,
    GRASP_READY,
    STABLE_ANGULAR_RAD_S,
    STABLE_LINEAR_MPS,
    STABLE_TILT_RAD,
    _render_video,
    _stash,
)

PHASE_NAMES = (
    "reset",
    "arm_walk",
    "wait_stable",
    "settle",
    "push",
    "stop_stable",
    "retract",
    "retract_stable",
    "object_stable",
    "backoff",
    "backoff_stable",
)


def _jsonable(value):
    """A value that json.dump accepts, falling back to repr for config objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _event_snapshot(cfg) -> dict:
    """Which reset/startup/step events this run has, so the DR surface is on file."""
    out: dict = {}
    for name, term in cfg.events.items():
        func = getattr(term.func, "__name__", type(term.func).__name__)
        out[name] = {
            "func": func,
            "mode": term.mode,
            "params": {str(k): _jsonable(v) for k, v in (term.params or {}).items()},
        }
    return out


def _quat_yaw_rad(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="jumper.five_foot")
    parser.add_argument("--backend", choices=("native",), default="native")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--object", choices=("push_box",), default="push_box")
    parser.add_argument("--orientation", choices=("upright",), default="upright")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--distance-m", type=float, default=0.10)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--lateral-speed", type=float, default=0.0)
    parser.add_argument("--backoff-lateral-speed", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--initial-gap-mm", type=float, default=8.0)
    parser.add_argument("--shoulder-offset-rad", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--backoff-m", type=float, default=0.05)
    parser.add_argument("--max-lateral-m", type=float, default=0.10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "diag",
        help="directory for the per-trial NPZ files and the JSON summary",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-trial", type=int, default=1)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if min(args.distance_m, args.speed, args.timeout_s, args.backoff_m) <= 0:
        parser.error("distance, speed, timeout and backoff must be positive")
    if args.initial_gap_mm < 0:
        parser.error("--initial-gap-mm must be non-negative")

    warnings.filterwarnings("ignore")
    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks
    from tasks.jumper.five_foot.claw import (
        ARM_JOINTS,
        FINGER_JOINT,
        FIVE_FOOT_JOINTS,
        GRIPPER_CLOSED,
        OBSERVED_JOINTS,
        SQUEEZE_LEAD,
    )
    from tasks.jumper.five_foot.env_cfg import GROUND_FOOT_GEOMS
    from tasks.jumper.five_foot.mdp.grasp import Claw
    from tasks.jumper.five_foot.objects import PUSH_BOX, _place_props, _prop, _solid_spec

    res = resolve(backend=args.backend, device="cpu", num_envs=1)
    use_backend(res)
    cfg = tasks.load_env_cfg(args.task, play=True, task_args={"objects": True})
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    # Same pin as push_objects.py: the validator writes commands directly, and
    # the play-config default (3.0, 8.0) s lets the command manager re-roll
    # time_left mid-trial, which (rel_standing_envs = 1.0) cancels the written
    # push/backoff command.  Keep the diagnostic twin's protocol identical.
    cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    push_box = PUSH_BOX
    cfg.scene.entities[push_box.name] = _prop(
        partial(_solid_spec, push_box), (0.0, 0.0, push_box.half_extents[2])
    )
    _place_props(cfg, {push_box.name: cfg.scene.entities[push_box.name]})
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="push_object_contact",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(
                mode="subtree", pattern=args.object, entity=args.object
            ),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=8,
            history_length=cfg.decimation,
        ),
    )
    cfg.sim.contact_sensor_maxmatch = max(int(cfg.sim.contact_sensor_maxmatch or 0), 512)
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)

    agent = tasks.load_agent_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (tasks.load_runner_cls(args.task) or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device=res.device
    )
    runner.load(
        args.checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=res.device,
    )
    policy = runner.get_inference_policy(device=res.device)

    robot = env.scene["robot"]
    prop = env.scene[args.object]
    claw = Claw(env)
    arm = robot.find_joints(list(ARM_JOINTS))[0]
    finger = robot.find_joints([FINGER_JOINT])[0][0]
    push_pose = list(GRASP_READY)
    push_pose[1] += args.shoulder_offset_rad
    arm_target = torch.tensor([push_pose], device=res.device)
    finger_target = torch.zeros(1, 1, device=res.device)
    grip_target = GRIPPER_CLOSED
    twist = env.command_manager.get_term("twist")
    pose = env.command_manager.get_term("body_pose")
    half_x, _half_y, half_z = push_box.half_extents
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=res.device)
    support_half_height = half_z
    forward_half_extent = half_x

    # Sim-model geom ids of the carried arm (the push face) and the box, for the
    # per-step nearest-distance reading.
    model = env.sim.mj_model
    lf_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            or ""
        ).startswith("robot/LF_")
    ]
    box_local, _ = prop.find_geoms([push_box.name])
    box_geoms = [int(prop.indexing.geom_ids[i]) for i in box_local]
    if not box_geoms:
        raise ValueError("the push_box entity has no collision geom in the model")
    foot_local, _ = robot.find_geoms(list(GROUND_FOOT_GEOMS), preserve_order=True)
    foot_geom_ids = [int(robot.indexing.geom_ids[i]) for i in foot_local]

    def command(vx: float, vy: float = 0.0) -> None:
        twist.cfg.ranges.lin_vel_x = (vx, vx)
        twist.cfg.ranges.lin_vel_y = (vy, vy)
        twist.cfg.ranges.ang_vel_z = (0.0, 0.0)
        twist.is_standing_env[:] = False
        twist.is_heading_env[:] = False
        twist.is_world_env[:] = False
        twist.vel_command_b[:] = torch.tensor([vx, vy, 0.0], device=res.device)
        twist.vel_command_w[:] = twist.vel_command_b
        pose.is_neutral_env[:] = args.pitch_deg == 0.0
        pose.pose_target_b[:, 0] = math.radians(args.pitch_deg)
        pose.pose_target_b[:, 1:] = 0.0

    def tilt() -> tuple[float, float]:
        gravity = robot.data.projected_gravity_b[0]
        pitch = math.asin(max(-1.0, min(1.0, float(gravity[0]))))
        roll = math.atan2(float(-gravity[1]), float(-gravity[2]))
        return pitch, roll

    def fell() -> bool:
        pitch, roll = tilt()
        height = float(robot.data.root_link_pos_w[0, 2] - env.scene.env_origins[0, 2])
        return height < 0.05 or max(abs(pitch), abs(roll)) > FALL_TILT_RAD

    def stable() -> bool:
        pitch, roll = tilt()
        return (
            float(robot.data.root_link_lin_vel_b[0, :2].norm()) < STABLE_LINEAR_MPS
            and float(robot.data.root_link_ang_vel_b[0].norm()) < STABLE_ANGULAR_RAD_S
            and max(abs(pitch), abs(roll)) < STABLE_TILT_RAD
            and not fell()
        )

    def nonfoot_detail() -> str | None:
        sensor = env.scene.sensors["non_foot_ground_contact"]
        if sensor.data.force is None:
            return None
        magnitudes = sensor.data.force[0].norm(dim=-1)
        index = int(magnitudes.argmax())
        force = float(magnitudes[index])
        if force <= 1.0:
            return None
        primary = sensor.primary_names[index // sensor.cfg.num_slots]
        return f"{primary} against terrain ({force:.2f} N)"

    def contact_now() -> bool:
        found = env.scene.sensors["push_object_contact"].data.found
        return found is not None and bool(found[0].any())

    def set_targets() -> None:
        robot.set_joint_position_target(arm_target, joint_ids=arm)
        angle = robot.data.joint_pos[:, [finger]]
        finger_target.fill_(min(grip_target, float(angle[0, 0]) + SQUEEZE_LEAD))
        robot.set_joint_position_target(finger_target, joint_ids=[finger])

    def nearest_claw_box_m() -> float:
        env_model, mjdata = env.sim.env_mjdata(0)
        return min(
            mujoco.mj_geomDistance(env_model, mjdata, g1, g2, 10.0, None)
            for g1 in lf_geoms
            for g2 in box_geoms
        )

    # ── Per-step recording ────────────────────────────────────────────────
    rec: dict[str, list] = {}
    prev_action: torch.Tensor | None = None
    step_index = 0
    phase = PHASE_NAMES[0]
    auto_reset = False

    def record(action: torch.Tensor) -> None:
        nonlocal prev_action, step_index, auto_reset
        # Every tensor append is copied: on the CPU backend ``.numpy()`` may
        # alias the live simulation or command buffer, and a later in-place
        # update would rewrite every previously recorded row.
        def copy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy().copy()

        rec["step"].append(step_index)
        step_index += 1
        rec["phase"].append(PHASE_NAMES.index(phase))
        episode = int(env.episode_length_buf[0])
        rec["episode_step"].append(episode)
        if episode == 0 and step_index > 1:
            auto_reset = True  # the env reset itself mid-trial (e.g. too_low)
        base_pos = copy(robot.data.root_link_pos_w[0])
        base_quat = copy(robot.data.root_link_quat_w[0])
        pitch, roll = tilt()
        rec["base_pos"].append(base_pos)
        rec["base_quat"].append(base_quat)
        rec["base_yaw_pitch_roll"].append(
            (_quat_yaw_rad(base_quat), pitch, roll)
        )
        rec["base_lin_vel_w"].append(copy(robot.data.root_link_lin_vel_w[0]))
        rec["base_lin_vel_b"].append(copy(robot.data.root_link_lin_vel_b[0]))
        rec["base_ang_vel_b"].append(copy(robot.data.root_link_ang_vel_b[0]))
        rec["cmd_vel"].append(copy(twist.vel_command_b[0]))
        rec["cmd_pose"].append(copy(pose.pose_target_b[0]))
        action_np = copy(action[0])
        rec["action"].append(action_np)
        rec["action_mean"].append(float(np.abs(action_np).mean()))
        rec["action_max_abs"].append(float(np.abs(action_np).max()))
        delta = (
            0.0
            if prev_action is None
            else float(np.abs(action_np - prev_action).sum())
        )
        rec["action_delta_l1"].append(delta)
        prev_action = action_np.copy()
        rec["claw_box_dist"].append(nearest_claw_box_m())
        rec["mouth_box_dist"].append(
            float((prop.data.root_link_pos_w[0] - claw.mouth()[0]).norm())
        )
        push_sensor = env.scene.sensors["push_object_contact"]
        rec["push_found"].append(contact_now())
        rec["push_force_max"].append(
            float(push_sensor.data.force[0].norm(dim=-1).max())
        )
        feet_sensor = env.scene.sensors["feet_ground_contact"]
        rec["foot_found"].append(
            (feet_sensor.data.found[0] > 0).detach().cpu().numpy().astype(bool)
        )
        rec["foot_force"].append(
            feet_sensor.data.force[0].norm(dim=-1).detach().cpu().numpy()
        )
        ground = env.scene.sensors["non_foot_ground_contact"]
        magnitudes = ground.data.force[0].norm(dim=-1).detach().cpu().numpy()
        rec["nonfoot_max_force"].append(float(magnitudes.max()))
        rec["nonfoot_argmax"].append(int(magnitudes.argmax()))
        rec["nonfoot_over_1n"].append(int((magnitudes > 1.0).sum()))
        rec["box_pos"].append(copy(prop.data.root_link_pos_w[0]))
        rec["box_quat"].append(copy(prop.data.root_link_quat_w[0]))
        rec["box_lin_vel_w"].append(copy(prop.data.root_link_lin_vel_w[0]))
        rec["box_disp"].append(
            (prop.data.root_link_pos_w[0, :2] - start_object).detach().cpu().numpy()
            if start_object is not None
            else np.zeros(2)
        )
        rec["arm_pos"].append(copy(robot.data.joint_pos[0, arm]))
        rec["arm_target"].append(copy(arm_target[0]))
        rec["finger_pos"].append(float(robot.data.joint_pos[0, finger]))
        rec["finger_target"].append(float(finger_target[0, 0]))

    def step(count: int) -> int:
        nonlocal obs
        contacts = 0
        # Contact-sensor buffers are reset in place between trials.  Creating a
        # lazily allocated buffer under inference_mode makes that later reset
        # illegal; no_grad still avoids the policy graph and keeps tensors mutable.
        with torch.no_grad():
            for _ in range(count):
                obs = env.observation_manager.compute()
                action = policy(obs)
                env.step(action)
                set_targets()
                contacts += int(contact_now())
                record(action)
                if recording:
                    _, data = env.sim.env_mjdata(0)
                    recorded_qpos.append(data.qpos.copy())
        return contacts

    def wait_stable(limit: int = 250, consecutive: int = 25) -> bool:
        good = 0
        for _ in range(limit):
            step(1)
            good = good + 1 if stable() else 0
            if good >= consecutive:
                return True
        return False

    def wait_object_stable(limit: int = 250, consecutive: int = 25) -> bool:
        good = 0
        for _ in range(limit):
            step(1)
            linear = float(prop.data.root_link_lin_vel_w[0, :2].norm())
            angular = float(prop.data.root_link_ang_vel_w[0].norm())
            good = good + 1 if linear < 0.01 and angular < 0.10 else 0
            if good >= consecutive:
                return True
        return False

    def object_tilt_deg() -> float:
        quat = prop.data.root_link_quat_w[0].detach().cpu().numpy()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        matrix = rotation.reshape(3, 3)
        return math.degrees(
            math.acos(max(-1.0, min(1.0, float(matrix[2, 2]))))
        )

    recorded_qpos: list[np.ndarray] = []
    recording = False
    obs = None
    trials: list[dict] = []

    for trial in range(1, args.trials + 1):
        rec = {key: [] for key in (
            "step", "phase", "episode_step", "base_pos", "base_quat",
            "base_yaw_pitch_roll", "base_lin_vel_w", "base_lin_vel_b",
            "base_ang_vel_b", "cmd_vel", "cmd_pose", "action", "action_mean",
            "action_max_abs", "action_delta_l1", "claw_box_dist",
            "mouth_box_dist", "push_found", "push_force_max", "foot_found",
            "foot_force", "nonfoot_max_force", "nonfoot_argmax",
            "nonfoot_over_1n", "box_pos", "box_quat", "box_lin_vel_w",
            "box_disp", "arm_pos", "arm_target", "finger_pos", "finger_target",
        )}
        prev_action = None
        step_index = 0
        auto_reset = False
        start_object = None

        env.reset()
        command(0.0)
        # The reset samples, before anything is moved or stashed.
        observed_ids, observed_names = robot.find_joints(
            list(OBSERVED_JOINTS), preserve_order=True
        )
        reset_sample = {
            "base_pos": [float(v) for v in robot.data.root_link_pos_w[0]],
            "base_quat": [float(v) for v in robot.data.root_link_quat_w[0]],
            "base_yaw_deg": math.degrees(
                _quat_yaw_rad(robot.data.root_link_quat_w[0])
            ),
            "joint_pos": {
                name: float(robot.data.joint_pos[0, i])
                for i, name in zip(observed_ids, observed_names, strict=True)
            },
            "foot_friction": {
                name: [float(v) for v in model.geom_friction[gid]]
                for name, gid in zip(GROUND_FOOT_GEOMS, foot_geom_ids, strict=True)
            },
        }
        other_props = tuple(name for name in ("soap", "notebook", "can") if name != args.object)
        for index, name in enumerate(other_props):
            _stash(env, name, res.device, index)
        # Walk the arm to the push pose with bounded adjacent targets.
        start = robot.data.joint_pos[0, arm].clone()
        goal = torch.tensor(push_pose, device=res.device)
        count = max(1, math.ceil(float((goal - start).abs().max()) / 0.01))
        phase = "arm_walk"
        for index in range(count):
            arm_target[0] = start + (goal - start) * ((index + 1) / count)
            step(1)
        phase = "wait_stable"
        wait_stable()

        mouth = claw.mouth()[0]
        position = torch.tensor(
            [[
                float(mouth[0]) + forward_half_extent + args.initial_gap_mm / 1000.0,
                float(mouth[1]),
                support_half_height,
            ]],
            device=res.device,
        )
        prop.write_root_link_pose_to_sim(torch.cat([position, root_quat], dim=1))
        prop.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=res.device))
        env.sim.forward()
        phase = "settle"
        step(25)
        start_object = prop.data.root_link_pos_w[0, :2].clone()
        start_base = robot.data.root_link_pos_w[0, :2].clone()
        initial_nonfoot = nonfoot_detail()

        recording = args.video is not None and trial == args.video_trial
        contacts = 0
        command(args.speed, args.lateral_speed)
        phase = "push"
        for _ in range(math.ceil(args.timeout_s / env.step_dt)):
            contacts += step(1)
            displacement = prop.data.root_link_pos_w[0, :2] - start_object
            if float(displacement.norm()) >= args.distance_m:
                break
        command(0.0)
        phase = "stop_stable"
        stopped = wait_stable()
        stop_object = prop.data.root_link_pos_w[0, :2].clone()
        stop_base = robot.data.root_link_pos_w[0, :2].clone()
        object_forward = float(stop_object[0] - start_object[0])
        object_lateral = float(abs(stop_object[1] - start_object[1]))
        object_distance = float((stop_object - start_object).norm())
        object_tilt = object_tilt_deg()
        hazard = initial_nonfoot or nonfoot_detail()
        did_fall = fell()

        # Lift the contact face clear before retreating.  Reversing with the
        # low, diagonal claw still beside the prop swept it sideways by 29 mm.
        retract = arm_target[0].clone()
        retract[1] -= 0.30
        begin = robot.data.joint_pos[0, arm].clone()
        count = max(1, math.ceil(float((retract - begin).abs().max()) / 0.01))
        phase = "retract"
        for index in range(count):
            arm_target[0] = begin + (retract - begin) * ((index + 1) / count)
            step(1)
        phase = "retract_stable"
        wait_stable()
        phase = "object_stable"
        wait_object_stable()
        backoff_object_start = prop.data.root_link_pos_w[0, :2].clone()
        retract_motion = float((backoff_object_start - stop_object).norm())

        command(-args.speed, args.backoff_lateral_speed)
        phase = "backoff"
        for _ in range(math.ceil(args.timeout_s / env.step_dt)):
            step(1)
            backed = float(stop_base[0] - robot.data.root_link_pos_w[0, 0])
            if backed >= args.backoff_m:
                break
        command(0.0)
        phase = "backoff_stable"
        wait_stable()
        recording = False
        backed = float(stop_base[0] - robot.data.root_link_pos_w[0, 0])
        drag = float((prop.data.root_link_pos_w[0, :2] - backoff_object_start).norm())

        checks = (
            (contacts >= 3, "no_sustained_contact"),
            (object_distance >= 0.9 * args.distance_m, "push_distance"),
            (object_lateral <= args.max_lateral_m, "lateral_deviation"),
            (object_tilt <= 20.0, "object_tipped"),
            (stopped, "stop_not_stable"),
            (retract_motion <= 0.015, "object_moved_during_retract"),
            (backed >= 0.9 * args.backoff_m, "backoff_distance"),
            (drag <= 0.01, "object_dragged_during_backoff"),
            (not did_fall, "fell_over_or_too_low"),
            (hazard is None, "nonfoot_ground_contact"),
        )
        failure = next((reason for ok, reason in checks if not ok), None)

        # Contact structure over the push phase, from the per-step record.
        push_found = np.asarray(rec["push_found"], dtype=bool)
        phases = np.asarray(rec["phase"], dtype=np.int64)
        in_push = phases == PHASE_NAMES.index("push")
        found_in_push = push_found & in_push
        runs: list[list[int]] = []
        run_start = None
        for i, ok in enumerate(found_in_push):
            if ok and run_start is None:
                run_start = i
            elif not ok and run_start is not None:
                runs.append([run_start, i - run_start])
                run_start = None
        if run_start is not None:
            runs.append([run_start, len(found_in_push) - run_start])
        losses = [
            [runs[k][0] + runs[k][1], runs[k + 1][0] - runs[k][0] - runs[k][1]]
            for k in range(len(runs) - 1)
        ]
        first_contact = int(np.nonzero(found_in_push)[0][0]) if found_in_push.any() else None

        npz_path = args.out / f"push_diag_seed{args.seed}_trial{trial}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            **{key: np.asarray(value) for key, value in rec.items()},
        )
        trials.append({
            "trial": trial,
            "npz": npz_path.name,
            "steps": step_index,
            "auto_reset_detected": auto_reset,
            "reset": reset_sample,
            "phase_steps": {
                name: int((phases == k).sum())
                for k, name in enumerate(PHASE_NAMES)
            },
            "push": {
                "first_contact_step": first_contact,
                "contact_runs": runs,
                "contact_loss_intervals": losses,
                "contact_steps": contacts,
                "max_contact_force_n": float(np.asarray(rec["push_force_max"]).max()),
                "min_claw_box_dist_m": float(np.asarray(rec["claw_box_dist"]).min()),
            },
            "checks": {
                "success": failure is None,
                "failure": failure,
                "contact_steps": contacts,
                "object_forward_m": object_forward,
                "object_lateral_m": object_lateral,
                "object_planar_distance_m": object_distance,
                "object_tilt_deg": object_tilt,
                "base_forward_m": float(stop_base[0] - start_base[0]),
                "stopped_stably": stopped,
                "backed_away_m": backed,
                "object_motion_during_retract_m": retract_motion,
                "object_motion_during_backoff_m": drag,
                "fell": did_fall,
                "nonfoot_ground_contact": hazard is not None,
                "nonfoot_contact": hazard,
            },
        })
        print(
            f"diag trial {trial:02d}: {'ok' if failure is None else 'FAILED ' + str(failure)}; "
            f"push={object_distance * 1000:.0f} mm "
            f"(x={object_forward * 1000:.0f}, y={object_lateral * 1000:.0f}) "
            f"tilt={object_tilt:.1f} deg backoff={backed * 1000:.0f} mm "
            f"contact_steps={contacts}"
        )

    report = {
        "tool": "push_diagnose",
        "task": args.task,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "object": args.object,
        "orientation": args.orientation,
        "interaction": "rigid_body_contact_push",
        "parameters": {
            "distance_m": args.distance_m,
            "speed_m_s": args.speed,
            "lateral_speed_m_s": args.lateral_speed,
            "backoff_lateral_speed_m_s": args.backoff_lateral_speed,
            "initial_gap_mm": args.initial_gap_mm,
            "shoulder_offset_rad": args.shoulder_offset_rad,
            "pitch_deg": args.pitch_deg,
            "backoff_m": args.backoff_m,
            "max_lateral_m": args.max_lateral_m,
            "timeout_s": args.timeout_s,
        },
        "phase_names": list(PHASE_NAMES),
        "foot_names": list(GROUND_FOOT_GEOMS),
        "action_joint_names": list(FIVE_FOOT_JOINTS),
        "observed_joint_names": list(OBSERVED_JOINTS),
        "nonfoot_names": env.scene.sensors["non_foot_ground_contact"].primary_names,
        "events": _event_snapshot(cfg),
        "trials": trials,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / f"push_diag_seed{args.seed}.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {summary}")
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        _render_video(env.sim.mj_model, recorded_qpos, 1.0 / env.step_dt, args.video)
        print(f"video: {args.video}")
    env.close()
    return 0 if all(t["checks"]["success"] for t in trials) else 1


if __name__ == "__main__":
    raise SystemExit(main())
