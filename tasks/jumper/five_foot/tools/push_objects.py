#!/usr/bin/env python3
"""Validate low-speed physical pushing with the five-foot policy.

The prop is placed once, before contact, at a known pose in front of the closed
claw.  From then on it is a free rigid body: the policy moves the base and only
MuJoCo contact moves the prop.  A successful trial pushes it without tipping,
stops, and backs away without dragging it.  Thin props may be laid flat, which
is the natural and stable presentation for pushing rather than grasping.

This is deliberately a fixed-pose contact test, not autonomous perception or
approach.  It is the fallback interaction to validate before adding navigation.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
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


@dataclass
class PushResult:
    trial: int
    success: bool
    failure: str | None
    contact_steps: int
    object_forward_m: float
    object_lateral_m: float
    object_planar_distance_m: float
    object_tilt_deg: float
    base_forward_m: float
    stopped_stably: bool
    backed_away_m: float
    object_motion_during_retract_m: float
    object_motion_during_backoff_m: float
    fell: bool
    nonfoot_ground_contact: bool
    nonfoot_contact: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="jumper.five_foot")
    parser.add_argument("--backend", choices=("warp", "native"), default="native")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--object",
        choices=("push_box", "soap", "notebook", "can"),
        default="push_box",
    )
    parser.add_argument(
        "--orientation",
        choices=("upright", "sideways", "flat"),
        default="upright",
        help="sideways yaws 90 degrees; flat rotates a box-like prop onto its broad face",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--distance-m", type=float, default=0.10)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument(
        "--lateral-speed",
        type=float,
        default=0.0,
        help="simultaneous lateral command; negative counters the left-claw sweep",
    )
    parser.add_argument(
        "--backoff-lateral-speed",
        type=float,
        default=0.0,
        help="lateral command while retreating after the push face is raised",
    )
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--initial-gap-mm", type=float, default=8.0)
    parser.add_argument(
        "--shoulder-offset-rad",
        type=float,
        default=0.0,
        help="offset LF_J1 from grasp-ready; positive lowers the push face",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=0.0,
        help="held body-pitch command, within the policy's trained moving band",
    )
    parser.add_argument(
        "--open-guide",
        action="store_true",
        help="keep the claw open so its two jaws guide rather than pinch the prop",
    )
    parser.add_argument("--backoff-m", type=float, default=0.05)
    parser.add_argument("--max-lateral-m", type=float, default=0.10)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-trial", type=int, default=1)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if min(args.distance_m, args.speed, args.timeout_s, args.backoff_m) <= 0:
        parser.error("distance, speed, timeout and backoff must be positive")
    if args.initial_gap_mm < 0:
        parser.error("--initial-gap-mm must be non-negative")
    if args.video is not None and args.backend != "native":
        parser.error("--video currently requires --backend native")

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
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
        SQUEEZE_LEAD,
    )
    from tasks.jumper.five_foot.mdp.grasp import Claw
    from tasks.jumper.five_foot.objects import (
        PUSH_BOX,
        SOLIDS,
        _place_props,
        _prop,
        _solid_spec,
    )

    res = resolve(
        backend=args.backend,
        device="cuda" if args.backend == "warp" else "cpu",
        num_envs=1,
    )
    use_backend(res)
    cfg = tasks.load_env_cfg(args.task, play=True, task_args={"objects": True})
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    # Pin the twist term's resampling window for the whole run.  The validator
    # writes commands directly, and with the play-config default (3.0, 8.0) s
    # the command manager re-rolls time_left at the first expiry: because
    # _place_the_robot pins rel_standing_envs = 1.0, the resample flips
    # is_standing_env True and zeroes vel_command_b on every step after
    # (velocity_command.py _update_command), silently cancelling the push or
    # backoff command mid-phase.  Measured before this pin (Apple M4, macOS 26.3,
    # native:cpu): time_left 4.90 s at reset -> 0.02 s at push start -> resample
    # on push step 1 -> command zero for the remaining 399 steps; 13/13 trials
    # whose command survived push step 1 pushed successfully.  The other
    # five_foot check tools (vel_check, twist_check, gait_check) pin the same.
    cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    push_box = PUSH_BOX
    if args.object == "push_box":
        # Validation-only free rigid body.  Keeping it out of SOLIDS avoids
        # changing the grasping row or the normal replay scene.
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
    grip_target = GRIPPER_OPEN if args.open_guide else GRIPPER_CLOSED
    twist = env.command_manager.get_term("twist")
    pose = env.command_manager.get_term("body_pose")
    solid = push_box if args.object == push_box.name else next(
        s for s in SOLIDS if s.name == args.object
    )
    half_x, half_y, half_z = solid.half_extents
    if args.orientation == "flat":
        if solid.kind == "cylinder":
            parser.error("--orientation flat is only supported for box-like props")
        # Rotate local +y onto world +z.  The former thin y dimension becomes
        # the support height while the broad x-z face rests on the floor.
        root_quat = torch.tensor(
            [[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]], device=res.device
        )
        support_half_height = half_y
        forward_half_extent = half_x
    elif args.orientation == "sideways":
        root_quat = torch.tensor(
            [[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]], device=res.device
        )
        support_half_height = half_z
        forward_half_extent = half_y
    else:
        root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=res.device)
        support_half_height = half_z
        forward_half_extent = half_x
    results: list[PushResult] = []
    recorded_qpos: list[np.ndarray] = []
    recording = False
    obs = None

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
        if args.open_guide:
            finger_target.fill_(GRIPPER_OPEN)
        else:
            finger_target.fill_(min(grip_target, float(angle[0, 0]) + SQUEEZE_LEAD))
        robot.set_joint_position_target(finger_target, joint_ids=[finger])

    def step(count: int) -> int:
        nonlocal obs
        contacts = 0
        # Contact-sensor buffers are reset in place between trials.  Creating a
        # lazily allocated buffer under inference_mode makes that later reset
        # illegal; no_grad still avoids the policy graph and keeps tensors mutable.
        with torch.no_grad():
            for _ in range(count):
                obs = env.observation_manager.compute()
                env.step(policy(obs))
                set_targets()
                contacts += int(contact_now())
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
        # Upright props stand on local z; flat boxes stand on local y after the
        # x-quarter-turn above.  Yaw is intentionally ignored.
        world_up_component = matrix[2, 1] if args.orientation == "flat" else matrix[2, 2]
        return math.degrees(
            math.acos(max(-1.0, min(1.0, float(world_up_component))))
        )

    for trial in range(1, args.trials + 1):
        env.reset()
        command(0.0)
        # The drop bin is fixed-base and remains at the far end of the row;
        # only the other free props need moving out of this contact test.
        other_props = tuple(name for name in ("soap", "notebook", "can") if name != args.object)
        for index, name in enumerate(other_props):
            _stash(env, name, res.device, index)
        # Walk the arm to the push pose with bounded adjacent targets.
        start = robot.data.joint_pos[0, arm].clone()
        goal = torch.tensor(push_pose, device=res.device)
        count = max(1, math.ceil(float((goal - start).abs().max()) / 0.01))
        for index in range(count):
            arm_target[0] = start + (goal - start) * ((index + 1) / count)
            step(1)
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
        step(25)
        start_object = prop.data.root_link_pos_w[0, :2].clone()
        start_base = robot.data.root_link_pos_w[0, :2].clone()
        initial_nonfoot = nonfoot_detail()

        recording = args.video is not None and trial == args.video_trial
        contacts = 0
        command(args.speed, args.lateral_speed)
        for _ in range(math.ceil(args.timeout_s / env.step_dt)):
            contacts += step(1)
            displacement = prop.data.root_link_pos_w[0, :2] - start_object
            if float(displacement.norm()) >= args.distance_m:
                break
        command(0.0)
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
        for index in range(count):
            arm_target[0] = begin + (retract - begin) * ((index + 1) / count)
            step(1)
        wait_stable()
        wait_object_stable()
        backoff_object_start = prop.data.root_link_pos_w[0, :2].clone()
        retract_motion = float((backoff_object_start - stop_object).norm())

        command(-args.speed, args.backoff_lateral_speed)
        for _ in range(math.ceil(args.timeout_s / env.step_dt)):
            step(1)
            backed = float(stop_base[0] - robot.data.root_link_pos_w[0, 0])
            if backed >= args.backoff_m:
                break
        command(0.0)
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
        result = PushResult(
            trial=trial,
            success=failure is None,
            failure=failure,
            contact_steps=contacts,
            object_forward_m=object_forward,
            object_lateral_m=object_lateral,
            object_planar_distance_m=object_distance,
            object_tilt_deg=object_tilt,
            base_forward_m=float(stop_base[0] - start_base[0]),
            stopped_stably=stopped,
            backed_away_m=backed,
            object_motion_during_retract_m=retract_motion,
            object_motion_during_backoff_m=drag,
            fell=did_fall,
            nonfoot_ground_contact=hazard is not None,
            nonfoot_contact=hazard,
        )
        results.append(result)
        print(
            f"trial {trial:02d}: {'ok' if result.success else 'FAILED ' + str(failure)}; "
            f"push={object_distance * 1000:.0f} mm "
            f"(x={object_forward * 1000:.0f}, y={object_lateral * 1000:.0f}) "
            f"tilt={object_tilt:.1f} deg backoff={backed * 1000:.0f} mm"
        )

    report = {
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
            "open_guide": args.open_guide,
            "backoff_m": args.backoff_m,
            "max_lateral_m": args.max_lateral_m,
        },
        "trials": [asdict(result) for result in results],
        "summary": {
            "successes": sum(result.success for result in results),
            "trials": len(results),
            "falls": sum(result.fell for result in results),
            "nonfoot_ground_contacts": sum(
                result.nonfoot_ground_contact for result in results
            ),
        },
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.json}")
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        _render_video(env.sim.mj_model, recorded_qpos, 1.0 / env.step_dt, args.video)
        print(f"video: {args.video}")
    env.close()
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
