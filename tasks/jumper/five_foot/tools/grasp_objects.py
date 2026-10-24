#!/usr/bin/env python3
"""End-to-end: can the claw actually pick up each object in the row?

Nothing is welded. The claw holds what it squeezes, by friction, under the row's
own friction settings (`objects.py::PROP_CONE`), and every part of that can be
right on its own while the robot still holds nothing -- the jaws' collision
geometry, where the prop sits in the mouth, whether the row's cone reached the
model, how hard the finger's servo squeezes. None of those fails loudly. This
drives the whole path per object and reports what happened, so "the claw picks
things up" is a measurement rather than a recollection.

    python tasks/jumper/five_foot/tools/grasp_objects.py
    python tasks/jumper/five_foot/tools/grasp_objects.py --backend native
    python tasks/jumper/five_foot/tools/grasp_objects.py --checkpoint <path>.pt
    mjpython tasks/jumper/five_foot/tools/grasp_objects.py --backend native \
        --grasp-ready --object notebook --video grasp.mp4

The task is built the way `play.py --objects` builds it, so what this checks is the
row an operator gets.

For each graspable prop it:

  1. opens the claw, puts the prop `HOLD_DEPTH` into the mouth and holds it
     there, then **lets go of it and closes the claw on it** -- never both at
     once: a prop held in place while the finger squeezes cannot yield, and the
     overlap throws it out when it is let go (`mdp/grasp.py::PLACE_STEPS`);
  2. raises the claw with the shoulder: **grasped** is a prop that came up with
     it, at least `FOLLOWED` of the way;
  3. swings the shoulder for two seconds and measures how far the prop moved
     **relative to the claw**: **carried** is under `CARRY_SLIP_MM`;
  4. opens the jaws: **released** is a prop back on the floor. A claw that cannot
     put anything down is broken too.

Then one negative control: a prop parked beside the claw, out of the mouth, let go
of and the claw closed, must **not** come up with the claw. Without
it every "grasped" above could be a prop resting on a jaw rather than squeezed
between them.

Last, a reset: a prop squeezed in the claw, the rest of the row out of the way,
and then `env.reset()` -- which has to bring every prop home, or restarting the
episode leaves half of the old one behind.

Before any of that it checks two things that fail silently in the least helpful
way. **Where the episode starts**: the robot on its spawn facing the row,
commanding nothing (`objects.py::START`) -- otherwise the replay opens with the
robot walking off on a sampled command. And **which friction cone the model runs**:
a row whose `PROP_CONE` never reached the model builds, runs, and drops everything
the claw picks up.

Runs on either backend. Run it from the repository root, like everything else here
-- the editable install is what puts `tasks` on the path, and nothing
in this repository manipulates `sys.path` to paper over a missing one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path

import numpy as np

#: How far a prop has to rise, as a share of how far the claw's anvil rose, to
#: have come up with the claw. A held prop sits further out along the claw than
#: the anvil, so it rises by more -- measured 94 to 120% -- and one left behind
#: rises by nothing.
FOLLOWED = 0.8

# A tall ground object need not rise one-for-one with the rotating shoulder:
# while its lower edge clears the floor it can pivot between the jaws.  The
# physical gate is therefore clearance under the object's primary collision
# shape, not an arbitrary fraction of the anvil's arc.
LIFT_CLEARANCE_M = 0.002

#: The negative control's ceiling on the same share: a prop beside the claw that
#: rises by a fifth of the claw has been carried by something.
NOT_FOLLOWED = 0.2

#: How far a carried prop may move relative to the claw over the swing, in mm.
#: Measured walking a carried prop back 0.6 m and turning it: 4.5 to 10.6 mm
#: (`tasks/jumper/five_foot/objects.py::PROP_CONE`).
CARRY_SLIP_MM = 20.0

#: How far above where it stood before the lift a released prop may still be, in
#: mm: back on the floor, upright or on its side.
RELEASED_MM = 5.0

#: Shoulder travel, in radians: the lift, and the swing about the lifted angle.
#: The swing is smaller than the lift so the claw never comes back down past where
#: it closed and puts the prop on the floor, which would read as slip.
LIFT_RAD = 0.30
SWING_RAD = 0.20

# A task-local candidate for the contact probe, not the task's carried-arm home.
# Re-derived on V1.6.1 from ``grasp_pose.py`` on 2026-10-06.  Full-mesh endpoint
# measurement: 24.6 mm forearm clearance, 56.0 mm floor clearance, 80 mm/rad
# shoulder lift and a 73.4 mm open mouth.  It remains a candidate until this
# script's contact, carry and release checks pass; do not deploy it as a preset.
GRASP_READY = (-0.87566764, -1.44724518, -0.84847052, 0.54775259)

STABLE_LINEAR_MPS = 0.02
STABLE_ANGULAR_RAD_S = 0.10
STABLE_TILT_RAD = math.radians(5.0)
OPEN_ERROR_RAD = 0.02
FINGER_STALL_RAD_S = 0.05
ALIGN_POSITION_M = 0.010
ALIGN_ANGLE_RAD = math.radians(10.0)
MIN_BILATERAL_STEPS = 3
MIN_LIFT_M = 0.005
FALL_TILT_RAD = math.radians(50.0)


class Phase(StrEnum):
    STABILIZE = "stabilize"
    OPEN = "open"
    ALIGN = "align"
    CLOSE = "close"
    LIFT = "lift"
    CARRY = "carry"
    RELEASE = "release"
    COMPLETE = "complete"


@dataclass
class TrialResult:
    trial: int
    object: str
    initial_offset_mm: tuple[float, float, float]
    success: bool
    failed_phase: str | None
    failure: str | None
    align_position_mm: float
    align_angle_deg: float
    moving_contact_steps: int
    fixed_contact_steps: int
    bilateral_contact_steps: int
    finger_speed_rad_s: float
    claw_lift_mm: float
    object_lift_mm: float
    floor_clearance_mm: float
    follow_ratio: float
    slip_mm: float
    base_carry_command_m: float
    base_carry_actual_m: float
    prelift_pull_actual_m: float
    prelift_rock_actual_m: float
    carry_stable: bool
    released: bool
    fell: bool
    nonfoot_ground_contact: bool
    nonfoot_contact: str | None


def _let_go(settle: int, step, grip: float) -> None:
    """Open the jaws, which is what drops an object.

    A helper rather than a bare `step(...)` because the release path is the one
    thing here with no second witness: nothing else in the run commands the claw
    open, so a claw that never let go would otherwise be reported as one that was
    never asked to.
    """
    step(settle, grip)


def _stash(env: object, name: str, device: object, index: int) -> None:
    """Put a finished prop out of reach, standing on the floor.

    On the floor rather than in the air, and each at its own spot: a prop left
    hanging above the claw falls back into the mouth, which is how the negative
    control below first reported a grab it had set up itself.
    """
    import torch

    angle = 2.0 * math.pi * index / 8.0 + 0.4
    entity = env.scene[name]
    pos = torch.tensor(
        [[3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.1]], device=device
    )
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=1))
    entity.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=device))


def _render_video(model: object, qpos: list[object], fps: float, path: Path) -> None:
    """Render recorded native-backend states to an H.264 MP4."""
    import imageio_ffmpeg
    import mujoco

    if sys.platform != "darwin":
        os.environ.setdefault("MUJOCO_GL", "egl")
    width, height = 1280, 720
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.08, 0.04, 0.10)
    camera.distance = 0.72
    camera.azimuth = 145.0
    camera.elevation = -18.0
    renderer = mujoco.Renderer(model, height=height, width=width)
    writer = imageio_ffmpeg.write_frames(
        str(path), size=(width, height), fps=fps, macro_block_size=1,
        codec="libx264", pix_fmt_out="yuv420p", quality=7,
    )
    writer.send(None)
    try:
        for state in qpos:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.send(renderer.render().tobytes())
    finally:
        renderer.close()
        writer.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="jumper.five_foot")
    p.add_argument("--backend", choices=("warp", "native"), default="warp")
    p.add_argument("--checkpoint", default=None,
                   help="a policy to hold the robot up; without one it stands on "
                        "its reset pose, which is enough to test the claw")
    p.add_argument("--seed", type=int, default=42,
                   help="simulation seed; fixed by default so checkpoint comparisons "
                        "and rendered evidence are reproducible")
    p.add_argument("--grasp-ready", action="store_true",
                   help="move all four carried-arm joints to the measured grasp-ready "
                        "candidate before testing contact")
    p.add_argument("--grasp-ready-shoulder-offset-rad", type=float, default=0.0,
                   help="experimental shoulder offset applied to the grasp-ready pose")
    p.add_argument("--depth-mm", type=float, default=20.0,
                   help="how far the prop is seated into the mouth before closing")
    p.add_argument("--object", choices=("soap", "notebook", "can"), default=None,
                   help="test one prop instead of the whole row")
    p.add_argument("--object-pitch-deg", type=float, default=0.0,
                   help="pitch the fixed-pose prop before closing (degrees)")
    p.add_argument("--trials", type=int, default=1,
                   help="repeat each selected fixed-pose grasp this many times")
    p.add_argument("--grid-span-mm", type=float, default=0.0,
                   help="test a 3x3x3 XYZ grid at -span, 0, +span; this replaces "
                        "--trials with 27 trials")
    p.add_argument("--center-offset-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="world-XYZ offset of the validation grid centre")
    p.add_argument("--json", type=Path, default=None,
                   help="write machine-readable trial results to this path")
    p.add_argument("--video", type=Path, default=None,
                   help="record one trial to an H.264 MuJoCo MP4 (native only)")
    p.add_argument("--video-trial", type=int, default=1,
                   help="one-based trial to record when --video is set")
    p.add_argument("--base-carry-m", type=float, default=0.0,
                   help="after lifting, walk forward this many metres before releasing")
    p.add_argument("--carry-speed", type=float, default=0.10,
                   help="forward command in m/s for --base-carry-m")
    p.add_argument("--carry-direction", choices=("forward", "backward"),
                   default="forward",
                   help="direction of the policy-driven base carry")
    p.add_argument("--carry-pitch-deg", type=float, default=0.0,
                   help="body-pitch command during base carry; positive is nose down")
    p.add_argument("--carry-timeout-s", type=float, default=8.0,
                   help="maximum walking time for --base-carry-m")
    p.add_argument("--carry-ramp-s", type=float, default=0.0,
                   help="linearly ramp the forward command over this many seconds")
    p.add_argument("--carry-start-speed", type=float, default=None,
                   help="optional short start pulse in m/s after lifting; the command "
                        "then returns to --carry-speed")
    p.add_argument("--carry-start-s", type=float, default=0.0,
                   help="duration of --carry-start-speed in seconds")
    p.add_argument("--prelift-pull-m", type=float, default=0.0,
                   help="policy-driven backward pull after closing and before lifting")
    p.add_argument("--prelift-rock-m", type=float, default=0.0,
                   help="policy-driven forward rock after the backward pre-lift pull")
    p.add_argument("--settle", type=int, default=60)
    p.add_argument("--close", type=int, default=80)
    p.add_argument("--lift", type=int, default=50)
    p.add_argument("--lift-rad", type=float, default=LIFT_RAD,
                   help="shoulder travel used to lift the closed claw")
    p.add_argument("--sweep", type=int, default=100)
    args = p.parse_args()
    if args.trials < 1:
        p.error("--trials must be at least 1")
    if args.video_trial < 1:
        p.error("--video-trial must be at least 1")
    if args.grid_span_mm < 0:
        p.error("--grid-span-mm must be non-negative")
    if args.video is not None and args.backend != "native":
        p.error("--video currently requires --backend native")
    if (args.base_carry_m < 0 or args.prelift_pull_m < 0
            or args.prelift_rock_m < 0 or args.carry_speed <= 0
            or args.carry_timeout_s <= 0 or args.carry_ramp_s < 0
            or args.carry_start_s < 0
            or (args.carry_start_speed is not None
                and args.carry_start_speed <= 0)):
        p.error("carry distance must be non-negative; speed and timeout must be positive")
    if (args.carry_start_speed is None) != (args.carry_start_s == 0.0):
        p.error("--carry-start-speed and a positive --carry-start-s must be used together")
    if args.lift_rad <= 0:
        p.error("--lift-rad must be positive")
    if args.base_carry_m and args.checkpoint is None:
        p.error("--base-carry-m requires --checkpoint; scripted root motion is not allowed")

    import warnings

    warnings.filterwarnings("ignore")
    import mujoco
    import torch
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend=args.backend,
                  device="cuda" if args.backend == "warp" else "cpu", num_envs=1)
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv

    from tasks.jumper.five_foot.claw import (
        ARM_JOINTS,
        FINGER_JOINT,
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
        SQUEEZE_LEAD,
    )
    from tasks.jumper.five_foot.mdp.grasp import Claw, _rotate
    from tasks.jumper.five_foot.objects import (
        GRASPABLE,
        PROP_CONE,
        PROP_IMPRATIO,
        SOLIDS,
    )

    cfg = tasks.load_env_cfg(args.task, play=True, task_args={"objects": True})
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    moving_geoms = (
        "LF_finger_link_meshcol", "LF_finger_link_meshcol1",
        "LF_finger_link_meshcol2", "LF_finger_tip_link_meshcol",
        "LF_finger_grip_insert_link_meshcol",
    )
    fixed_geoms = (
        "LF_palm_link_meshcol", "LF_palm_link_meshcol1", "LF_palm_link_meshcol2",
        "LF_palm_link_meshcol3", "LF_palm_link_meshcol4", "LF_palm_link_meshcol5",
        "LF_palm_pad_f_link_meshcol", "LF_palm_grip_insert_link_meshcol",
    )
    grasp_sensors = []
    for prop in ("soap", "notebook", "can"):
        for side, geoms in (("moving", moving_geoms), ("fixed", fixed_geoms)):
            grasp_sensors.append(ContactSensorCfg(
                name=f"grasp_{prop}_{side}",
                primary=ContactMatch(mode="geom", pattern=geoms, entity="robot"),
                secondary=ContactMatch(mode="subtree", pattern=prop, entity=prop),
                fields=("found", "force"),
                reduce="maxforce",
                num_slots=1,
                history_length=cfg.decimation,
            ))
    cfg.scene.sensors = (cfg.scene.sensors or ()) + tuple(grasp_sensors)
    cfg.sim.contact_sensor_maxmatch = max(int(cfg.sim.contact_sensor_maxmatch or 0), 512)
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)

    policy = None
    if args.checkpoint:
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.rl.runner import MjlabOnPolicyRunner

        agent = tasks.load_agent_cfg(args.task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = (tasks.load_runner_cls(args.task) or MjlabOnPolicyRunner)(
            wrapped, asdict(agent), device=res.device
        )
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=res.device)
        policy = runner.get_inference_policy(device=res.device)

    props = (args.object,) if args.object is not None else GRASPABLE
    if not props:
        print("no graspable props in the row; nothing to check")
        return 1

    robot = env.scene["robot"]
    finger = robot.find_joints([FINGER_JOINT])[0][0]
    arm = robot.find_joints(list(ARM_JOINTS))[0]
    shoulder = arm[1]
    grasp_ready = list(GRASP_READY)
    grasp_ready[1] += args.grasp_ready_shoulder_offset_rad
    arm_target = torch.tensor([grasp_ready], device=res.device)
    object_pitch = math.radians(args.object_pitch_deg)
    object_quat = torch.tensor(
        [[math.cos(object_pitch / 2), 0.0, math.sin(object_pitch / 2), 0.0]],
        device=res.device,
    )
    claw = Claw(env)
    zero = torch.zeros(env.action_space.shape, device=res.device)
    one = torch.zeros(1, 1, device=res.device)
    obs = None
    contact_counts: dict[str, dict[str, int]] = {}
    trial_hazards: dict[str, Phase | None] | None = None
    current_phase: Phase | None = None
    recorded_qpos: list[object] = []
    recording = False
    twist_term = env.command_manager.get_term("twist")
    pose_term = env.command_manager.get_term("body_pose")

    def command_twist(vx: float) -> None:
        """Pin the replay command without moving or rewriting the simulated base."""
        twist_term.cfg.ranges.lin_vel_x = (vx, vx)
        twist_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        twist_term.cfg.ranges.ang_vel_z = (0.0, 0.0)
        twist_term.is_standing_env[:] = False
        twist_term.is_heading_env[:] = False
        twist_term.is_world_env[:] = False
        twist_term.vel_command_b[:] = torch.tensor(
            [vx, 0.0, 0.0], device=res.device
        )
        twist_term.vel_command_w[:] = twist_term.vel_command_b

    def command_pitch(degrees: float) -> None:
        """Set the existing body-pose command; physics still moves the robot."""
        pose_term.is_neutral_env[:] = False
        pose_term.pose_target_b[:, 0] = math.radians(degrees)

    def step(n: int, grip: float, park: str | None = None,
             offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
             shoulder_at: float | None = None) -> None:
        nonlocal obs
        with torch.inference_mode():
            for _ in range(n):
                if policy is None:
                    env.step(zero)
                else:
                    # The whole dict, not one group: the model concatenates the
                    # groups its own config names (`obs_groups`), so handing it a
                    # single tensor indexes a 2-d tensor with a string.
                    obs = env.observation_manager.compute()
                    env.step(policy(obs))
                if recording:
                    _, live_data = env.sim.env_mjdata(0)
                    recorded_qpos.append(live_data.qpos.copy())
                for contact_name, counts in contact_counts.items():
                    moving = jaw_contact(contact_name, "moving")
                    fixed = jaw_contact(contact_name, "fixed")
                    counts["moving"] += int(moving)
                    counts["fixed"] += int(fixed)
                    counts["both"] += int(moving and fixed)
                if trial_hazards is not None:
                    phase = current_phase or Phase.STABILIZE
                    if fell() and trial_hazards["fell"] is None:
                        trial_hazards["fell"] = phase
                    if (nonfoot_ground_contact()
                            and trial_hazards["nonfoot"] is None):
                        trial_hazards["nonfoot"] = phase
                        trial_hazards["nonfoot_detail"] = nonfoot_contact_detail()
                # Squeezed as the operator squeezes and no harder
                # (`claw.py::SQUEEZE_LEAD`), so a claw that only holds at the
                # servo's full torque fails here rather than in replay.
                one.fill_(min(grip, float(robot.data.joint_pos[0, finger]) + SQUEEZE_LEAD))
                robot.set_joint_position_target(one, joint_ids=[finger])
                if shoulder_at is not None:
                    arm_target[0, 1] = shoulder_at
                if args.grasp_ready:
                    robot.set_joint_position_target(arm_target, joint_ids=arm)
                elif shoulder_at is not None:
                    one_s = torch.full((1, 1), shoulder_at, device=res.device)
                    robot.set_joint_position_target(one_s, joint_ids=[shoulder])
                if park is not None:
                    # Held in place while the claw is open, so it is where the
                    # seat says when the jaws start to close -- and only then:
                    # `mdp/grasp.py::PLACE_STEPS` has what holding it through
                    # the squeeze does.
                    entity = env.scene[park]
                    pos = claw.seat(
                        env, park, depth=args.depth_mm / 1000.0, quat=object_quat
                    ) \
                        + torch.tensor([offset], device=res.device)
                    entity.write_root_link_pose_to_sim(
                        torch.cat([pos, object_quat], dim=1)
                    )
                    entity.write_root_com_velocity_to_sim(
                        torch.zeros(1, 6, device=res.device)
                    )

    def height(name: str) -> float:
        return float(env.scene[name].data.root_link_pos_w[0, 2])

    def floor_clearance(name: str) -> float:
        """Clearance below the prop's primary solid, in world metres."""
        solid = next(item for item in SOLIDS if item.name == name)
        quat = env.scene[name].data.root_link_quat_w[0].detach().cpu().numpy()
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        vertical = np.abs(rotation.reshape(3, 3)[2])
        if solid.kind in ("box", "hull"):
            extent = float(vertical @ np.asarray(solid.size))
        elif solid.kind == "cylinder":
            radius, half_height = solid.size
            extent = float(radius * np.hypot(vertical[0], vertical[1])
                           + half_height * vertical[2])
        else:
            raise ValueError(f"floor clearance is not implemented for {solid.kind!r}")
        floor = float(env.scene.env_origins[0, 2])
        return height(name) - extent - floor

    def anvil_height() -> float:
        return float(claw.mouth()[0, 2])

    def in_claw(name: str) -> torch.Tensor:
        """Where `name` is relative to the anvil, in the palm's frame."""
        inverse = robot.data.body_link_quat_w[:, claw.palm].clone()
        inverse[:, 1:] = -inverse[:, 1:]
        return _rotate(inverse, env.scene[name].data.root_link_pos_w - claw.mouth())[0]

    def jaw_contact(name: str, side: str) -> bool:
        found = env.scene.sensors[f"grasp_{name}_{side}"].data.found
        assert found is not None
        return bool(found[0].sum() > 0)

    def tilt() -> tuple[float, float]:
        gravity = robot.data.projected_gravity_b[0]
        pitch = math.asin(max(-1.0, min(1.0, float(gravity[0]))))
        roll = math.atan2(float(-gravity[1]), float(-gravity[2]))
        return pitch, roll

    def fell() -> bool:
        pitch, roll = tilt()
        base_z = float(robot.data.root_link_pos_w[0, 2] - env.scene.env_origins[0, 2])
        return base_z < 0.05 or max(abs(pitch), abs(roll)) > FALL_TILT_RAD

    def nonfoot_ground_contact() -> bool:
        try:
            sensor = env.scene.sensors["non_foot_ground_contact"]
        except KeyError:
            return False
        if sensor.data.force is None:
            return False
        return bool(sensor.data.force[0].norm(dim=-1).max() > 1.0)

    def nonfoot_contact_detail() -> str | None:
        """Name the strongest non-foot/terrain sensor match."""
        sensor = env.scene.sensors["non_foot_ground_contact"]
        if sensor.data.force is None:
            return None
        magnitudes = sensor.data.force[0].norm(dim=-1)
        index = int(magnitudes.argmax())
        force = float(magnitudes[index])
        primary = sensor.primary_names[index // sensor.cfg.num_slots]
        return f"{primary} against terrain ({force:.2f} N)"

    def stable_now() -> bool:
        pitch, roll = tilt()
        linear = float(robot.data.root_link_lin_vel_b[0, :2].norm())
        angular = float(robot.data.root_link_ang_vel_b[0].norm())
        return (
            linear < STABLE_LINEAR_MPS
            and angular < STABLE_ANGULAR_RAD_S
            and max(abs(pitch), abs(roll)) < STABLE_TILT_RAD
            and not fell()
        )

    def move_arm_safely(target: tuple[float, ...], grip: float) -> None:
        """Command a joint-space path whose adjacent targets differ by <= 0.01 rad."""
        start = robot.data.joint_pos[0, arm].clone()
        goal = torch.tensor(target, device=res.device)
        count = max(1, math.ceil(float((goal - start).abs().max()) / 0.01))
        for k in range(count):
            arm_target[0] = start + (goal - start) * ((k + 1) / count)
            step(1, grip)

    def wait_stable(grip: float, consecutive: int = 25) -> bool:
        good = 0
        for _ in range(args.settle):
            step(1, grip)
            good = good + 1 if stable_now() else 0
            if good >= consecutive:
                return True
        return False

    def lift(name: str, rest: float, grip: float) -> tuple[float, float]:
        """Raise the claw by `LIFT_RAD`: (how far `name` rose, how far the anvil did)."""
        z0, a0 = height(name), anvil_height()
        half = max(1, args.lift // 2)
        for k in range(half):
            step(1, grip, shoulder_at=rest + raise_by * (k + 1) / half)
        step(args.lift - half, grip, shoulder_at=rest + raise_by)
        return height(name) - z0, anvil_height() - a0

    print(f"--objects  backend={args.backend}  arm="
          f"{'grasp-ready' if args.grasp_ready else 'stow'}  props={', '.join(props)}\n")
    failures = 0

    # ── The friction the props are held under ────────────────────────────
    model = env.sim.mj_model
    cone = "elliptic" if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal"
    wanted = (PROP_CONE, float(PROP_IMPRATIO))
    same = (cone, float(model.opt.impratio)) == wanted
    failures += not same
    print(f"friction       the model runs the {cone} cone at impratio "
          f"{model.opt.impratio:g}; the row asks for {wanted[0]} at {wanted[1]:g}   "
          f"{'ok' if same else '<- NOT what the row asked for'}")

    # ── Where the episode starts ─────────────────────────────────────────
    # Read straight after a reset, which is the only moment the spawn is visible:
    # a few steps later the robot has settled, slumped or walked, and none of
    # those is distinguishable from having started somewhere else.
    env.reset()
    origin = env.scene.env_origins[0]
    start = (robot.data.root_link_pos_w[0] - origin).clone()
    yaw = float(torch.atan2(
        2.0 * (robot.data.root_link_quat_w[0, 0] * robot.data.root_link_quat_w[0, 3]
               + robot.data.root_link_quat_w[0, 1] * robot.data.root_link_quat_w[0, 2]),
        1.0 - 2.0 * (robot.data.root_link_quat_w[0, 2] ** 2
                     + robot.data.root_link_quat_w[0, 3] ** 2),
    ))
    placed = abs(float(start[0])) < 0.01 and abs(float(start[1])) < 0.01 \
        and abs(yaw) < 0.02
    failures += not placed
    print(f"start          robot at ({float(start[0]):+.3f}, {float(start[1]):+.3f}) "
          f"m, yaw {math.degrees(yaw):+.1f} deg   "
          f"{'on its spawn, facing the row' if placed else '<- NOT the spawn'}")

    # The command, over a couple of seconds of stepping rather than at t=0: the
    # sampler resamples on its own schedule, so one reading proves nothing about
    # whether the robot is going to stay put.
    worst = 0.0
    for _ in range(args.settle):
        step(1, GRIPPER_OPEN)
        for cmd_name in env.command_manager.active_terms:
            worst = max(worst, float(
                env.command_manager.get_command(cmd_name)[0].abs().max()
            ))
    still = worst < 1e-6
    failures += not still
    drift = float((robot.data.root_link_pos_w[0, :2] - origin[:2] - start[:2]).norm())
    print(f"command        largest commanded value over {args.settle} steps: "
          f"{worst:.3f}   {'nothing commanded' if still else '<- NOT still'}")
    print(f"drift          {drift * 1000:.0f} mm in that time "
          f"{'(no policy: the robot is holding its reset pose)' if policy is None else ''}")

    # Home, for the reset check at the end. Read **after** a reset rather than at
    # construction: mjlab applies `env_origins` inside the prop reset event, so
    # until one has run every prop reads as being at the world origin.
    home = {n: env.scene[n].data.root_link_pos_w[0].clone() for n in props}

    # And where the row is from here, which is what "a little in front of the
    # objects" has to mean in metres.
    here = claw.mouth()[0]
    for name in props:
        to = env.scene[name].data.root_link_pos_w[0] - here
        print(f"  {name:<11} {float(to[0]) * 1000:+5.0f} mm ahead of the mouth, "
              f"{float(to[1]) * 1000:+5.0f} mm across, {float(to[2]) * 1000:+5.0f} mm up")

    # Which way the shoulder raises the claw, asked of the robot rather than
    # assumed: the sign depends on how the joint is mounted.
    if args.grasp_ready:
        move_arm_safely(tuple(grasp_ready), GRIPPER_OPEN)
        step(args.settle, GRIPPER_OPEN)
        rest = grasp_ready[1]
    else:
        rest = float(robot.data.joint_pos[0, shoulder])
    before = anvil_height()
    step(40, GRIPPER_OPEN, shoulder_at=rest - args.lift_rad)
    raise_by = -args.lift_rad if anvil_height() > before else args.lift_rad
    step(40, GRIPPER_OPEN, shoulder_at=rest)
    print(f"  the shoulder raises the claw at {raise_by:+.2f} rad; props go "
          f"{args.depth_mm:.0f} mm into the mouth\n")

    print(f"{'trial/object':<18}{'contact':>13}{'follow':>10}{'slip':>10}   verdict")
    results: list[TrialResult] = []
    if args.grid_span_mm:
        grid = (-args.grid_span_mm / 1000.0, 0.0, args.grid_span_mm / 1000.0)
        initial_offsets = list(product(grid, repeat=3))
    else:
        initial_offsets = [(0.0, 0.0, 0.0)] * args.trials
    centre = tuple(value / 1000.0 for value in args.center_offset_mm)
    initial_offsets = [
        tuple(offset[axis] + centre[axis] for axis in range(3))
        for offset in initial_offsets
    ]
    for i, name in enumerate(props):
        for trial, initial_offset in enumerate(initial_offsets, start=1):
            # Every grid point is an independent experiment.  Without a reset
            # the policy's base drift and gait phase accumulate across points,
            # so the later offsets are not trials of the stated initial pose.
            with torch.inference_mode():
                env.reset()
            command_twist(0.0)
            command_pitch(0.0)
            if args.grasp_ready:
                move_arm_safely(tuple(grasp_ready), GRIPPER_OPEN)
                step(args.settle, GRIPPER_OPEN)
            recording = (
                args.video is not None
                and trial == args.video_trial
                and not recorded_qpos
            )
            trial_hazards = {
                "fell": None,
                "nonfoot": None,
                "nonfoot_detail": None,
            }
            current_phase = Phase.STABILIZE
            stable = wait_stable(GRIPPER_OPEN)
            open_error = abs(float(robot.data.joint_pos[0, finger]) - GRIPPER_OPEN)
            open_speed = abs(float(robot.data.joint_vel[0, finger]))
            opened = open_error < OPEN_ERROR_RAD and open_speed < FINGER_STALL_RAD_S

            # ALIGN is deliberately still a known-pose fixture.  It proves the
            # contact sequence independently of perception and locomotion.
            current_phase = Phase.ALIGN
            step(args.settle, GRIPPER_OPEN, park=name, offset=initial_offset,
                 shoulder_at=rest)
            # Compare against the current claw pose after settling, not the pose
            # before normal body/arm settling moved the whole fixture.
            desired = claw.seat(
                env, name, depth=args.depth_mm / 1000.0, quat=object_quat
            )[0].clone()
            desired += torch.tensor(initial_offset, device=res.device)
            align_mm = float((env.scene[name].data.root_link_pos_w[0] - desired).norm()) * 1000
            quat_dot = abs(float(
                (env.scene[name].data.root_link_quat_w[0] * object_quat[0]).sum()
            ))
            align_angle = 2.0 * math.acos(max(-1.0, min(1.0, quat_dot)))
            aligned = align_mm < ALIGN_POSITION_M * 1000 and align_angle < ALIGN_ANGLE_RAD

            current_phase = Phase.CLOSE
            contact_counts[name] = {"moving": 0, "fixed": 0, "both": 0}
            step(args.close, GRIPPER_CLOSED, shoulder_at=rest)
            contacts_seen = contact_counts.pop(name)
            finger_speed = abs(float(robot.data.joint_vel[0, finger]))
            closed_on_object = (
                contacts_seen["both"] >= MIN_BILATERAL_STEPS
                or finger_speed < FINGER_STALL_RAD_S
            )

            current_phase = Phase.LIFT
            prelift_pull_actual = 0.0
            prelift_rock_actual = 0.0
            if args.prelift_pull_m:
                pull_start = robot.data.root_link_pos_w[0, :2].clone()
                command_twist(-args.carry_speed)
                pull_limit = math.ceil(min(3.0, args.carry_timeout_s) / env.step_dt)
                for _ in range(pull_limit):
                    step(1, GRIPPER_CLOSED, shoulder_at=rest)
                    prelift_pull_actual = float(
                        (robot.data.root_link_pos_w[0, :2] - pull_start).norm()
                    )
                    if prelift_pull_actual >= args.prelift_pull_m:
                        break
                command_twist(0.0)
            if args.prelift_rock_m:
                rock_start = robot.data.root_link_pos_w[0, :2].clone()
                command_twist(args.carry_speed)
                rock_limit = math.ceil(min(3.0, args.carry_timeout_s) / env.step_dt)
                for _ in range(rock_limit):
                    step(1, GRIPPER_CLOSED, shoulder_at=rest)
                    prelift_rock_actual = float(
                        (robot.data.root_link_pos_w[0, :2] - rock_start).norm()
                    )
                    if prelift_rock_actual >= args.prelift_rock_m:
                        break
                command_twist(0.0)
            z_before = height(name)
            rose, claw_rose = lift(name, rest, GRIPPER_CLOSED)
            share = rose / claw_rose if claw_rose > 1e-4 else 0.0
            clearance = floor_clearance(name)
            grasped = (
                claw_rose > MIN_LIFT_M
                and rose > MIN_LIFT_M
                and clearance >= LIFT_CLEARANCE_M
                and closed_on_object
            )

            # CARRY: either walk under the loaded five-foot policy or, for the
            # contact-only probe, swing the arm.  Neither path moves the root by
            # writing state; base travel comes only from the policy and contacts.
            current_phase = Phase.CARRY
            top = rest + raise_by
            was = in_claw(name).clone()
            base_carry_actual = 0.0
            carry_stable = True
            if args.base_carry_m and grasped:
                base_start = robot.data.root_link_pos_w[0, :2].clone()
                command_pitch(args.carry_pitch_deg)
                limit = math.ceil(args.carry_timeout_s / env.step_dt)
                ramp_steps = math.ceil(args.carry_ramp_s / env.step_dt)
                start_steps = math.ceil(args.carry_start_s / env.step_dt)
                direction = 1.0 if args.carry_direction == "forward" else -1.0
                for carry_step in range(limit):
                    if carry_step < start_steps:
                        speed = args.carry_start_speed
                    else:
                        nominal_step = carry_step - start_steps
                        scale = min(1.0, (nominal_step + 1) / max(1, ramp_steps))
                        speed = args.carry_speed * scale
                    command_twist(direction * speed)
                    step(1, GRIPPER_CLOSED, shoulder_at=top)
                    base_carry_actual = float(
                        (robot.data.root_link_pos_w[0, :2] - base_start).norm()
                    )
                    if base_carry_actual >= args.base_carry_m:
                        break
                command_twist(0.0)
                command_pitch(0.0)
                carry_stable = wait_stable(GRIPPER_CLOSED)
            elif not args.base_carry_m:
                for k in range(args.sweep):
                    step(1, GRIPPER_CLOSED,
                         shoulder_at=top + SWING_RAD * math.sin(k / 16.0))
            slip_mm = float((in_claw(name) - was).norm()) * 1000.0
            reached = (
                not args.base_carry_m
                or base_carry_actual >= 0.9 * args.base_carry_m
            )
            carried = grasped and slip_mm < CARRY_SLIP_MM and reached and carry_stable

            current_phase = Phase.RELEASE
            _let_go(args.settle, step, GRIPPER_OPEN)
            recording = False
            released = (height(name) - z_before) * 1000.0 < RELEASED_MM

            checks = (
                (Phase.STABILIZE, stable, "stability_timeout"),
                (Phase.OPEN, opened, "gripper_not_open_or_moving"),
                (Phase.ALIGN, aligned, "fixture_alignment_error"),
                (Phase.CLOSE, closed_on_object, "no_bilateral_contact_and_finger_not_stalled"),
                (Phase.LIFT, grasped, "object_did_not_follow_lift"),
                (Phase.CARRY, carried, "relative_slip"),
                (Phase.RELEASE, released, "object_not_released"),
            )
            failed_phase = next((phase for phase, ok, _ in checks if not ok), None)
            failure = next((reason for _, ok, reason in checks if not ok), None)
            phase_order = {phase: index for index, phase in enumerate(Phase)}
            hazards = (
                (trial_hazards["fell"], "fell_over_or_too_low"),
                (trial_hazards["nonfoot"], "nonfoot_ground_contact"),
            )
            hazard_phase, hazard_reason = min(
                ((phase, reason) for phase, reason in hazards if phase is not None),
                key=lambda item: phase_order[item[0]],
                default=(None, None),
            )
            if (hazard_phase is not None
                    and (failed_phase is None
                         or phase_order[hazard_phase] < phase_order[failed_phase])):
                failed_phase, failure = hazard_phase, hazard_reason
            ok = failed_phase is None
            failures += not ok
            result = TrialResult(
                trial=trial, object=name,
                initial_offset_mm=tuple(value * 1000 for value in initial_offset),
                success=ok,
                failed_phase=str(failed_phase) if failed_phase else None,
                failure=failure, align_position_mm=align_mm,
                align_angle_deg=math.degrees(align_angle),
                moving_contact_steps=contacts_seen["moving"],
                fixed_contact_steps=contacts_seen["fixed"],
                bilateral_contact_steps=contacts_seen["both"],
                finger_speed_rad_s=finger_speed, claw_lift_mm=claw_rose * 1000,
                object_lift_mm=rose * 1000,
                floor_clearance_mm=clearance * 1000,
                follow_ratio=share, slip_mm=slip_mm, released=released,
                base_carry_command_m=args.base_carry_m,
                base_carry_actual_m=base_carry_actual,
                prelift_pull_actual_m=prelift_pull_actual,
                prelift_rock_actual_m=prelift_rock_actual,
                carry_stable=carry_stable,
                fell=trial_hazards["fell"] is not None,
                nonfoot_ground_contact=trial_hazards["nonfoot"] is not None,
                nonfoot_contact=trial_hazards["nonfoot_detail"],
            )
            results.append(result)
            verdict = "ok" if ok else f"FAILED {failed_phase}: {failure}"
            carry_text = (
                f" carry={base_carry_actual:.3f}m" if args.base_carry_m else ""
            )
            print(f"{trial:02d}/{name:<14}"
                  f"{contacts_seen['both']:>8} both"
                  f"{share:>10.0%}{slip_mm:>9.1f} mm"
                  f" clear={clearance * 1000:.1f}mm   "
                  f"{verdict}{carry_text}")
            if result.nonfoot_contact is not None:
                print(f"  contact       {result.nonfoot_contact}")
            trial_hazards = None
            current_phase = None
            step(args.settle, GRIPPER_OPEN, shoulder_at=rest)
        _stash(env, name, res.device, i)

    successes = sum(result.success for result in results)
    label = "position-grid" if args.grid_span_mm else "fixed-point"
    print(f"\n{label:<14}{successes}/{len(results)} successful; "
          f"falls={sum(r.fell for r in results)}, "
          f"nonfoot_ground={sum(r.nonfoot_ground_contact for r in results)}")

    # Negative control: beside the mouth, held while the claw closes, never in it.
    decoy = props[0]
    aside = (0.12, 0.0, 0.0)
    step(args.settle, GRIPPER_OPEN, park=decoy, offset=aside, shoulder_at=rest)
    step(args.close, GRIPPER_CLOSED, shoulder_at=rest)
    rose, claw_rose = lift(decoy, rest, GRIPPER_CLOSED)
    share = rose / claw_rose if claw_rose > 1e-4 else 0.0
    caught = share >= NOT_FOLLOWED
    failures += caught
    print(f"\n{'(control)':<12}{decoy} parked 120 mm aside, claw shut and raised: it rose "
          f"{share:.0%} of the claw"
          f"{'  <- the checks above prove nothing' if caught else ', as it should'}")
    _let_go(args.settle, step, GRIPPER_OPEN)
    step(args.settle, GRIPPER_OPEN, shoulder_at=rest)

    # ── A reset: the row comes back ──────────────────────────────────────
    # Every prop is 3 m away on the floor by now and one is about to be squeezed
    # in the claw, so a reset has both halves to undo. Run **inside inference
    # mode**, the way a replay loop steps: the managers' buffers are then inference
    # tensors, and resetting one from outside raises `Inplace update to inference
    # tensor outside InferenceMode` instead of restarting.
    step(args.settle, GRIPPER_OPEN, park=decoy, offset=centre, shoulder_at=rest)
    step(args.close, GRIPPER_CLOSED, shoulder_at=rest)
    holding = float(in_claw(decoy).norm()) < 0.10
    with torch.inference_mode():
        env.reset()
    step(1, GRIPPER_OPEN)
    worst, where = 0.0, ""
    for name in props:
        moved = float((env.scene[name].data.root_link_pos_w[0] - home[name]).norm())
        if moved > worst:
            worst, where = moved, name
    reset_ok = holding and worst < 0.001
    failures += not reset_ok
    print(f"\n(reset)     holding {'yes' if holding else 'NO'}, then reset: "
          f"props back to within {worst * 1000:.1f} mm ({where})   "
          f"{'ok' if reset_ok else 'FAILED'}")

    if args.json is not None:
        solids = {solid.name: solid for solid in SOLIDS}
        report = {
            "task": args.task,
        "backend": args.backend,
            "seed": args.seed,
            "video_trial": args.video_trial if args.video is not None else None,
        "checkpoint": args.checkpoint,
            "control": "policy" if policy is not None else "zero action / joint targets",
            "arm_pose_rad": grasp_ready if args.grasp_ready else None,
            "depth_mm": args.depth_mm,
            "object_pitch_deg": args.object_pitch_deg,
            "grid_span_mm": args.grid_span_mm,
            "center_offset_mm": list(args.center_offset_mm),
            "base_carry_m": args.base_carry_m,
            "carry_speed_mps": args.carry_speed,
            "carry_direction": args.carry_direction,
            "carry_pitch_deg": args.carry_pitch_deg,
            "carry_timeout_s": args.carry_timeout_s,
            "carry_ramp_s": args.carry_ramp_s,
            "carry_start_speed_mps": args.carry_start_speed,
            "carry_start_s": args.carry_start_s,
            "prelift_pull_m": args.prelift_pull_m,
            "prelift_rock_m": args.prelift_rock_m,
            "simulation": {
                "physics_dt": float(model.opt.timestep),
                "control_dt": float(model.opt.timestep) * cfg.decimation,
                "cone": cone,
                "impratio": float(model.opt.impratio),
            },
            "objects": {
                name: {
                    "kind": solids[name].kind,
                    "size_parameter_m": list(solids[name].size),
                    "mass_kg": solids[name].mass,
                }
                for name in props
            },
            "thresholds": {
                "stable_linear_mps": STABLE_LINEAR_MPS,
                "stable_angular_rad_s": STABLE_ANGULAR_RAD_S,
                "stable_tilt_deg": math.degrees(STABLE_TILT_RAD),
                "align_position_mm": ALIGN_POSITION_M * 1000,
                "align_angle_deg": math.degrees(ALIGN_ANGLE_RAD),
                "minimum_bilateral_steps": MIN_BILATERAL_STEPS,
                "minimum_lift_mm": MIN_LIFT_M * 1000,
                "minimum_floor_clearance_mm": LIFT_CLEARANCE_M * 1000,
                "follow_ratio_diagnostic_reference": FOLLOWED,
                "maximum_slip_mm": CARRY_SLIP_MM,
                "release_height_mm": RELEASED_MM,
            },
            "summary": {
                "successes": successes,
                "trials": len(results),
                "falls": sum(result.fell for result in results),
                "nonfoot_ground_contacts": sum(
                    result.nonfoot_ground_contact for result in results
                ),
            },
            "trials": [asdict(result) for result in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report         {args.json}")

    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        _render_video(
            env.sim.render_model, recorded_qpos,
            fps=1.0 / (float(model.opt.timestep) * cfg.decimation), path=args.video,
        )
        print(f"video          {args.video} ({len(recorded_qpos)} frames)")

    env.close()
    print(f"\n{'FAILED' if failures else 'all objects pick up, carry and release'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
