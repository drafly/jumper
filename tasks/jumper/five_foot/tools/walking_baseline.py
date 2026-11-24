#!/usr/bin/env python3
"""Walking baseline: 20 independent episodes per fixed command.

The training dashboard is a reward, and `vel_check.py` measures a steady-state
mean over locked environments. This is the gate the handoff needs instead:
per-condition fall counts, displacement and raw tracking error over 20
independent reset episodes, with both command terms pinned so neither resamples
under the measurement.

    python tasks/jumper/five_foot/tools/walking_baseline.py \
        --task jumper.five_foot --checkpoint <model.pt> --out walking.json

`stop` is two phases: 3 s of 0.25 m/s, then 3 s of zero; the residual speed in
the second phase is the stopping result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

#: name -> (vx, vy, wz, step at which to switch to zero or None)
CONDITIONS = {
    "stand": (0.0, 0.0, 0.0, None),
    "fwd0.25": (0.25, 0.0, 0.0, None),
    "back0.25": (-0.25, 0.0, 0.0, None),
    "turn0.5": (0.0, 0.0, 0.5, None),
    "stop": (0.25, 0.0, 0.0, 150),
}
FALL_Z = 0.06
FALL_UP_Z = 0.4


def _up_z(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    del w, z
    return 1.0 - 2.0 * (x * x + y * y)


def _heading(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading_frame(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    cos, sin = math.cos(yaw), math.sin(yaw)
    return cos * dx + sin * dy, -sin * dx + cos * dy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="jumper.five_foot")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=300,
                   help="control steps per episode; 300 is 6 s at 50 Hz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=("warp", "native"), default="warp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    resolution = resolve(backend=args.backend, device=args.device,
                         num_envs=args.episodes)
    use_backend(resolution)
    agent_cfg = tasks.load_agent_cfg(args.task)
    env_cfg = tasks.load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.episodes
    env_cfg.seed = args.seed
    env_cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands["twist"].rel_standing_envs = 0.0
    env_cfg.commands["twist"].rel_heading_envs = 0.0
    pose_cfg = env_cfg.commands.get("body_pose")
    if pose_cfg is not None:
        pose_cfg.resampling_time_range = (1.0e9, 1.0e9)
        pose_cfg.rel_neutral_envs = 1.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (tasks.load_runner_cls(args.task) or MjlabOnPolicyRunner)(
        wrapped, asdict(agent_cfg), device=resolution.device)
    runner.load(str(Path(args.checkpoint)), load_cfg={"actor": True}, strict=True,
                map_location=resolution.device)
    policy = runner.get_inference_policy(device=resolution.device)
    robot = env.scene["robot"]
    twist = env.command_manager.get_term("twist")
    pose = env.command_manager.get_term("body_pose") if pose_cfg is not None else None

    def pin(vx: float, vy: float, wz: float) -> None:
        twist.cfg.ranges.lin_vel_x = (vx, vx)
        twist.cfg.ranges.lin_vel_y = (vy, vy)
        twist.cfg.ranges.ang_vel_z = (wz, wz)
        twist.vel_command_b[:] = torch.tensor([[vx, vy, wz]], device=env.device)
        twist.time_left[:] = 1.0e9
        twist.is_standing_env[:] = False
        twist.is_heading_env[:] = False
        if pose is not None:
            pose.pose_target_b[:] = 0.0
            pose.pose_command_b[:] = 0.0
            pose.time_left[:] = 1.0e9
            pose.is_neutral_env[:] = False

    results: dict[str, dict] = {}
    with torch.inference_mode():
        for name, (vx, vy, wz, switch) in CONDITIONS.items():
            obs, _ = wrapped.reset()
            pin(vx, vy, wz)
            start = robot.data.root_link_pos_w.clone()
            start_yaw = [_heading(robot.data.root_link_quat_w[i])
                         for i in range(args.episodes)]
            alive = [True] * args.episodes
            fell = [False] * args.episodes
            counts = [0] * args.episodes
            lin_err = [0.0] * args.episodes
            yaw_err = [0.0] * args.episodes
            speed = [0.0] * args.episodes
            stop_speed = [0.0] * args.episodes
            stop_count = [0] * args.episodes
            last_pos = robot.data.root_link_pos_w.clone()
            for step in range(args.steps):
                if switch is not None and step == switch:
                    pin(0.0, 0.0, 0.0)
                obs, _, dones, _ = wrapped.step(policy(obs))
                for i in range(args.episodes):
                    if not alive[i]:
                        continue
                    pos = robot.data.root_link_pos_w[i]
                    vel = robot.data.root_link_lin_vel_b[i]
                    ang = robot.data.root_link_ang_vel_b[i]
                    last_pos[i] = pos.clone()
                    if float(pos[2]) < FALL_Z or _up_z(
                            robot.data.root_link_quat_w[i]) < FALL_UP_Z:
                        fell[i], alive[i] = True, False
                        continue
                    counts[i] += 1
                    lin_err[i] += abs(float(vel[0]) - vx) + abs(float(vel[1]) - vy)
                    yaw_err[i] += abs(float(ang[2]) - wz)
                    speed[i] += math.hypot(float(vel[0]), float(vel[1]))
                    if switch is not None and step >= switch:
                        stop_speed[i] += math.hypot(float(vel[0]), float(vel[1]))
                        stop_count[i] += 1
                for i in range(args.episodes):
                    if alive[i] and bool(dones[i]) and step < args.steps - 1:
                        fell[i], alive[i] = True, False
                if not any(alive):
                    break
            detail = []
            for i in range(args.episodes):
                count = max(counts[i], 1)
                end = last_pos[i]
                dx = float(end[0] - start[i][0])
                dy = float(end[1] - start[i][1])
                detail.append({
                    "episode": i,
                    "alive_steps": counts[i],
                    "fell": fell[i],
                    "lin_error": lin_err[i] / count,
                    "yaw_error": yaw_err[i] / count,
                    "speed": speed[i] / count,
                    "net_disp_heading_xy": list(_heading_frame(dx, dy, start_yaw[i])),
                    "residual_speed": (stop_speed[i] / max(stop_count[i], 1)
                                       if switch is not None else None),
                })
            results[name] = {
                "command": {"vx": vx, "vy": vy, "wz": wz},
                "episodes": args.episodes,
                "falls": sum(1 for item in detail if item["fell"]),
                "no_fall_episodes": sum(1 for item in detail if not item["fell"]),
                "mean_lin_error": sum(item["lin_error"] for item in detail) / args.episodes,
                "mean_yaw_error": sum(item["yaw_error"] for item in detail) / args.episodes,
                "mean_speed": sum(item["speed"] for item in detail) / args.episodes,
                "mean_net_disp_heading_xy": [
                    sum(item["net_disp_heading_xy"][0] for item in detail) / args.episodes,
                    sum(item["net_disp_heading_xy"][1] for item in detail) / args.episodes,
                ],
                "mean_residual_speed": (
                    sum(item["residual_speed"] for item in detail) / args.episodes
                    if switch is not None else None),
                "detail": detail,
            }
    env.close()
    return {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "seed": args.seed,
        "steps": args.steps,
        "backend": args.backend,
        "conditions": results,
    }


def main() -> int:
    args = parse_args()
    report = run(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(f"{'condition':<10}{'falls':>8}{'fwd disp':>10}{'vx err':>9}"
          f"{'wz err':>9}{'speed':>8}{'stop speed':>11}")
    for name, row in report["conditions"].items():
        residual = ("" if row["mean_residual_speed"] is None
                    else f"{row['mean_residual_speed']:>11.3f}")
        print(f"{name:<10}{row['falls']:>3}/{row['episodes']:<4}"
              f"{row['mean_net_disp_heading_xy'][0]:>10.3f}"
              f"{row['mean_lin_error']:>9.3f}{row['mean_yaw_error']:>9.3f}"
              f"{row['mean_speed']:>8.3f}{residual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
