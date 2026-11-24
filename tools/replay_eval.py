#!/usr/bin/env python3
"""Measure a checkpoint's command tracking: standing, straight and turning.

`scripts/play.py` proves a checkpoint loads and runs; it does not say whether
the robot stands, moves in the commanded direction or falls. This tool does, on
one environment, with the command pinned so three scenarios are comparable
across checkpoints:

    python tools/replay_eval.py --task jumper.flat --checkpoint <model_19.pt> \
        --episodes 2 --steps 250 --seed 0 --out eval.json

What is reported per scenario: fall rate (a base height or tilt threshold, or
the environment's own termination), mean survival in steps, mean absolute
linear/yaw tracking error, the achieved speed, and the **net displacement**,
projected onto the heading the episode started with. Displacement is what tells
a policy that tracks a command apart from one that never moves: a small mean
error on a five-second episode is also what a robot standing still in a noisy
body frame produces.

`backward` is part of the vocabulary, not part of the default sweep; the
default stays `stand,straight,turn` so existing command lines keep measuring
what they measured.

The environment is built on the native backend with one environment, so it runs
on a CPU and needs no display. The command is pinned by writing the velocity
command term after every reset and disarming its resampling; without that the
term would overwrite the scenario's command on its own schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

#: Five centimetres below the nominal stand height, and a 0.4 up-axis: a robot
#: lying on its shell fails both, a robot crouched but upright fails neither.
FALL_Z = 0.06
FALL_UP_Z = 0.4

SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "straight": (0.3, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.5),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=250, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument(
        "--scenarios", default="stand,straight,turn",
        help="comma-separated subset of stand,straight,turn",
    )
    return parser.parse_args(argv)


def _up_z(quat) -> float:
    """The world z component of the body's up axis, from a wxyz quaternion."""
    w, x, y, z = (float(value) for value in quat)
    del w, z
    return 1.0 - 2.0 * (x * x + y * y)


def _heading(quat) -> float:
    """The body's yaw in the world, in radians, from a wxyz quaternion."""
    w, x, y, z = (float(value) for value in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _in_heading_frame(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    """Rotate a world displacement into the frame the episode started in.

    Forward is +x of that frame. Without it, a policy that walks in whatever
    direction it reset facing would be reported as not having advanced along
    the commanded axis -- the robot's initial heading is the axis the command
    means.
    """
    cos, sin = math.cos(yaw), math.sin(yaw)
    return cos * dx + sin * dy, -sin * dx + cos * dy


def run(args: argparse.Namespace) -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    from mjrl.replay import skip_rewards

    import tasks

    torch.manual_seed(args.seed)
    spec = tasks.get(args.task)
    asset = spec.resolve_asset(None)
    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)

    env_cfg = tasks.load_env_cfg(args.task, asset=asset, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    skip_rewards(env)
    agent_cfg = tasks.load_agent_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=resolution.device)
    runner.load(
        str(Path(args.checkpoint)), load_cfg={"actor": True}, strict=True,
        map_location=resolution.device,
    )
    policy = runner.get_inference_policy(device=resolution.device)

    commands = env.unwrapped.command_manager
    term = commands.get_term("twist")
    robot = term.robot
    # Pin the command: no resampling, no standing/heading re-rolls, and a
    # time_left that outlives the run.
    term.cfg.resampling_time_range = (1.0e9, 1.0e9)
    term.cfg.rel_standing_envs = 0.0
    term.cfg.rel_heading_envs = 0.0

    def pin(vx: float, vy: float, wz: float) -> torch.Tensor:
        command = torch.tensor([[vx, vy, wz]], device=env.device)
        term.vel_command_b[:] = command
        term.time_left[:] = 1.0e9
        term.is_standing_env[:] = False
        term.is_heading_env[:] = False
        return command

    results = {}
    try:
        for name in args.scenarios.split(","):
            name = name.strip()
            if name not in SCENARIOS:
                continue
            vx, vy, wz = SCENARIOS[name]
            pin(vx, vy, wz)
            obs = wrapped.get_observations()
            episodes = []
            for _ in range(args.episodes):
                obs = wrapped.get_observations()
                pin(vx, vy, wz)
                start_position = robot.data.root_link_pos_w[0].clone()
                start_yaw = _heading(robot.data.root_link_quat_w[0])
                steps = 0
                fell = False
                lin_error = 0.0
                yaw_error = 0.0
                speed_sum = 0.0
                vx_sum = 0.0
                wz_sum = 0.0
                with torch.inference_mode():
                    while steps < args.steps:
                        obs, _, dones, _ = wrapped.step(policy(obs))
                        steps += 1
                        position = robot.data.root_link_pos_w[0]
                        velocity = robot.data.root_link_lin_vel_b[0]
                        angular = robot.data.root_link_ang_vel_b[0]
                        if float(position[2]) < FALL_Z or _up_z(
                            robot.data.root_link_quat_w[0]
                        ) < FALL_UP_Z:
                            fell = True
                        lin_error += abs(float(velocity[0]) - vx) + abs(float(velocity[1]) - vy)
                        yaw_error += abs(float(angular[2]) - wz)
                        speed_sum += (float(velocity[0]) ** 2 + float(velocity[1]) ** 2) ** 0.5
                        vx_sum += float(velocity[0])
                        wz_sum += float(angular[2])
                        if bool(dones[0]):
                            break
                        pin(vx, vy, wz)
                moving = max(steps, 1)
                end = robot.data.root_link_pos_w[0]
                dx = float(end[0] - start_position[0])
                dy = float(end[1] - start_position[1])
                disp_x, disp_y = _in_heading_frame(dx, dy, start_yaw)
                episodes.append({
                    "steps": steps,
                    "fell": fell,
                    "lin_error": lin_error / moving,
                    "yaw_error": yaw_error / moving,
                    "speed": speed_sum / moving,
                    "mean_vx": vx_sum / moving,
                    "mean_wz": wz_sum / moving,
                    "net_disp_world_xy": [dx, dy],
                    "net_disp_heading_xy": [disp_x, disp_y],
                })
            fell_count = sum(1 for episode in episodes if episode["fell"])
            count = max(len(episodes), 1)
            results[name] = {
                "command": {"vx": vx, "vy": vy, "wz": wz},
                "episodes": len(episodes),
                "falls": fell_count,
                "fall_rate": fell_count / count,
                "mean_survival_steps": sum(e["steps"] for e in episodes) / count,
                "mean_lin_error": sum(e["lin_error"] for e in episodes) / count,
                "mean_yaw_error": sum(e["yaw_error"] for e in episodes) / count,
                "mean_speed": sum(e["speed"] for e in episodes) / count,
                "mean_vx": sum(e["mean_vx"] for e in episodes) / count,
                "mean_wz": sum(e["mean_wz"] for e in episodes) / count,
                "mean_net_disp_heading_xy": [
                    sum(e["net_disp_heading_xy"][0] for e in episodes) / count,
                    sum(e["net_disp_heading_xy"][1] for e in episodes) / count,
                ],
                "mean_net_disp_world_xy": [
                    sum(e["net_disp_world_xy"][0] for e in episodes) / count,
                    sum(e["net_disp_world_xy"][1] for e in episodes) / count,
                ],
                "detail": episodes,
            }
    finally:
        env.close()
    return {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seed": args.seed,
        "max_steps": args.steps,
        "scenarios": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, "utf-8")
    else:
        sys.stdout.write(text)
    print(f"{'scenario':<10}{'falls':>7}{'survival':>10}{'lin err':>9}{'yaw err':>9}{'speed':>8}")
    for name, row in report["scenarios"].items():
        print(
            f"{name:<10}{row['falls']}/{row['episodes']:<5}"
            f"{row['mean_survival_steps']:>10.0f}{row['mean_lin_error']:>9.3f}"
            f"{row['mean_yaw_error']:>9.3f}{row['mean_speed']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
