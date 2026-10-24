#!/usr/bin/env python3
"""Inspect the physical close/lift phases used by ``--contact-notebook``."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend="native", device="cpu", num_envs=1)
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv

    from tasks.jumper.five_foot.claw import FINGER_JOINT
    from tasks.jumper.five_foot.mdp.grasp import (
        Claw,
        _notebook_floor_clearance,
    )

    cfg = tasks.load_env_cfg(
        "jumper.five_foot",
        task_args={"grasp_ready_hold": True, "contact_notebook": True},
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    robot = env.scene["robot"]
    shoulder = robot.find_joints(["LF_J1_joint"])[0][0]
    finger = robot.find_joints([FINGER_JOINT])[0][0]
    claw = Claw(env)
    action = torch.zeros(1, env.action_manager.total_action_dim, device=res.device)

    def report(step: int) -> None:
        clearance = float(
            _notebook_floor_clearance(env, "notebook", (0.0525, 0.015, 0.0725))[0]
        )
        distance = float(torch.linalg.norm(
            env.scene["notebook"].data.root_link_pos_w[0] - claw.mouth()[0]
        ))
        print(
            f"step={step:3d} shoulder={float(robot.data.joint_pos[0, shoulder]):+.4f} "
            f"finger={float(robot.data.joint_pos[0, finger]):+.4f} "
            f"clearance={clearance * 1000:+.2f}mm mouth_dist={distance * 1000:.2f}mm"
        )

    env.reset()
    report(0)
    with torch.inference_mode():
        for step in range(1, 227):
            env.step(action)
            if step in (1, 25, 50, 99, 100, 125, 150, 175, 199, 200, 224, 225, 226):
                report(step)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
