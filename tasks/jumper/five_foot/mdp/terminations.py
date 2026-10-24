"""Termination gates specific to the five-foot contact-carry fine-tune."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.terminations import illegal_contact

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def illegal_contact_after_grace(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
    grace_steps: int,
) -> torch.Tensor:
    """End on non-foot ground contact after the reset/grip transient.

    The robot is dropped onto its stance at reset while the finger closes around
    the free notebook. Contact during that short physical settling interval is
    not the CARRY failure this term represents. Once ``grace_steps`` have elapsed,
    it is the same force gate as the task's graded ground-contact reward.
    """
    live = env.episode_length_buf >= grace_steps
    return live & illegal_contact(env, sensor_name, force_threshold)
