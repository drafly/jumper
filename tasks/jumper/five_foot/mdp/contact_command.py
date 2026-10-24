"""Phase-matched backward command for physical contact fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from .grasp import CONTACT_RESTART_STEP


class ContactCarryVelocityCommand(UniformVelocityCommand):
    """Hold still through grasp/lift, then restart backward under load.

    The normal sampler is deliberately bypassed for this opt-in fine-tune.  A
    random command from reset made the policy walk during finger closure and arm
    lift, while validation first stabilizes and only commands locomotion after
    the object is clear.  This term presents that same state transition without
    moving the robot or prop directly: the policy remains the only source of
    base motion.
    """

    cfg: ContactCarryVelocityCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_w[env_ids] = 0.0
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        age = self._env.episode_length_buf[ids] - self.cfg.restart_step
        moving = age >= 0
        pulse = moving & (age < self.cfg.start_pulse_steps)
        self.vel_command_b[ids] = 0.0
        self.vel_command_b[ids[moving], 0] = (
            self.cfg.direction_sign * self.cfg.carry_speed
        )
        self.vel_command_b[ids[pulse], 0] = (
            self.cfg.direction_sign * self.cfg.start_pulse_speed
        )
        self.vel_command_w[ids] = self.vel_command_b[ids]


@dataclass(kw_only=True)
class ContactCarryVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the phase-matched physical-load restart."""

    restart_step: int = CONTACT_RESTART_STEP
    carry_speed: float = 0.25
    # Deterministic replay needs a short command pulse to leave the loaded
    # static equilibrium across startup-DR seeds.  The validation sweep found
    # 0.30 m/s stalled while 0.35 m/s walked 100 mm; exposing that transition
    # here lets the ground-contact penalty teach a safe response to it.
    start_pulse_speed: float = 0.35
    start_pulse_steps: int = 20
    direction_sign: float = -1.0

    def build(self, env) -> ContactCarryVelocityCommand:
        return ContactCarryVelocityCommand(self, env)


class ContactPushVelocityCommand(ContactCarryVelocityCommand):
    """Forward specialization used by the rigid-body push fixture."""


@dataclass(kw_only=True)
class ContactPushVelocityCommandCfg(ContactCarryVelocityCommandCfg):
    """Stand briefly, then move forward into and through a free push prop."""

    restart_step: int = 50
    direction_sign: float = 1.0

    def build(self, env) -> ContactPushVelocityCommand:
        return ContactPushVelocityCommand(self, env)
