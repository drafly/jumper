"""jumper.flat -- no gait prior; the one-variable control for tetrapod and ripple.

It adds no gait prior at all, so against `jumper.tetrapod` and `jumper.ripple` it is a
genuine control: those two add exactly one reward term each and change nothing
else. **Not against `jumper.tripod`**, which adds two reward terms (`tripod_gait`
and `stance_load`) and a `gait_phase` observation -- three variables at once, so
an outcome there cannot be attributed to the gait prior among them. See the
measured table in `tasks/jumper/common/ppo.py`.

It already walks -- tracking rates of 94/91/99% (see docs/DESIGN.md). The value of
the gait tasks is in regularising the shape of the gait, not in making walking
work at all.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`) and its hyper-parameters
(`rl_cfg.py`). It adds no gait prior, so it has no `mdp/`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.flat",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-ground velocity tracking; no gait prior (the one-variable control for tetrapod and ripple)",
    tags=("locomotion", "jumper", "flat", "base"),
)


def cli_args(group):
    """This task's own command-line arguments; see `tasks.load_cli_args`.

    A diagnostic knob, not a tuning value: the default is None, which leaves the
    shared skeleton's `rel_forward_envs = 0.0` exactly as the baseline has it.
    `--flat-forward-envs 0.2` is the single-factor change of the controlled
    forward-bias experiment; the reasoning for why this task's ladder makes the
    0.3 m/s upstream floor legitimate is in `env_cfg.py` beside the assignment.

    Declared here rather than in `scripts/_cli.py` because it is this task's
    vocabulary, and `__init__.py` is the one module of a task that is always
    imported; declaring an argument needs `argparse` and nothing else.
    """
    group.add_argument(
        "--flat-forward-envs",
        dest="flat_forward_envs",
        type=float,
        default=None,
        metavar="FRACTION",
        help="fraction of environments given upstream's forward-only command "
        "(0.3 m/s floor); None keeps the baseline 0.0. Diagnostic, for the "
        "controlled forward-command-bias experiment",
    )
    return ("flat_forward_envs",)
