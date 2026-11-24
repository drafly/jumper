"""The replay metric's two geometry decisions, pinned.

The tool exists because "the checkpoint loaded" is not "the robot walks". Its
fall test is the part that can pass vacuously -- a threshold that never fires
reports zero falls for a robot on its back -- so the quaternion helper is
checked against an upright and a ninety-degree tilt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "replay_eval_under_test", REPO / "tools" / "replay_eval.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay_eval = _load()


def test_up_axis_is_one_upright_and_minus_one_on_its_back() -> None:
    assert replay_eval._up_z((1.0, 0.0, 0.0, 0.0)) == pytest.approx(1.0)
    assert replay_eval._up_z((0.0, 1.0, 0.0, 0.0)) == pytest.approx(-1.0)
    # Ninety degrees about x: half the cosine terms each, up axis on its side.
    half = 2 ** -0.5
    assert replay_eval._up_z((half, half, 0.0, 0.0)) == pytest.approx(0.0)


def test_the_scenarios_are_distinct_commands() -> None:
    assert replay_eval.SCENARIOS["stand"] == (0.0, 0.0, 0.0)
    assert replay_eval.SCENARIOS["straight"][0] > 0
    assert replay_eval.SCENARIOS["backward"][0] < 0
    assert replay_eval.SCENARIOS["turn"][2] > 0
    assert len(set(replay_eval.SCENARIOS.values())) == 4


def test_displacement_is_rotated_into_the_episode_heading() -> None:
    """A robot facing +y that moves +y has moved forward, not sideways."""
    import math

    forward, left = replay_eval._in_heading_frame(0.0, 1.0, math.pi / 2)
    assert forward == pytest.approx(1.0)
    assert left == pytest.approx(0.0)
