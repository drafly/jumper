"""The post-fetch pipeline's silent-failure contracts.

The failures this pins all look like success when they happen: a rerun that
sends the same message twice, a retry that re-runs the simulation, a task with
no profile that gets measured anyway, an evaluation that reports "failed" but
means "did not perform". None of these are visible in a passing run, so each is
written down as a test.

The package is loaded the way the other tool tests load their module -- by path,
under its own name -- because this repository has no `sys.path` set-up and
`tests/test_layout.py` keeps it that way.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "tools" / "postfetch"


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "postfetch_under_test",
        PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_package()
config = importlib.import_module("postfetch_under_test.config")
state = importlib.import_module("postfetch_under_test.state")
artifacts = importlib.import_module("postfetch_under_test.artifacts")
evaluate = importlib.import_module("postfetch_under_test.evaluate")
pipeline = importlib.import_module("postfetch_under_test.pipeline")
video = importlib.import_module("postfetch_under_test.video")


def _toml(tmp_path: Path, *, thresholds: str = "", video_backend: str = "local",
          video_enabled: bool = True) -> Path:
    path = tmp_path / "postfetch.toml"
    path.write_text(
        f"""
[paths]
results = "{tmp_path / 'results'}"
out = "{tmp_path / 'out'}"

[eval]
steps = 250
episodes = 2
seeds = [0, 1]

[eval.thresholds]
{thresholds}

[profiles]
"jumper.flat" = "walking"

[video]
enabled = {"true" if video_enabled else "false"}
backend = "{video_backend}"
""",
        "utf-8",
    )
    return path


def _job(tmp_path: Path, *, task: str = "jumper.flat", supported: bool = True) -> object:
    checkpoint = tmp_path / "tree" / "model_299.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"not a real checkpoint, only a path")
    return artifacts.FetchedJob(
        source_dir=tmp_path,
        job_id="20260101-000000-abcd12",
        task=task,
        checkpoint=checkpoint,
        checkpoint_sha256="a" * 64,
        git_head="deadbeef" * 5,
        code={},
        meta={"task": task},
        fetch={"verified": 8, "missing": []},
        supported=supported,
        reason="" if supported else f"task {task!r} has no evaluation profile",
        compatible=True,
        compatibility_reason="",
        warnings=[],
    )


# --------------------------------------------------------------------------
# idempotency key and evaluation version
# --------------------------------------------------------------------------


def test_unit_key_is_the_job_the_bytes_and_the_measurement() -> None:
    base = state.unit_key("job-a", "sha-a", "v1")
    assert base == state.unit_key("job-a", "sha-a", "v1")
    assert base != state.unit_key("job-b", "sha-a", "v1")
    assert base != state.unit_key("job-a", "sha-b", "v1")
    assert base != state.unit_key("job-a", "sha-a", "v2")


def test_eval_version_ignores_delivery_settings_and_tracks_thresholds(tmp_path: Path) -> None:
    """A new chat is not a new measurement; a new threshold is."""
    strict = config.load(_toml(tmp_path, thresholds="straight_disp_min = 0.5"))
    loose = config.load(_toml(tmp_path, thresholds="straight_disp_min = 0.1"))
    assert strict.eval_version("jumper.flat") != loose.eval_version("jumper.flat")
    again = config.load(_toml(tmp_path, thresholds="straight_disp_min = 0.5",
                              video_backend="server", video_enabled=False))
    assert again.eval_version("jumper.flat") == strict.eval_version("jumper.flat")


# --------------------------------------------------------------------------
# state: a restart must resume, not restart
# --------------------------------------------------------------------------


def test_state_survives_a_restart_and_a_failed_stage_is_retryable(tmp_path: Path) -> None:
    store = state.Store(tmp_path / "state.json")
    entry = state.new_entry(
        key="k1", job_id="j1", task="jumper.flat",
        checkpoint=tmp_path / "model_299.pt", checkpoint_sha256="a" * 64,
        eval_version="v1", source_dir=tmp_path,
    )
    store.put(entry)
    store.set_stage("k1", "download", "ok")
    store.set_stage("k1", "eval", "failed", "boom: shape")
    reopened = state.Store(tmp_path / "state.json")
    persisted = reopened.get("k1")
    assert persisted is not None
    assert persisted["stages"]["eval"]["status"] == "failed"
    assert persisted["stages"]["eval"]["reason"].startswith("boom")
    assert state.needs_work(persisted, max_attempts=3)
    reopened.set_stage("k1", "eval", "ok")
    reopened.set_stage("k1", "video", "skipped", "no evaluation")
    reopened.set_stage("k1", "notify", "skipped", "disabled")
    assert not state.needs_work(state.Store(tmp_path / "state.json").get("k1"), max_attempts=3)


# --------------------------------------------------------------------------
# unsupported tasks are not measured with the walking profile
# --------------------------------------------------------------------------


def test_a_task_without_a_profile_is_unsupported_not_measured(tmp_path: Path) -> None:
    job_dir = tmp_path / "remote" / "20260101-000000-abcd12"
    (job_dir / "tree").mkdir(parents=True)
    (job_dir / "tree" / "model_0.pt").write_bytes(b"x")
    (job_dir / "FETCH.json").write_text(json.dumps({"verified": 8, "missing": [],
                                                    "job_id": "20260101-000000-abcd12",
                                                    "mode": "selective"}))
    (job_dir / "meta.json").write_text(json.dumps({"task": "jumper.unknown"}))
    cfg = config.load(_toml(tmp_path))
    job = artifacts.read_job(job_dir, cfg)
    assert not job.supported
    assert "no evaluation profile" in job.reason
    assert job.compatible  # nothing tried to load it


# --------------------------------------------------------------------------
# judge: program success and performance are different answers
# --------------------------------------------------------------------------


def _row(**overrides) -> dict:
    row = {
        "command": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
        "episodes": 4,
        "falls": 0,
        "fall_rate": 0.0,
        "mean_survival_steps": 250.0,
        "mean_lin_error": 0.01,
        "mean_yaw_error": 0.01,
        "mean_speed": 0.0,
        "mean_vx": 0.0,
        "mean_wz": 0.0,
        "mean_net_disp_heading_xy": [0.0, 0.0],
        "min_survival_steps": 250.0,
    }
    row.update(overrides)
    return row


def test_judge_separates_a_missed_threshold_from_a_failed_program(tmp_path: Path) -> None:
    cfg = config.load(_toml(tmp_path, thresholds="straight_disp_min = 0.5"))
    rows = {
        "stand": _row(),
        "straight": _row(mean_lin_error=0.30, mean_net_disp_heading_xy=[0.01, 0.0]),
    }
    judged = evaluate.judge(rows, cfg)
    assert judged["stand"]["pass"] is True
    assert judged["straight"]["pass"] is False
    failed = [check["name"] for check in judged["straight"]["checks"] if not check["passed"]]
    assert set(failed) == {"lin_error", "net_disp_heading_x"}


# --------------------------------------------------------------------------
# pipeline: run once, retry only what failed
# --------------------------------------------------------------------------


def _fake_eval(store_counter: dict):
    def fake(job, cfg, entry_dir: Path) -> dict:
        store_counter["eval"] = store_counter.get("eval", 0) + 1
        entry_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "task": job.task,
            "job_id": job.job_id,
            "checkpoint_name": job.checkpoint.name,
            "checkpoint_sha256": job.checkpoint_sha256,
            "performance_pass": False,
            "program_ok": True,
            "eval_version": "v1",
            "config": {"scenarios": ["stand"]},
            "scenarios": {},
            "judgement": {},
        }
        (entry_dir / "eval.json").write_text(json.dumps(result), "utf-8")
        return result

    return fake


def test_a_second_run_does_not_evaluate_or_notify_again(tmp_path: Path, monkeypatch) -> None:
    cfg = config.load(_toml(tmp_path))
    job = _job(tmp_path)
    monkeypatch.setattr(pipeline.artifacts, "read_job", lambda *args, **kwargs: job)
    counter: dict = {}
    monkeypatch.setattr(pipeline.evaluate, "evaluate", _fake_eval(counter))
    monkeypatch.setattr(pipeline.video, "render", lambda *args, **kwargs: {"status": "skipped",
                                                                           "reason": ""})
    notified: dict = {}

    def fake_notify(*args, **kwargs) -> dict:
        notified["count"] = notified.get("count", 0) + 1
        return {"status": "ok", "reason": "", "detail": {}}

    monkeypatch.setattr(pipeline, "_notify", fake_notify)
    store = state.Store(cfg.state_path)
    pipeline.process_job(job.source_dir, cfg, store)
    pipeline.process_job(job.source_dir, cfg, store)
    assert counter["eval"] == 1
    assert notified["count"] == 1


def test_a_failed_notification_is_retried_without_rerunning_the_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = config.load(_toml(tmp_path))
    job = _job(tmp_path)
    monkeypatch.setattr(pipeline.artifacts, "read_job", lambda *args, **kwargs: job)
    counter: dict = {}
    monkeypatch.setattr(pipeline.evaluate, "evaluate", _fake_eval(counter))
    monkeypatch.setattr(pipeline.video, "render", lambda *args, **kwargs: {"status": "skipped",
                                                                           "reason": ""})
    outcomes = [{"status": "failed", "reason": "network down"},
                {"status": "ok", "reason": "", "detail": {}}]

    def fake_notify(*args, **kwargs) -> dict:
        return outcomes.pop(0)

    monkeypatch.setattr(pipeline, "_notify", fake_notify)
    store = state.Store(cfg.state_path)
    entry = pipeline.process_job(job.source_dir, cfg, store)
    assert entry["stages"]["notify"]["status"] == "failed"
    assert state.needs_work(entry, cfg.max_attempts)
    store.reset_attempts(entry["key"], "notify")
    entry = pipeline.process_job(job.source_dir, cfg, store)
    assert entry["stages"]["notify"]["status"] == "ok"
    assert counter["eval"] == 1  # the simulation was not re-run for a notification


def test_a_video_failure_does_not_block_the_notification(tmp_path: Path, monkeypatch) -> None:
    cfg = config.load(_toml(tmp_path))
    job = _job(tmp_path)
    monkeypatch.setattr(pipeline.artifacts, "read_job", lambda *args, **kwargs: job)
    counter: dict = {}
    monkeypatch.setattr(pipeline.evaluate, "evaluate", _fake_eval(counter))
    monkeypatch.setattr(
        pipeline.video, "render",
        lambda *args, **kwargs: {"status": "failed", "reason": "cgl unavailable"},
    )
    sent: dict = {}

    def fake_notify(*args, **kwargs) -> dict:
        sent["yes"] = True
        return {"status": "ok", "reason": "", "detail": {}}

    monkeypatch.setattr(pipeline, "_notify", fake_notify)
    store = state.Store(cfg.state_path)
    entry = pipeline.process_job(job.source_dir, cfg, store)
    assert entry["stages"]["video"]["status"] == "failed"
    assert "cgl unavailable" in entry["stages"]["video"]["reason"]
    assert entry["stages"]["notify"]["status"] == "ok"
    assert sent.get("yes")


def test_message_says_unsupported_instead_of_passing(tmp_path: Path) -> None:
    job = _job(tmp_path, task="jumper.unknown", supported=False)
    text = pipeline.message_text(job, None, None)
    assert "不支持" in text
    assert "未套用行走指标" in text


# --------------------------------------------------------------------------
# the two entry points: fetch --post and python -m tools.postfetch
# --------------------------------------------------------------------------


def test_fetch_post_passes_the_verified_directory_to_the_pipeline(tmp_path: Path,
                                                                  monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "remote_train_under_test", REPO / "tools" / "remote_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))

        class Done:
            returncode = 0

        return Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._run_post(tmp_path) == 0
    assert calls and calls[0][1:3] == ["-m", "tools.postfetch"]
    assert "--job-dir" in calls[0]
    assert str(tmp_path) in calls[0]
    # The driver is stdlib-only and may be started by the system python3; the
    # pipeline needs the venv (tomllib, torch), so the interpreter is chosen
    # explicitly rather than inherited.
    venv = REPO / ".venv" / "bin" / "python"
    if venv.is_file():
        assert calls[0][0] == str(venv)


def test_config_before_the_subcommand_is_not_discarded() -> None:
    """argparse applies a subparser's defaults over the parent namespace.

    With the shared options duplicated verbatim, `postfetch --config X run ...`
    was parsed as the default config and the file was silently ignored -- the
    kind of failure that looks like a wrong threshold. SUPPRESS on the shared
    options keeps whichever position supplied the value.
    """
    cli = importlib.import_module("postfetch_under_test.cli")
    before = cli._parser().parse_args(["--config", "a.toml", "run", "--job", "j"])
    after = cli._parser().parse_args(["run", "--job", "j", "--config", "b.toml"])
    assert before.config == "a.toml"
    assert after.config == "b.toml"


def test_global_options_are_accepted_after_the_subcommand() -> None:
    """`run --webhook ...` is the order a person types; a parent parser makes it work."""
    cli = importlib.import_module("postfetch_under_test.cli")
    args = cli._parser().parse_args(
        ["run", "--job", "j1", "--webhook", "http://127.0.0.1:1/hook", "--notify", "off"]
    )
    assert args.command == "run"
    assert args.webhook == "http://127.0.0.1:1/hook"
    assert args.notify == "off"


def test_the_module_entry_point_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tools.postfetch", "--help"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "postfetch" in proc.stdout


def test_video_server_backend_is_declared_not_attempted(tmp_path: Path) -> None:
    cfg = config.load(_toml(tmp_path, video_backend="server"))
    result = video.render(None, cfg, tmp_path)
    assert result["status"] == "failed"
    assert "libEGL" in result["reason"]
    assert not any(tmp_path.rglob("*.mp4"))
