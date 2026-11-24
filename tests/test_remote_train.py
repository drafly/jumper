"""The remote-training driver's own contracts.

`tools/remote_train.py` decides from **text on the other side of an ssh
connection** whether a job succeeded. Every failure here is silent: a job whose
tmux session died looks finished, a copy that dropped a file exits 0, a status
file that was never written looks queued. These tests pin the parsing and the
refusals, not the network.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "remote_train.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("remote_train_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while decorating.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rt = _load()


# --------------------------------------------------------------------------
# status classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("events", "alive", "checkpoints", "expected"),
    [
        ([], False, 0, "queued"),
        ([{"event": "start"}], True, 0, "running"),
        # tmux gone, no end record: a reboot or a crash, not a success.
        ([{"event": "start"}], False, 0, "unknown"),
        (
            [{"event": "start"}, {"event": "end", "rc": "0", "reason": "completed"}],
            False, 1, "succeeded",
        ),
        # exit 0 but nothing on disk: the trainer said it finished and did not.
        (
            [{"event": "start"}, {"event": "end", "rc": "0", "reason": "completed"}],
            False, 0, "failed",
        ),
        (
            [{"event": "start"}, {"event": "end", "rc": "1", "reason": "exit"}],
            False, 1, "failed",
        ),
        (
            [{"event": "start"}, {"event": "end", "rc": "130", "reason": "stopped"}],
            False, 1, "stopped",
        ),
        (
            [{"event": "start"}, {"event": "end", "rc": "124", "reason": "timeout"}],
            False, 0, "failed",
        ),
    ],
)
def test_status_needs_both_the_exit_code_and_a_checkpoint(
    events: list[dict], alive: bool, checkpoints: int, expected: str
) -> None:
    """A run is `succeeded` only if rc==0 *and* a checkpoint exists.

    The tempting implementation -- tmux gone means done -- is the one this
    guards: it reports a rebooted machine as a successful training run.
    """
    assert rt.classify(events, alive, checkpoints) == expected


def test_status_tsv_keeps_the_last_end_record() -> None:
    text = (
        "start\tts=2026-11-22T23:00:00+08:00\tpid=10\n"
        "end\tts=2026-11-22T23:01:00+08:00\trc=1\treason=exit\n"
        "end\tts=2026-11-22T23:02:00+08:00\trc=0\treason=completed\n"
    )
    events = rt.parse_status_tsv(text)
    assert [event["event"] for event in events] == ["start", "end", "end"]
    assert rt.classify(events, alive=False, checkpoints=2) == "succeeded"


def test_progress_reads_the_rsl_rl_line_and_the_resume_target() -> None:
    """The iteration number is how a resumed run proves it continued."""
    log = (
        "[mjrl] resuming from /root/autodl-tmp/jumper/runs/a/tree/x/model_19.pt\n"
        "Learning iteration 19/22\n"
        "Learning iteration 21/22\n"
    )
    progress = rt.progress_from_log(log)
    assert progress["iteration"] == 21
    assert progress["total"] == 22
    assert progress["resuming_from"].endswith("model_19.pt")


# --------------------------------------------------------------------------
# command construction
# --------------------------------------------------------------------------


def test_job_script_quotes_arguments_instead_of_splicing_them() -> None:
    """An argument is data, not shell syntax.

    `--extra` is user input; a `;` in it must reach train.py, not the shell.
    """
    cfg = rt.Config(hostname="example", workdir="/root/w")
    argv = ["scripts/train.py", "--task", "jumper.flat", "--note", "; rm -rf /"]
    script = rt.build_job_sh(cfg, "20260101-000000-abcdef", "r1", argv, 3600)
    assert "' ; rm -rf /'" not in script
    assert "'; rm -rf /'" in script
    assert "timeout --foreground --signal=INT --kill-after=60 3600" in script
    assert 'status end "ts=$(date -Is)" "rc=$rc" "reason=$reason"' in script


def test_a_zero_limit_removes_the_timeout_wrapper() -> None:
    cfg = rt.Config(hostname="example", workdir="/root/w")
    script = rt.build_job_sh(cfg, "job", "r1", ["scripts/train.py"], 0)
    assert "timeout --signal" not in script


def test_ssh_command_pins_the_same_host_key_rsync_must_use() -> None:
    """rsync spawns its own ssh.

    Without `-e`, rsync's ssh reads the user's default known_hosts, and a first
    connection dies with `Host key verification failed` before transferring a
    byte -- which is how the first sync of this tool failed.
    """
    cfg = rt.Config(hostname="example", port=24227, workdir="/root/w",
                    known_hosts=str(REPO / ".remote" / "known_hosts"))
    remote = rt.Remote(cfg)
    command = remote.ssh_command()
    assert "-p 24227" in command
    assert "UserKnownHostsFile=" in command
    assert "BatchMode=yes" in command
    assert "example" not in command.split()[-1]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_config_rejects_both_host_forms(tmp_path: Path) -> None:
    path = tmp_path / "remote.toml"
    path.write_text('host = "a"\nhostname = "b"\nworkdir = "/root/w"\n', "utf-8")
    with pytest.raises(rt.RemoteTrainError, match="exactly one"):
        rt.load_config(str(path))


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "remote.toml"
    path.write_text('hostname = "h"\nworkdir = "/root/w"\nreslts = "oops"\n', "utf-8")
    with pytest.raises(rt.RemoteTrainError, match="unknown keys"):
        rt.load_config(str(path))


def test_missing_config_points_at_the_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no .remote.toml in the working directory
    with pytest.raises(rt.RemoteTrainError, match="remote_train.example.toml"):
        rt.load_config(str(tmp_path / "absent.toml"))


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


class FakeRemote:
    """Enough of Remote to reach the refusal, and no further."""

    target = "fake"

    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout
        self.started = False

    def run(self, script: str, *, timeout: int = 120, check: bool = True):
        return subprocess.CompletedProcess(["ssh"], 0, self.stdout, "")

    def push_tree(self, source: Path, dest: str) -> None:
        self.started = True


def test_submit_refuses_while_another_job_is_alive(monkeypatch, tmp_path: Path) -> None:
    """Two jobs on one GPU is the default mistake; it must not be silent."""
    cfg = rt.Config(hostname="example", workdir="/root/w", results=str(tmp_path))
    monkeypatch.setattr(rt, "active_jobs", lambda remote, cfg: [{"job_id": "old", "pid": 42}])
    with pytest.raises(rt.RemoteTrainError, match="already running"):
        rt.submit_job(
            cfg, remote=FakeRemote(), task="jumper.flat", release="r1",
            argv=["scripts/train.py"], max_hours=1, dry_run=False,
            allow_busy=False, force=False,
        )


def test_submit_refuses_a_busy_gpu_without_killing_anything(monkeypatch, tmp_path: Path) -> None:
    cfg = rt.Config(hostname="example", workdir="/root/w", results=str(tmp_path))
    monkeypatch.setattr(rt, "active_jobs", lambda remote, cfg: [])
    remote = FakeRemote(stdout="1234, 9000\n")
    with pytest.raises(rt.RemoteTrainError, match="does not stop unknown processes"):
        rt.submit_job(
            cfg, remote=remote, task="jumper.flat", release="r1",
            argv=["scripts/train.py"], max_hours=1, dry_run=False,
            allow_busy=False, force=False,
        )
    assert remote.started is False


# --------------------------------------------------------------------------
# download verification
# --------------------------------------------------------------------------


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_verification_catches_a_dropped_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", "utf-8")
    manifest = f"{_sha(tmp_path / 'a.txt')}  ./a.txt\n{_sha(tmp_path / 'a.txt')}  ./b.txt\n"
    with pytest.raises(rt.RemoteTrainError, match="missing"):
        rt.verify_manifest(tmp_path, manifest)


def test_manifest_verification_ignores_macos_apple_double_files(tmp_path: Path) -> None:
    """bsdtar on macOS adds `._name` siblings; they are not part of the release."""
    (tmp_path / "a.txt").write_text("a", "utf-8")
    manifest = f"{_sha(tmp_path / 'a.txt')}  ./a.txt\n" + "0" * 64 + "  ./._a.txt\n"
    rt.verify_manifest(tmp_path, manifest)


def test_manifest_verification_checks_provenance_by_hash(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", "utf-8")
    manifest = f"{_sha(tmp_path / 'a.txt')}  ./a.txt\n" + "0" * 64 + "  ./CODE.patch\n"
    with pytest.raises(rt.RemoteTrainError, match="provenance"):
        rt.verify_manifest(tmp_path, manifest, extra_hashes={"CODE.patch": "1" * 64})


def test_download_verification_refuses_a_truncated_checkpoint(tmp_path: Path) -> None:
    """A half-written `.pt` loads far enough to look like a policy bug."""
    checkpoint = tmp_path / "tree" / "model_19.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"truncated")
    manifest = f"{'0' * 64}  tree/model_19.pt\n"
    with pytest.raises(rt.RemoteTrainError, match="SHA-256 mismatch"):
        rt.verify_download(tmp_path, manifest, skip=set())


# --------------------------------------------------------------------------
# connection failures are not job failures
# --------------------------------------------------------------------------


def test_unreachable_ssh_is_reported_as_a_connection_error(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config = tmp_path / "remote.toml"
    config.write_text('hostname = "example"\nworkdir = "/root/w"\n', "utf-8")

    def fail(*args, **kwargs):
        raise rt.Unreachable("cannot reach example")

    monkeypatch.setattr(rt.Remote, "run", fail)
    assert rt.main(["-c", str(config), "check"]) == 2
    assert "connection error" in capsys.readouterr().err


# --------------------------------------------------------------------------
# what travels
# --------------------------------------------------------------------------


def test_the_release_excludes_machine_state_and_large_outputs() -> None:
    for pattern in (".venv", ".git", "logs", "out", "deploy/fsm/target"):
        assert pattern in rt.EXCLUDES
    for pattern in ("assets", "scenes", "tasks"):
        assert pattern not in rt.EXCLUDES


def test_release_ids_record_the_commit_and_dirty_state() -> None:
    clean = {"head": "a" * 40, "dirty": False, "patch": ""}
    dirty = {"head": "a" * 40, "dirty": True, "patch": "diff --git a b\n"}
    assert rt.release_id(clean, None).endswith("aaaaaaa")
    assert "-dirty-" in rt.release_id(dirty, None)
    assert rt.release_id(dirty, "fixed") == "fixed"


def test_resume_drops_a_replaced_flag_with_its_value() -> None:
    """`--max-iterations 20` must lose both tokens, not just the flag.

    Removing only the flag left `--num_envs 64 20`, so the resumed job died on
    argparse and the only evidence was a failed run in `runs/`.
    """
    original = [
        "scripts/train.py", "--task", "jumper.flat", "--num_envs", "64",
        "--max-iterations", "20", "--headless",
    ]
    out = rt.resume_argv(original, "/runs/a/model_19.pt", 5)
    assert out[out.index("--num_envs") + 1] == "64"
    assert out[out.index("--max-iterations") + 1] == "5"
    assert "20" not in out
    assert out.count("--resume") == 1
    assert out[out.index("--checkpoint") + 1] == "/runs/a/model_19.pt"


def test_remote_spec_carries_host_but_not_the_ssh_port() -> None:
    """rsync needs `host:path`; the port belongs in `-e ssh -p`.

    `root@host:24227:/path` is not a path, and the download failed with
    `change_dir ... No such file or directory`.
    """
    cfg = rt.Config(hostname="example", port=24227, user="root", workdir="/w")
    remote = rt.Remote(cfg)
    assert remote.remote_spec("/runs/a") == "root@example:/runs/a/"
    assert remote.label == "root@example:24227"


def test_a_stop_that_killed_the_group_is_stopped_not_unknown() -> None:
    """The wrapper is killed before it can write an `end`, and that is a stop.

    Without this the job looked like `unknown` after `stop` -- indistinguishable
    from a reboot, which is exactly what the status table exists to separate.
    """
    events = [{"event": "start"}, {"event": "stop"}]
    assert rt.classify(events, alive=False, checkpoints=1, stop_requested=True) == "stopped"
    assert rt.classify(events, alive=True, checkpoints=1, stop_requested=True) == "running"
    assert rt.classify(events, alive=False, checkpoints=1) == "unknown"


def test_admission_is_exclusive_by_default_and_capped_at_two() -> None:
    """Exclusive unless asked, and never more than two managed jobs.

    An unbounded `--concurrent` would turn one typo into a swap storm on a card
    whose headroom was measured with two jobs at most.
    """
    one = [{"job_id": "a", "pid": 1}]
    two = one + [{"job_id": "b", "pid": 2}]
    rt.admit_job([], concurrent=False, force=False)
    with pytest.raises(rt.RemoteTrainError, match="already running"):
        rt.admit_job(one, concurrent=False, force=False)
    rt.admit_job(one, concurrent=True, force=False)
    with pytest.raises(rt.RemoteTrainError, match="at most two"):
        rt.admit_job(two, concurrent=True, force=False)
    rt.admit_job(two, concurrent=False, force=True)


def test_a_job_runs_in_its_own_root_not_the_shared_release() -> None:
    """Two jobs must not share one `logs` symlink.

    Re-pointing `release/logs` at a second job's tree would move the first job's
    checkpoints mid-run; the per-job root keeps each `logs` its own.
    """
    cfg = rt.Config(hostname="example", workdir="/root/w")
    script = rt.build_job_sh(cfg, "job", "r1", ["scripts/train.py"], 60)
    assert "cd /root/w/runs/job/root" in script
    assert rt.job_root(cfg, "job") == "/root/w/runs/job/root"


def test_fetch_defaults_to_the_final_checkpoint_and_keeps_all_explicit() -> None:
    """The default download is the evidence, not 33 checkpoints.

    A timed 330 MB download at a throttled link is the failure this pins; the
    full trajectory stays available through `--all`.
    """
    job = {
        "job_id": "j", "dir": "/w/runs/j", "status": "succeeded",
        "meta": {"release": "r3"},
        "checkpoints": [
            {"path": "/w/runs/j/tree/m/t/model_0.pt"},
            {"path": "/w/runs/j/tree/m/t/model_1599.pt"},
        ],
    }
    plan = rt.fetch_plan(job, full=False)
    assert plan["mode"] == "selective"
    assert plan["tree"].endswith("tree/m/t")
    assert plan["tree_include"] == "model_1599.pt"
    assert plan["tree_exclude"] == "model_*.pt"
    full = rt.fetch_plan(job, full=True)
    assert full["mode"] == "all" and full["root"] == "/w/runs/j"


def test_remote_spec_trails_a_slash_only_for_directories() -> None:
    cfg = rt.Config(hostname="example", workdir="/w")
    remote = rt.Remote(cfg)
    assert remote.remote_spec("/w/runs/j") == "root@example:/w/runs/j/"
    assert remote.remote_spec("/w/runs/j/meta.json", directory=False) == \
        "root@example:/w/runs/j/meta.json"
