#!/usr/bin/env python3
"""Train Jumper on a remote GPU box over ssh and rsync.

This is the **driver**, not another training entry point. It never imports a
task, a simulator or torch: it copies the repository to a remote machine, runs
`scripts/train.py` there, and brings the result back. The training command stays
the one the repository already documents, so nothing here can drift from it.

    python tools/remote_train.py check
    python tools/remote_train.py sync
    python tools/remote_train.py submit --task jumper.flat --release <id> --num-envs 64
    python tools/remote_train.py status
    python tools/remote_train.py logs <job_id> -f
    python tools/remote_train.py fetch <job_id>
    python tools/remote_train.py eval <job_id> --steps 200
    python tools/remote_train.py resume <job_id> --max-iterations 5
    python tools/remote_train.py stop <job_id>

Why it is shaped this way
-------------------------

**A release is immutable, a job is a directory.** `sync` writes a code snapshot to
`<workdir>/releases/<release_id>/` and never writes there again; `submit` runs that
snapshot with its `logs/` symlinked to `<workdir>/runs/<job_id>/tree/`. A later
sync therefore cannot change the code a running job imported -- which matters
because `pip install -e` is the only way imports resolve in this repository, and a
shared checkout would silently hand the next import to whatever was synced last.

**The exit code and the checkpoint decide success, not tmux.** A tmux session
disappears when the machine reboots, when the process is killed, and when the job
finished; only the last of those is success. `status` reads a `status.tsv` written
by the job wrapper (start/end, exit code, reason) and requires a checkpoint that
is actually on disk. tmux being gone with no end record is `unknown`, never
`succeeded`.

**An unreachable ssh is not a failed run.** ssh exits 255 for its own failures;
that is reported as a connection error and never written down as a job failure,
because the training may be running perfectly on the other side of a broken
network.

**Nothing secret is written down.** Key-based ssh only (`BatchMode=yes`): a
connection that would prompt fails instead of hanging, and the tool keeps no copy
of a key or token beyond the path in the config file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10, which the repository supports
    import tomli as tomllib  # type: ignore[no-redef]

REPO = Path(__file__).resolve().parents[1]

#: Where ssh writes the host key it has seen. Inside the workspace on purpose: a
#: missing known_hosts would otherwise be re-accepted on every call.
KNOWN_HOSTS = REPO / ".remote" / "known_hosts"

#: Files that must not travel to a training box. `.git` is excluded because it is
#: 235 MB in this checkout (a handoff branch carries a checkpoint) and the
#: provenance it would provide is recorded in `CODE_VERSION.json` instead.
#: `deploy/fsm/target` is a 900 MB Rust build directory that training never reads.
EXCLUDES = (
    ".git", ".venv", "logs", "out", "outputs", "runs",
    "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", "*.egg-info",
    ".DS_Store", "MUJOCO_LOG.TXT", "docs/media", "deploy/fsm/target",
    ".remote", ".remote.toml",
)

#: Paths excluded from the tar stream that pushes a release. Same intent as
#: EXCLUDES, spelled for bsdtar: the patterns are matched against the archive
#: names, which start with `./`.
TAR_EXCLUDES = (
    "./.git", "./.venv", "./logs", "./out", "./outputs", "./runs",
    "./.remote", "./.remote.toml", "./docs/media", "./deploy/fsm/target",
    "./.pytest_cache", "./.ruff_cache", "./MUJOCO_LOG.TXT", "./.DS_Store",
    # A task's own `out/` and `media/` are exports and choreography: 130 MB in
    # this checkout, none of it read by training. `.gitignore` covers them at
    # every depth; tar needs the depth spelled out.
    "./tasks/*/*/out", "./tasks/*/*/media", "./tasks/*/*/.cache",
)

#: Files uploaded beside the code so a release explains itself.
PROVENANCE_FILES = ("CODE_VERSION.json", "CODE.patch", "MANIFEST.sha256")

STATUSES = ("queued", "running", "succeeded", "failed", "stopped", "unknown")

#: ssh's own exit code. A remote command that happens to exit 255 cannot be told
#: apart, which is why the wrapper records its own exit code in `status.tsv`.
SSH_ERROR = 255

CONFIG_EXAMPLE = """\
# Copy to .remote.toml (git-ignored) or ~/.config/jumper/remote.toml.
# Nothing here is secret: `identity` is a path, not a key.

# Either an ssh alias from ~/.ssh/config ...
host = "autodl"

# ... or the connection details directly. Use one or the other, not both.
# hostname = "connect.xxx.seetacloud.com"
# port = 24227
# user = "root"
# identity = "~/.ssh/id_ed25519"
# known_hosts = ".remote/known_hosts"

workdir = "/root/autodl-tmp/jumper"      # absolute; keep it on the data disk
python = "/root/miniconda3/bin/python"   # interpreter that creates .venv
results = "~/jumper-runs"                # local download root
max_hours = 6.0                          # default wall-clock limit per job

# Training defaults, overridden per submit.
backend = "warp"
device = "cuda:0"
num_envs = 64
tensorboard_port = 6006
"""


class RemoteTrainError(RuntimeError):
    """A failure this tool understands and can explain."""


class Unreachable(RemoteTrainError):
    """ssh could not reach the host. Never a job failure."""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Where the box is, where the code goes, and what a job defaults to."""

    host: str = ""
    hostname: str = ""
    port: int = 22
    user: str = "root"
    identity: str = ""
    known_hosts: str = str(KNOWN_HOSTS)
    workdir: str = "/root/autodl-tmp/jumper"
    python: str = "python3"
    results: str = "~/jumper-runs"
    max_hours: float = 6.0
    backend: str = "warp"
    device: str = "cuda:0"
    num_envs: int = 64
    tensorboard_port: int = 6006
    pip_index: str = ""
    mujoco_gl: str = "egl"
    extra_ssh_options: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if bool(self.host) == bool(self.hostname):
            raise RemoteTrainError(
                "config: set exactly one of `host` (an ssh alias) or `hostname`"
            )
        if not self.workdir.startswith("/"):
            raise RemoteTrainError(f"config: workdir must be absolute, got {self.workdir!r}")
        if self.workdir.rstrip("/") in ("", "/"):
            raise RemoteTrainError("config: workdir must be a directory, not /")
        if self.max_hours <= 0:
            raise RemoteTrainError("config: max_hours must be positive")

    @property
    def results_dir(self) -> Path:
        return Path(os.path.expanduser(self.results))


def _config_paths(explicit: str | None) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    env = os.environ.get("JUMPER_REMOTE_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths += [Path(".remote.toml"), Path.home() / ".config" / "jumper" / "remote.toml"]
    return paths


def load_config(explicit: str | None = None) -> Config:
    for path in _config_paths(explicit):
        if path.is_file():
            raw = tomllib.loads(path.read_text("utf-8"))
            known = {f.name for f in Config.__dataclass_fields__.values()}
            unknown = sorted(set(raw) - known)
            if unknown:
                raise RemoteTrainError(f"{path}: unknown keys {unknown}")
            cfg = Config(**raw)
            cfg.validate()
            return cfg
    raise RemoteTrainError(
        "no config found. Copy the template and edit it:\n"
        "  cp tools/remote_train.example.toml .remote.toml"
    )


# --------------------------------------------------------------------------
# ssh / rsync
# --------------------------------------------------------------------------


def _q(value: object) -> str:
    return shlex.quote(str(value))


class Remote:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def base(self, tty: bool = False) -> list[str]:
        cfg = self.cfg
        known = Path(os.path.expanduser(cfg.known_hosts))
        known.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
            "-o", f"UserKnownHostsFile={known}",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if tty:
            cmd.append("-t")
        for opt in cfg.extra_ssh_options:
            cmd += ["-o", opt]
        if cfg.identity:
            cmd += ["-i", os.path.expanduser(cfg.identity), "-o", "IdentitiesOnly=yes"]
        if cfg.host:
            cmd.append(cfg.host)
        else:
            cmd += ["-p", str(cfg.port), f"{cfg.user}@{cfg.hostname}"]
        return cmd

    def run(
        self, script: str, *, timeout: int = 120, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        try:
            proc = subprocess.run(
                self.base() + ["bash -s"],
                input=script,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteTrainError(f"ssh timed out after {timeout}s") from exc
        except FileNotFoundError as exc:
            raise RemoteTrainError("ssh is not installed locally") from exc
        if proc.returncode == SSH_ERROR:
            raise Unreachable(f"cannot reach {self.label}: {proc.stderr.strip()}")
        if check and proc.returncode != 0:
            raise RemoteTrainError(
                f"remote command failed (rc={proc.returncode}):\n"
                f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
            )
        return proc

    def ssh_command(self) -> str:
        """The ssh rsync should use.

        rsync runs ssh itself, and its default ssh does not read the
        `UserKnownHostsFile` this tool pinned -- on a first connection it dies
        with `Host key verification failed` before transferring a byte. Passing
        the same options keeps one answer to "which host, which key".
        """
        return " ".join(_q(part) for part in self.base()[:-1])

    @property
    def target(self) -> str:
        """`user@host` for scp/rsync -- the port travels in `-e ssh -p`."""
        return self.cfg.host or f"{self.cfg.user}@{self.cfg.hostname}"

    @property
    def label(self) -> str:
        """How to say where this box is in a message or a report."""
        if self.cfg.host:
            return self.cfg.host
        return f"{self.cfg.user}@{self.cfg.hostname}:{self.cfg.port}"

    def rsync(
        self,
        sources: list[str],
        dest: str,
        *,
        delete: bool = False,
        extra: tuple[str, ...] = (),
        check: bool = True,
    ) -> None:
        """Copy local sources to a remote directory.

        `-a` keeps mtimes: `--resume` picks the newest checkpoint by mtime, so a
        copy that rewrote them would silently change which run is continued.
        """
        cmd = ["rsync", "-a", "--partial", "-e", self.ssh_command()] + list(extra)
        if delete:
            cmd.append("--delete")
        for pattern in EXCLUDES:
            cmd += ["--exclude", pattern]
        cmd += list(sources) + [dest]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == SSH_ERROR:
            raise Unreachable(f"rsync could not reach {self.label}: {proc.stderr.strip()}")
        if proc.returncode != 0:
            if check:
                raise RemoteTrainError(f"rsync failed (rc={proc.returncode}): {proc.stderr.strip()}")
            return

    def push_tree(self, source: Path, dest: str, *, clean: bool = True) -> None:
        """Send a directory as one tar stream over ssh.

        Not rsync on this Mac: the system rsync is openrsync 2.6.9, which fails
        on this checkout with `mkpath: Operation not permitted` before sending a
        byte. A tar stream is one pipe, has no protocol to disagree about, and
        over a slow uplink is faster than rsync's per-file handshake. rsync is
        still used for the download, where its resumability matters.
        """
        dest_q = _q(dest)
        clean_part = f"rm -rf {dest_q}; " if clean else ""
        remote_cmd = (
            f"set -euo pipefail; {clean_part}mkdir -p {dest_q}; "
            f"tar -xzf - -C {dest_q}"
        )
        tar_cmd = [
            "tar", "-czf", "-", "-C", str(source),
            *[f"--exclude={pattern}" for pattern in TAR_EXCLUDES], ".",
        ]
        env = dict(os.environ, COPYFILE_DISABLE="1")
        tar = subprocess.Popen(
            tar_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        ssh = subprocess.Popen(
            self.base() + [remote_cmd],
            stdin=tar.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert tar.stdout is not None
        tar.stdout.close()
        _out, err = ssh.communicate()
        tar_err = tar.stderr.read() if tar.stderr else b""
        rc = tar.wait()
        if rc != 0:
            raise RemoteTrainError(f"tar failed (rc={rc}): {tar_err.decode()[-800:]}")
        if ssh.returncode == SSH_ERROR:
            raise Unreachable(f"cannot reach {self.label}: {err.decode()[-800:]}")
        if ssh.returncode != 0:
            raise RemoteTrainError(
                f"remote extract failed (rc={ssh.returncode}): {err.decode()[-800:]}"
            )

    def remote_spec(self, path: str, *, directory: bool = True) -> str:
        """`host:path/` for a directory, `host:path` for one file."""
        suffix = "/" if directory else ""
        return f"{self.target}:{path.rstrip('/')}{suffix}"

    def rsync_from(
        self,
        remote_path: str,
        dest: Path,
        extra: tuple[str, ...] = (),
        *,
        directory: bool = True,
    ) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync", "-a", "--partial", "-e", self.ssh_command(),
            *extra, self.remote_spec(remote_path, directory=directory),
            str(dest) + "/",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == SSH_ERROR:
            raise Unreachable(f"rsync could not reach {self.label}: {proc.stderr.strip()}")
        if proc.returncode != 0:
            raise RemoteTrainError(f"rsync failed (rc={proc.returncode}): {proc.stderr.strip()}")

    def stream(self, script: str, *, tty: bool) -> int:
        try:
            return subprocess.run(
                self.base(tty=tty) + ["bash -s"], input=script, text=True, check=False
            ).returncode
        except KeyboardInterrupt:
            return 0
        except FileNotFoundError as exc:
            raise RemoteTrainError("ssh is not installed locally") from exc


# --------------------------------------------------------------------------
# git provenance
# --------------------------------------------------------------------------


def git(*args: str) -> str:
    proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RemoteTrainError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def git_provenance() -> dict:
    status = git("status", "--porcelain")
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "patch": git("diff", "HEAD"),
        "untracked": [line[3:].strip() for line in status.splitlines() if line.startswith("?? ")],
        "status": status,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_id(prov: dict, label: str | None) -> str:
    if label:
        return label
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}-{prov['head'][:7]}"
    if prov["dirty"]:
        base += "-dirty-" + sha256_bytes(prov["patch"].encode())[:8]
    return base


# --------------------------------------------------------------------------
# remote paths
# --------------------------------------------------------------------------


def run_dir(cfg: Config, job_id: str) -> str:
    return f"{cfg.workdir}/runs/{job_id}"


def release_dir(cfg: Config, release: str) -> str:
    return f"{cfg.workdir}/releases/{release}"


def venv_python(cfg: Config) -> str:
    return f"{cfg.workdir}/.venv/bin/python"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


CHECK_SCRIPT = r"""
set -u
line() { printf '%s=%s\n' "$1" "$2"; }
line HOST "$(hostname)"
line KERNEL "$(uname -sr)"
line NPROC "$(nproc)"
line MEM_GB "$(free -g | awk '/^Mem:/{print $2}')"
line DISK_WORKDIR "$(df -Ph __PARENT__ 2>/dev/null | tail -1 | awk '{print $4}')"
line DISK_ROOT "$(df -Ph / | tail -1 | awk '{print $4}')"
line WORKDIR_EXISTS "$([ -d __WORKDIR__ ] && echo yes || echo no)"
line WORKDIR_WRITABLE "$( { [ -w __WORKDIR__ ] || [ -w __PARENT__ ]; } && echo yes || echo no)"
line TMUX "$(tmux -V 2>/dev/null || echo missing)"
line RSYNC "$(rsync --version 2>/dev/null | head -1 || echo missing)"
line PYTHON "$(__PY__ -V 2>&1 || echo missing)"
line VENV "$(__VENV_PY__ -V 2>&1 || echo missing)"
line GPU "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo missing)"
line GPU_MEM_MIB "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)"
line GPU_USED_MIB "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
line DRIVER "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo missing)"
echo "GPU_APPS_BEGIN"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
echo "GPU_APPS_END"
echo "TORCH_BEGIN"
__VENV_PY__ - <<'PY' 2>/dev/null || echo "TORCH=missing"
import torch
print("TORCH=" + torch.__version__)
print("TORCH_CUDA=" + str(torch.version.cuda))
print("CUDA_AVAILABLE=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("DEVICE=" + torch.cuda.get_device_name(0))
    print("DEVICE_CAP=" + str(torch.cuda.get_device_capability(0)))
PY
echo "TORCH_END"
echo "IMPORTS_BEGIN"
__VENV_PY__ - <<'PY' 2>/dev/null || echo "IMPORTS=missing"
for name in ("mujoco", "mujoco_warp", "warp", "mjrl", "mjlab", "rsl_rl", "tasks"):
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "ok")
        print(name + "=" + str(version) + "@" + str(getattr(mod, "__file__", "?")))
    except Exception as exc:  # noqa: BLE001 - the report is the point
        print(name + "=IMPORT_FAIL:" + type(exc).__name__ + ":" + str(exc)[:80])
PY
echo "IMPORTS_END"
"""


def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    script = (
        CHECK_SCRIPT.replace("__WORKDIR__", _q(cfg.workdir))
        .replace("__PARENT__", _q(str(Path(cfg.workdir).parent)))
        .replace("__VENV_PY__", _q(venv_python(cfg)))
        .replace("__PY__", _q(cfg.python))
    )
    try:
        out = remote.run(script, timeout=240).stdout
    except Unreachable as exc:
        print(f"connection error: {exc}", file=sys.stderr)
        return 2
    info = parse_kv(out)
    gpu_apps = section(out, "GPU_APPS_BEGIN", "GPU_APPS_END").strip()
    torch_info = parse_kv(section(out, "TORCH_BEGIN", "TORCH_END"))
    imports = parse_module_versions(section(out, "IMPORTS_BEGIN", "IMPORTS_END"))

    rows = [
        ("host", info.get("HOST", "?")),
        ("kernel", info.get("KERNEL", "?")),
        ("cpus / RAM", f"{info.get('NPROC', '?')} / {info.get('MEM_GB', '?')} GiB"),
        ("data disk free", info.get("DISK_WORKDIR", "?")),
        ("system disk free", info.get("DISK_ROOT", "?")),
        ("workdir", (f"{cfg.workdir} (exists={info.get('WORKDIR_EXISTS')}, "
                     f"writable={info.get('WORKDIR_WRITABLE')})")),
        ("tmux / rsync", f"{info.get('TMUX', '?')} / {info.get('RSYNC', '?')}"),
        ("python", info.get("PYTHON", "?")),
        ("venv python", info.get("VENV", "?")),
        ("GPU", f"{info.get('GPU', '?')} ({info.get('GPU_MEM_MIB', '?')} MiB)"),
        ("GPU used", f"{info.get('GPU_USED_MIB', '?')} MiB"),
        ("driver", info.get("DRIVER", "?")),
        ("torch", (f"{torch_info.get('TORCH', '?')} "
                   f"cuda={torch_info.get('TORCH_CUDA', '?')} "
                   f"available={torch_info.get('CUDA_AVAILABLE', '?')}")),
        ("device", torch_info.get("DEVICE", "-")),
    ]
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")
    print(f"{'GPU processes':<{width}}  {'; '.join(gpu_apps.split()) or 'none'}")
    if imports:
        print(f"{'imports':<{width}}  " + ", ".join(
            f"{key}:{value.split('@')[0]}" for key, value in imports.items()
        ))

    problems = []
    if info.get("VENV", "missing") == "missing":
        problems.append("no .venv — run `bootstrap`")
    if torch_info.get("CUDA_AVAILABLE") != "True":
        problems.append("torch cannot see CUDA — check the driver or the torch build")
    for name in ("mujoco", "mujoco_warp", "warp", "mjrl", "mjlab", "tasks"):
        if name not in imports or imports[name].startswith("IMPORT_FAIL"):
            problems.append(f"import {name} failed: {imports.get(name, 'missing')}")
    if info.get("TMUX", "missing") == "missing":
        problems.append("tmux missing — apt-get install -y tmux on the box")
    if info.get("WORKDIR_WRITABLE") != "yes":
        problems.append(f"workdir {cfg.workdir} is not writable")
    if problems:
        print("\nNOT READY:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nready: connection, GPU, imports and tools verified")
    return 0


def parse_kv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match:
            out[match.group(1)] = match.group(2).strip()
    return out


def parse_module_versions(text: str) -> dict:
    """Parse `name=version@path` lines, where `name` is a module name.

    `parse_kv` deliberately accepts only UPPER_CASE keys, so it drops every
    module name; this is the reader for the import probe.
    """
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.replace("_", "").isalnum() and value:
            out[key] = value
    return out


def section(text: str, begin: str, end: str) -> str:
    if begin not in text:
        return ""
    return text.split(begin, 1)[1].split(end, 1)[0]


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


def cmd_sync(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    tmpdir = None
    prov = {"head": "", "branch": "", "dirty": False, "patch": "", "untracked": [], "status": ""}
    source = REPO
    try:
        if args.rev:
            prov = {"head": git("rev-parse", args.rev).strip(), "branch": args.rev,
                    "dirty": False, "patch": "", "untracked": [], "status": ""}
            base = REPO / ".remote"
            base.mkdir(exist_ok=True)
            tmpdir = tempfile.mkdtemp(prefix="release-", dir=str(base))
            archive = subprocess.run(
                ["git", "-C", str(REPO), "archive", "--format=tar", args.rev],
                capture_output=True,
                check=False,
            )
            if archive.returncode != 0:
                raise RemoteTrainError(f"git archive {args.rev} failed")
            subprocess.run(["tar", "-x", "-C", tmpdir], input=archive.stdout, check=True)
            source = Path(tmpdir)
        else:
            prov = git_provenance()

        rel = release_id(prov, args.label)
        remote.run(
            f"mkdir -p {_q(release_dir(cfg, rel))} {_q(cfg.workdir)}/releases {_q(cfg.workdir)}/runs"
        )
        print(f"syncing {source} -> {remote.label}:{release_dir(cfg, rel)}")
        remote.push_tree(source, release_dir(cfg, rel))

        staging = Path(tempfile.mkdtemp(prefix="prov-", dir=str(REPO / ".remote")))
        try:
            version = {
                "release": rel,
                "git_head": prov["head"],
                "git_branch": prov["branch"],
                "git_dirty": prov["dirty"],
                "git_status": prov["status"],
                "untracked": prov["untracked"],
                "patch_sha256": sha256_bytes(prov["patch"].encode()),
                "created_at": now_iso(),
                "created_from": str(REPO),
            }
            (staging / "CODE_VERSION.json").write_text(
                json.dumps(version, indent=2, sort_keys=True) + "\n", "utf-8"
            )
            (staging / "CODE.patch").write_text(prov["patch"], "utf-8")
            extra = {name: sha256_file(staging / name) for name in PROVENANCE_FILES[:2]}
            remote.push_tree(staging, release_dir(cfg, rel), clean=False)

            manifest = remote.run(
                f"cd {_q(release_dir(cfg, rel))} && "
                "find . -type f -print0 | sort -z | xargs -0 sha256sum"
            ).stdout
            verify_manifest(source, manifest, extra_hashes=extra)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        size = remote.run(f"du -sh {_q(release_dir(cfg, rel))} | cut -f1").stdout.strip()
        count = len([line for line in manifest.splitlines() if line.strip()])
        print(f"release {rel}: {count} files, {size} on the box")
        print(f"  git {prov['head'][:12]}{' (dirty: patch recorded)' if prov['dirty'] else ''}")
        return 0
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def verify_manifest(root: Path, manifest: str, extra_hashes: dict | None = None) -> None:
    """Check every byte the box received against this checkout.

    rsync is trusted for the transfer, not for the contents: an over-eager
    exclude, a stale edit, or a half-written asset all exit 0. The provenance
    files are checked by hash because they were uploaded from a staging
    directory that no longer exists here.
    """
    extra_hashes = extra_hashes or {}
    missing, wrong = [], []
    for line in manifest.splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        name = rel.split("/")[-1]
        if name.startswith("._"):
            continue
        if name in extra_hashes:
            if extra_hashes[name] != digest:
                wrong.append(f"{rel} (provenance file differs)")
            continue
        path = root / rel.removeprefix("./")
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != digest:
            wrong.append(rel)
    if missing or wrong:
        raise RemoteTrainError(
            "the box and this checkout disagree after sync:\n  "
            + "\n  ".join(f"missing: {item}" for item in missing[:10])
            + "\n  ".join(f"differs: {item}" for item in wrong[:10])
        )


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def cmd_bootstrap(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    rel = args.release or latest_release(remote, cfg)
    if not rel:
        raise RemoteTrainError("no release yet — run `sync` first")
    index = f"-i {_q(cfg.pip_index)} " if cfg.pip_index else ""
    # `network_turbo` is not sourced here on purpose: on AutoDL it proxies
    # GitHub/HuggingFace and makes PyPI *slower*, which is how a first bootstrap
    # stalls for half an hour. A mirror is configured explicitly instead.
    script = f"""
set -euo pipefail
mkdir -p {_q(cfg.workdir)}
cd {_q(cfg.workdir)}
if [ ! -x .venv/bin/python ]; then
  # --system-site-packages reuses the image's torch (2.8+cu128 here) instead of
  # downloading ~3 GB over the instance's uplink. `check` verifies the CUDA
  # build actually works; nothing from the image can shadow this repository's
  # mjlab/rsl_rl because pip's editable install wins over site-packages.
  {_q(cfg.python)} -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -U pip setuptools wheel {index}
# Pin the torch the image already provides. Without this, pip sees the project's
# torchvision>=0.5 and the newest torchvision wants a newer torch: it would
# download another multi-GB torch instead of using the image's CUDA build.
TORCH_VER=$({_q(cfg.python)} -c 'import torch; print(torch.__version__.split("+")[0])')
TV_VER=$({_q(cfg.python)} -c 'import torchvision; print(torchvision.__version__.split("+")[0])' 2>/dev/null || true)
if [ -z "$TV_VER" ]; then
  case "$TORCH_VER" in
    2.8.*) TV_VER=0.23.0 ;;
    2.7.*) TV_VER=0.22.0 ;;
    2.9.*) TV_VER=0.24.0 ;;
  esac
fi
CONSTRAINTS=$(mktemp)
printf 'torch==%s\n' "$TORCH_VER" > "$CONSTRAINTS"
[ -n "$TV_VER" ] && printf 'torchvision==%s\n' "$TV_VER" >> "$CONSTRAINTS"
.venv/bin/python -m pip install {index}-c "$CONSTRAINTS" -e {_q(release_dir(cfg, rel))}
rm -f "$CONSTRAINTS"
"""
    print(f"bootstrapping .venv from release {rel} (several minutes)")
    proc = remote.run(script, timeout=3600, check=False)
    sys.stdout.write(proc.stdout[-4000:])
    sys.stderr.write(proc.stderr[-6000:])
    if proc.returncode != 0:
        return proc.returncode
    print("bootstrap finished; run `check` next")
    return 0


def latest_release(remote: Remote, cfg: Config) -> str:
    return remote.run(
        f"ls -1t {_q(cfg.workdir)}/releases 2>/dev/null | head -1", check=False
    ).stdout.strip()


# --------------------------------------------------------------------------
# job state
# --------------------------------------------------------------------------


def parse_status_tsv(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        event = {"event": parts[0]}
        for part in parts[1:]:
            key, _, value = part.partition("=")
            event[key] = value
        events.append(event)
    return events


def classify(
    events: list[dict], alive: bool, checkpoints: int, stop_requested: bool = False
) -> str:
    end = next((event for event in reversed(events) if event["event"] == "end"), None)
    if end is not None:
        reason = end.get("reason", "")
        try:
            rc = int(end.get("rc", "1"))
        except ValueError:
            rc = 1
        if reason == "stopped":
            return "stopped"
        if reason == "timeout":
            return "failed"
        if rc == 0 and checkpoints > 0:
            return "succeeded"
        return "failed"
    # A stop that took the whole process group with it leaves no `end` record:
    # the wrapper is killed before it can write one. That is a stop, not a crash.
    if stop_requested and not alive:
        return "stopped"
    if alive:
        return "running"
    if events:
        return "unknown"
    return "queued"


ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
RESUME_RE = re.compile(r"resuming from (\S+)")


def progress_from_log(text: str) -> dict:
    found = ITER_RE.findall(text)
    out = {}
    if found:
        out["iteration"], out["total"] = int(found[-1][0]), int(found[-1][1])
    match = RESUME_RE.search(text)
    if match:
        out["resuming_from"] = match.group(1)
    return out


def read_job(remote: Remote, cfg: Config, job_id: str) -> dict:
    rd = run_dir(cfg, job_id)
    script = f"""
set -u
RUN={_q(rd)}
if [ ! -d "$RUN" ]; then echo NO_SUCH_JOB; exit 0; fi
echo "META_BEGIN"; cat "$RUN/meta.json" 2>/dev/null; echo "META_END"
echo "STATUS_BEGIN"; cat "$RUN/status.tsv" 2>/dev/null; echo "STATUS_END"
echo "TAIL_BEGIN"; tail -n 30 "$RUN/logs/train.log" 2>/dev/null; echo "TAIL_END"
echo "PROGRESS_BEGIN"
grep -aE "Learning iteration [0-9]+/[0-9]+|resuming from|resumed at iteration"   "$RUN/logs/train.log" 2>/dev/null | tail -4
echo "PROGRESS_END"
echo "STOP=$( [ -f "$RUN/stop.requested" ] && echo yes || echo no )"
ALIVE=0
if [ -f "$RUN/pid" ]; then
  PID=$(cat "$RUN/pid" 2>/dev/null || echo 0)
  echo "PID=$PID"
  if [ "$PID" -gt 0 ] && kill -0 "$PID" 2>/dev/null; then ALIVE=1; fi
  PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' \n')
  if [ "$ALIVE" -eq 0 ] && [ -n "$PGID" ] \
     && ps -eo pgid= 2>/dev/null | grep -qw "$PGID"; then ALIVE=1; fi
fi
echo "ALIVE=$ALIVE"
echo "CKPTS_BEGIN"
find "$RUN/tree" -type f -name 'model_*.pt' -size +100k -printf '%T@ %s %p\\n' 2>/dev/null | sort -n
echo "CKPTS_END"
echo "GPU_USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
"""
    out = remote.run(script, timeout=120).stdout
    if "NO_SUCH_JOB" in out:
        raise RemoteTrainError(f"no such job: {job_id}")
    meta = json.loads(section(out, "META_BEGIN", "META_END") or "{}")
    events = parse_status_tsv(section(out, "STATUS_BEGIN", "STATUS_END"))
    tail = section(out, "TAIL_BEGIN", "TAIL_END")
    checkpoints = []
    for line in section(out, "CKPTS_BEGIN", "CKPTS_END").splitlines():
        if line.strip():
            mtime, size, path = line.split(maxsplit=2)
            checkpoints.append({"mtime": float(mtime), "size": int(size), "path": path})
    alive = "ALIVE=1" in out
    gpu = re.search(r"GPU_USED_MIB=(\d+)", out)
    return {
        "job_id": job_id,
        "meta": meta,
        "events": events,
        "status": classify(
            events, alive, len(checkpoints), stop_requested="STOP=yes" in out
        ),
        "alive": alive,
        "stop_requested": "STOP=yes" in out,
        "checkpoints": checkpoints,
        "tail": tail,
        "progress": progress_from_log(
            section(out, "PROGRESS_BEGIN", "PROGRESS_END") or tail
        ),
        "gpu_used_mib": int(gpu.group(1)) if gpu else 0,
        "dir": rd,
    }


def job_ids(remote: Remote, cfg: Config, limit: int = 20) -> list[str]:
    """Run directories under `runs/`, newest last.

    **Directories, not names.** `ls -1` also returns stray files (a redirected
    `pip.log` sat beside the runs on the trial box), and every caller then treats
    that file as a job whose record cannot be read. The directory test is the
    job's definition here.
    """
    out = remote.run(
        f"cd {_q(cfg.workdir)}/runs 2>/dev/null && "
        'for d in */; do [ -d "$d" ] && printf "%s\\n" "${d%/}"; done '
        f"| sort | tail -n {int(limit)}",
        check=False,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def active_jobs(remote: Remote, cfg: Config) -> list[dict]:
    out = remote.run(
        f"""
for d in {_q(cfg.workdir)}/runs/*/; do
  [ -d "$d" ] || continue
  if [ -f "$d/status.tsv" ] && grep -q '^end' "$d/status.tsv"; then continue; fi
  if [ -f "$d/pid" ] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then
    echo "$(basename "$d") $(cat "$d/pid")"
  fi
done
""",
        check=False,
    ).stdout
    return [
        {"job_id": parts[0], "pid": int(parts[1])}
        for parts in (line.split() for line in out.splitlines())
        if len(parts) == 2
    ]


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


def ensure_runtime(remote: Remote, cfg: Config, rel: str) -> None:
    """Point the shared .venv at this release, then prove where imports landed.

    `pip install -e` is the only way `mjrl`, `mjlab`, `rsl_rl` and `tasks` resolve
    here, and it points at one directory. Installing, then checking the path, is
    the difference between running the snapshot and running whatever was
    installed before it -- the second is invisible until a result cannot be
    reproduced.
    """
    script = f"""
set -euo pipefail
cd {_q(cfg.workdir)}
test -x .venv/bin/python || {{ echo "NO_VENV"; exit 3; }}
.venv/bin/python -m pip install --no-deps --no-build-isolation -q -e {_q(release_dir(cfg, rel))}
.venv/bin/python - <<'PY'
import mjrl, mjlab, rsl_rl, tasks
for mod in (mjrl, mjlab, rsl_rl, tasks):
    print(mod.__name__ + "=" + str(mod.__file__))
PY
"""
    proc = remote.run(script, timeout=600, check=False)
    if "NO_VENV" in proc.stdout:
        raise RemoteTrainError("no .venv on the box — run `bootstrap` first")
    if proc.returncode != 0:
        raise RemoteTrainError(f"editable install failed:\n{proc.stderr[-2000:]}")
    expected = release_dir(cfg, rel)
    for line in proc.stdout.splitlines():
        if "=" in line:
            name, _, path = line.partition("=")
            if not path.startswith(expected):
                raise RemoteTrainError(
                    f"{name} resolves to {path}, not {expected}: the job would run "
                    "code other than its snapshot"
                )


def job_root(cfg: Config, job_id: str) -> str:
    """Per-job working directory: symlinks to the release, but a real logs dir.

    Training writes `logs/<model>/<task>/...` relative to the working directory.
    Running in the shared release would make every job share one `logs` symlink,
    and a second submit would re-point the first job's outputs mid-run.
    """
    return f"{run_dir(cfg, job_id)}/root"


def build_job_sh(
    cfg: Config, job_id: str, rel: str, argv: list[str], seconds: int
) -> str:
    rd = run_dir(cfg, job_id)
    inner = " ".join(_q(arg) for arg in argv)
    python = venv_python(cfg)
    command = (
        # --foreground keeps timeout in the job's process group. Without it,
        # timeout creates its own group, so `stop`'s signal to the wrapper's
        # group never reaches the trainer -- which is how a stopped job kept
        # training while `status` said STOPPED.
        f"timeout --foreground --signal=INT --kill-after=60 {seconds} "
        f"{_q(python)} {inner}"
        if seconds > 0
        else f"{_q(python)} {inner}"
    )
    # `$'\t'` is a raw string here so the generated script gets a backslash-t,
    # which bash turns into the field separator. Formatting it into the f-string
    # below would bake a literal tab into this Python source instead.
    join_fields = r"""status() { local IFS=$'\t'; printf '%s\n' "$*" >> "$RUN/status.tsv"; }"""
    return f"""#!/usr/bin/env bash
# generated by tools/remote_train.py; a job directory is immutable once started.
set -uo pipefail
export MUJOCO_GL="${{MUJOCO_GL:-{cfg.mujoco_gl}}}"
export PYTHONUNBUFFERED=1
RUN={_q(rd)}
cd {_q(job_root(cfg, job_id))}
mkdir -p "$RUN/logs" "$RUN/eval" "$RUN/tree"
echo $$ > "$RUN/pid"
{join_fields}
status start "ts=$(date -Is)" "pid=$$"
# `stop` signals the group; without this trap the wrapper dies first and the job
# looks like a crash to every later `status`.
trap 'status end "ts=$(date -Is)" "rc=143" "reason=stopped"; exit 143' TERM INT
{command} >> "$RUN/logs/train.log" 2>&1
rc=$?
if [ -f "$RUN/stop.requested" ]; then reason=stopped
elif [ "$rc" -eq 124 ]; then reason=timeout
elif [ "$rc" -eq 0 ]; then reason=completed
else reason=exit; fi
status end "ts=$(date -Is)" "rc=$rc" "reason=$reason"
exit "$rc"
"""


def admit_job(active: list[dict], *, concurrent: bool, force: bool) -> None:
    """One GPU runs one job unless `--concurrent` says otherwise, capped at two.

    The cap is the point: two 3080-sized jobs are the most this card was
    measured to hold with headroom, and an unbounded `--concurrent` would turn a
    typo into a swap storm.
    """
    if force or not active:
        return
    if concurrent:
        if len(active) >= 2:
            raise RemoteTrainError(
                "already two managed jobs running: "
                + ", ".join(job["job_id"] for job in active)
                + "\n--concurrent allows at most two; stop one first"
            )
        return
    raise RemoteTrainError(
        "a job is already running on this instance: "
        + ", ".join(f"{job['job_id']} (pid {job['pid']})" for job in active)
        + "\none GPU runs one job by default; use --concurrent for a second"
    )


def submit_job(
    cfg: Config,
    *,
    remote: Remote,
    task: str,
    release: str,
    argv: list[str],
    max_hours: float,
    dry_run: bool,
    allow_busy: bool,
    force: bool,
    concurrent: bool = False,
) -> str:
    active = active_jobs(remote, cfg)
    admit_job(active, concurrent=concurrent, force=force)
    gpu = remote.run(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits "
        "2>/dev/null | head -5",
        check=False,
    ).stdout
    busy = []
    for line in gpu.splitlines():
        if not line.strip():
            continue
        _, _, used = line.partition(", ")
        try:
            if int(used) > 512:
                busy.append(line)
        except ValueError:
            continue
    if busy and not allow_busy and not (concurrent and active):
        raise RemoteTrainError(
            "the GPU already has processes using more than 512 MiB:\n  "
            + "\n  ".join(busy)
            + "\nthis tool does not stop unknown processes; use --allow-busy to proceed"
        )

    rel = release
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    seconds = int(max_hours * 3600)
    prov = git_provenance()
    meta = {
        "job_id": job_id,
        "release": rel,
        "task": task,
        "argv": argv,
        "command": " ".join(shlex.quote(arg) for arg in argv),
        "created_at": now_iso(),
        "max_seconds": seconds,
        "host": remote.label,
        "workdir": cfg.workdir,
        "git_head": prov["head"],
        "git_dirty": prov["dirty"],
        "seed": next((argv[index + 1] for index, arg in enumerate(argv)
                      if arg == "--seed"), None),
    }
    if dry_run:
        print(json.dumps(meta, indent=2))
        print("job.sh:\n" + build_job_sh(cfg, job_id, rel, argv, seconds))
        return job_id

    ensure_runtime(remote, cfg, rel)
    rd = run_dir(cfg, job_id)
    root = job_root(cfg, job_id)
    remote.run(
        f"""
set -euo pipefail
mkdir -p {_q(rd)}/logs {_q(rd)}/eval {_q(rd)}/tree {_q(root)}
rm -rf {_q(root)}/*
for entry in {_q(release_dir(cfg, rel))}/* {_q(release_dir(cfg, rel))}/.[!.]*; do
  [ -e "$entry" ] || continue
  name=$(basename "$entry")
  [ "$name" = logs ] && continue
  ln -sfn "$entry" {_q(root)}/"$name"
done
ln -sfn {_q(rd)}/tree {_q(root)}/logs
"""
    )
    with tempfile.TemporaryDirectory(prefix="job-") as tmp:
        staging = Path(tmp)
        (staging / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
        (staging / "job.sh").write_text(
            build_job_sh(cfg, job_id, rel, argv, seconds), "utf-8"
        )
        remote.push_tree(staging, rd, clean=False)

    pane = remote.run(
        f"cd {_q(rd)} && tmux new-session -d -s {_q('jumper-' + job_id)} 'bash job.sh' "
        f"&& sleep 1 && tmux list-panes -t {_q('jumper-' + job_id)} -F '#{{pane_pid}}'",
        timeout=60,
    ).stdout.strip()

    print(f"submitted {job_id}")
    print(f"  release {rel}")
    print(f"  command {meta['command']}")
    print(f"  remote  {rd}")
    print(f"  tmux    jumper-{job_id} (pane {pane or '?'})")
    print(f"  limit   {seconds // 60} min")
    print(f"\nnext: python tools/remote_train.py status {job_id}")
    return job_id


def default_argv(cfg: Config, args: argparse.Namespace) -> list[str]:
    argv = [
        "scripts/train.py",
        "--task", args.task,
        "--backend", args.backend or cfg.backend,
        "--device", args.device or cfg.device,
        "--num_envs", str(args.num_envs or cfg.num_envs),
        "--max-iterations", str(args.max_iterations),
        "--headless",
        "--tb-port", str(args.tensorboard_port or cfg.tensorboard_port),
    ]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.resume_from:
        argv += ["--resume", "--checkpoint", args.resume_from]
    if args.extra:
        argv += shlex.split(args.extra)
    return argv


def cmd_submit(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    if not args.release:
        raise RemoteTrainError("submit needs --release (the id printed by `sync`)")
    argv = default_argv(cfg, args)
    hours = args.max_hours if args.max_hours is not None else cfg.max_hours
    submit_job(
        cfg,
        remote=remote,
        task=args.task,
        release=args.release,
        argv=argv,
        max_hours=hours,
        dry_run=args.dry_run,
        allow_busy=args.allow_busy,
        force=args.force,
        concurrent=args.concurrent,
    )
    return 0


# --------------------------------------------------------------------------
# status / logs / stop
# --------------------------------------------------------------------------


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    ids = [args.job] if args.job else job_ids(remote, cfg, limit=args.limit)
    if not ids:
        print("no jobs on the box")
        return 0
    rows = [read_job(remote, cfg, job_id) for job_id in ids]
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    for job in rows:
        meta, events = job["meta"], job["events"]
        started = next((event.get("ts", "") for event in events if event["event"] == "start"), "")
        end = next((event for event in reversed(events) if event["event"] == "end"), {})
        print(f"{job['job_id']}  {job['status'].upper()}")
        print(f"  task      {meta.get('task', '?')}  release {meta.get('release', '?')}")
        print(f"  started   {started or '-'}  ended {end.get('ts', '-')}"
              + (f"  rc={end.get('rc')}" if end else ""))
        print(f"  progress  {job['progress'].get('iteration', '-')}/"
              f"{job['progress'].get('total', '-')} iterations")
        if job["checkpoints"]:
            newest = job["checkpoints"][-1]
            print(f"  ckpts     {len(job['checkpoints'])}  newest "
                  f"{Path(newest['path']).name} ({newest['size'] // 1024} KiB)")
        else:
            print("  ckpts     0")
        print(f"  gpu       {job['gpu_used_mib']} MiB used")
        if job["tail"].strip():
            print("  log tail")
            for line in job["tail"].rstrip().splitlines()[-4:]:
                print(f"    {line}")
        print()
    return 0


def cmd_logs(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    path = f"{run_dir(cfg, args.job)}/logs/train.log"
    flag = "-F" if args.follow else ""
    script = f"tail -n {args.lines} {flag} {_q(path)}"
    return remote.stream(script, tty=sys.stdin.isatty() and args.follow)


def cmd_stop(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    job = read_job(remote, cfg, args.job)
    if job["status"] in ("succeeded", "failed", "stopped"):
        print(f"{args.job} already ended: {job['status']}")
        return 0
    script = f"""
set -u
RUN={_q(run_dir(cfg, args.job))}
touch "$RUN/stop.requested"
printf 'stop\tts=%s\n' "$(date -Is)" >> "$RUN/status.tsv"
PID=$(cat "$RUN/pid" 2>/dev/null || echo 0)
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' \n')
[ -n "$PGID" ] || PGID=$PID
echo "signalling pid=$PID pgid=$PGID"
kill -INT -- -"$PGID" 2>/dev/null || echo "INT_FAILED"
for i in $(seq 1 {args.timeout}); do
  if ! ps -eo pgid= 2>/dev/null | grep -qw "$PGID"; then
    echo "EXITED after ${{i}}s"
    exit 0
  fi
  sleep 1
done
echo "still alive after {args.timeout}s; TERM group $PGID"
kill -TERM -- -"$PGID" 2>/dev/null || echo "TERM_FAILED"
sleep 5
if ps -eo pgid= 2>/dev/null | grep -qw "$PGID"; then
  echo "forcing KILL group $PGID"
  kill -KILL -- -"$PGID" 2>/dev/null || echo "KILL_FAILED"
  sleep 2
fi
if ps -eo pgid= 2>/dev/null | grep -qw "$PGID"; then echo "STOP_FAILED"; exit 3; fi
echo "STOPPED"
"""
    proc = remote.run(script, timeout=args.timeout + 40, check=False)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        raise RemoteTrainError(
            f"could not stop {args.job}: the process group is still alive. "
            "The job was started by this tool, so `pkill -f scripts/train.py` "
            "on the box is safe for it; nothing was killed blindly."
        )
    print(
        "\nstop requested. rsl_rl writes a checkpoint every save_interval (50 "
        "iterations here) and at the end of a completed run, not on SIGINT or "
        "SIGTERM; the iterations since the last checkpoint can be lost, and this "
        "command cannot promise a save."
    )
    return 0


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def fetch_plan(job: dict, full: bool) -> dict:
    """What `fetch` takes by default, and what `--all` takes.

    The default is the evidence a run is judged on -- the final checkpoint, the
    config and status, the log and the evaluation -- not the 31 intermediate
    checkpoints that made the last timed download 330 MB. `--all` keeps the old
    behaviour for anyone who wants the whole trajectory.
    """
    rd = job["dir"]
    if full:
        return {"mode": "all", "root": rd, "small": [], "tree": None,
                "tree_include": None, "release": job["meta"].get("release")}
    checkpoints = job.get("checkpoints") or []
    if not checkpoints:
        raise RemoteTrainError(f"{job['job_id']} has no checkpoint to fetch")
    final = Path(checkpoints[-1]["path"])
    return {
        "mode": "selective",
        "root": rd,
        "small": [f"{rd}/meta.json", f"{rd}/status.tsv", f"{rd}/job.sh"],
        "logs": f"{rd}/logs",
        "eval": f"{rd}/eval",
        "tree": str(final.parent),
        "tree_include": final.name,
        "tree_exclude": "model_*.pt",
        "release": job["meta"].get("release"),
    }


def _verify_fetched(remote: Remote, cfg: Config, rd: str, dest: Path) -> tuple[dict, str]:
    """Hash exactly the files that landed locally, against the box.

    The old version asked the box for every checkpoint in the tree; a selective
    fetch would then report the ones deliberately not taken as missing.
    """
    local = [
        path for path in dest.rglob("*")
        if path.is_file() and path.name not in ("SHA256SUMS", "FETCH.json")
    ]
    relatives = [str(path.relative_to(dest)) for path in local]
    if not relatives:
        raise RemoteTrainError("nothing was downloaded")
    listing = "\n".join(relatives)
    checks = remote.run(
        f"cd {_q(rd)} && printf '%s\\n' {_q(listing)} | tr '\\n' '\\0' "
        "| xargs -0 -r sha256sum",
        timeout=600,
        check=False,
    ).stdout
    result = verify_download(dest, checks, skip=set())
    (dest / "SHA256SUMS").write_text(checks, "utf-8")
    return result, checks


def cmd_fetch(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    job = read_job(remote, cfg, args.job)
    dest = cfg.results_dir / args.job
    dest.mkdir(parents=True, exist_ok=True)
    plan = fetch_plan(job, args.all)

    # Files still being written are skipped rather than copied half-written:
    # torch.save writes in place, so a fresh mtime is the only practical signal.
    fresh = remote.run(
        f"find {_q(plan['root'])}/tree -type f -name 'model_*.pt' -newermt '-10 seconds' "
        "2>/dev/null",
        check=False,
    ).stdout.split()
    if fresh and job["status"] == "running":
        print(
            f"skipping {len(fresh)} checkpoint(s) written in the last 10s: "
            + ", ".join(Path(path).name for path in fresh)
        )

    if plan["mode"] == "all":
        remote.rsync_from(plan["root"], dest)
    else:
        for source in plan["small"]:
            remote.rsync_from(source, dest, directory=False)
        remote.rsync_from(plan["logs"], dest / "logs")
        if remote.run(f"test -d {_q(plan['eval'])}", check=False).returncode == 0:
            remote.rsync_from(plan["eval"], dest / "eval")
        relative_tree = str(Path(plan["tree"]).relative_to(plan["root"]))
        remote.rsync_from(
            plan["tree"], dest / relative_tree,
            extra=(
                f"--include={plan['tree_include']}",
                f"--exclude={plan['tree_exclude']}",
            ),
        )

    result, _ = _verify_fetched(remote, cfg, plan["root"], dest)

    # The release's provenance is not inside the run directory, so it is copied
    # explicitly: "which code produced this" is part of the download contract.
    release = plan.get("release")
    if release:
        proc = remote.run(
            f"cat {_q(release_dir(cfg, release))}/CODE_VERSION.json", check=False
        )
        if proc.returncode == 0 and proc.stdout.strip():
            (dest / "CODE_VERSION.json").write_text(proc.stdout, "utf-8")

    (dest / "FETCH.json").write_text(json.dumps({
        "job_id": args.job,
        "fetched_at": now_iso(),
        "mode": plan["mode"],
        "status": job["status"],
        "exit": next((e for e in reversed(job["events"]) if e["event"] == "end"), None),
        "checkpoints": [ckpt["path"] for ckpt in job["checkpoints"]],
        "skipped_in_flight": [Path(path).name for path in fresh],
        "verified": result["verified"],
        "missing": result["missing"],
        "local": str(dest),
    }, indent=2) + "\n", "utf-8")
    print(f"fetched {args.job} ({'all' if args.all else 'selective'}) -> {dest}")
    print(f"  verified {result['verified']} file(s) with SHA-256")
    if plan["mode"] == "selective":
        print("  (final checkpoint + logs + config + eval; --all for every checkpoint)")
    if result["missing"]:
        print(f"  MISSING {result['missing']}")
    if args.post:
        _run_post(dest)
    return 0


def _pipeline_python() -> str:
    """The interpreter that can run the pipeline: this repo's venv if present.

    `remote_train.py` itself is standard-library only and may be run by the
    system python3 (3.9 on this Mac), but the pipeline imports `tomllib` and
    torch, so it must use the venv python that runs training and replay. A
    missing `.venv` falls back to the current interpreter, which produces an
    honest import error in the pipeline instead of a silent skip.
    """
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def _run_post(dest: Path) -> int:
    """Hand a verified download to the post-fetch pipeline.

    Best effort by design. `fetch`'s contract is the verified copy; a pipeline
    that cannot start (no config, no Feishu target, a renderer that disagrees
    with the Mac) must not turn a good download into a failed command. The
    pipeline records its own stage statuses in its state file, so what it could
    not finish is visible there and from `postfetch status`.
    """
    argv = [_pipeline_python(), "-m", "tools.postfetch", "run", "--job-dir", str(dest)]
    proc = subprocess.run(argv, cwd=REPO, check=False)
    if proc.returncode != 0:
        print(f"  post-fetch pipeline exited {proc.returncode}; see `python -m tools.postfetch status`")
    return proc.returncode


def verify_download(root: Path, manifest: str, skip: set[str]) -> dict:
    verified, missing, wrong = 0, [], []
    for line in manifest.splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if Path(rel).name in skip:
            continue
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != digest:
            wrong.append(rel)
        else:
            verified += 1
    if wrong:
        raise RemoteTrainError(
            "SHA-256 mismatch after download (truncated or rewritten):\n  "
            + "\n  ".join(wrong[:10])
        )
    return {"verified": verified, "missing": missing}


# --------------------------------------------------------------------------
# eval / resume
# --------------------------------------------------------------------------


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    job = read_job(remote, cfg, args.job)
    if not job["checkpoints"]:
        raise RemoteTrainError(f"{args.job} has no checkpoint to evaluate")
    ckpt = args.checkpoint or job["checkpoints"][-1]["path"]
    meta = job["meta"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    local = cfg.results_dir / args.job / "eval"
    local.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        return _eval_metrics(cfg, remote, args, meta, ckpt, stamp, local)

    seconds = max(120, args.steps * 2)
    log = f"{run_dir(cfg, args.job)}/eval/play-{stamp}.log"
    script = f"""
set -u
cd {_q(release_dir(cfg, meta['release']))}
MUJOCO_GL={_q(cfg.mujoco_gl)} timeout --foreground --signal=INT --kill-after=30 \
  {seconds} \
  {_q(venv_python(cfg))} scripts/play.py \
  --task {_q(meta['task'])} --checkpoint {_q(ckpt)} --backend native --device cpu \
  --num_envs 1 --steps {args.steps} --headless > {_q(log)} 2>&1
rc=$?
echo "EVAL_RC=$rc"
echo "EVAL_LOG={log}"
echo "EVAL_BEGIN"
tail -n 40 {_q(log)}
echo "EVAL_END"
"""
    proc = remote.run(script, timeout=seconds + 180, check=False)
    body = section(proc.stdout, "EVAL_BEGIN", "EVAL_END")
    match = re.search(r"EVAL_RC=(\d+)", proc.stdout)
    rc = int(match.group(1)) if match else -1
    reached = "reached --steps" in body
    failed = "Traceback" in body or "Error" in body or "error" in body
    report = {
        "job_id": args.job,
        "checkpoint": ckpt,
        "steps": args.steps,
        "rc": rc,
        "replay_completed": reached,
        "errors_in_log": failed,
        "log_on_box": log,
        "finished_at": now_iso(),
    }
    (local / f"play-{stamp}.log").write_text(body, "utf-8")
    (local / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(f"eval {args.job}: rc={rc} replay_completed={reached}")
    for line in body.rstrip().splitlines()[-6:]:
        print(f"  {line}")
    return 0 if (rc == 0 and reached) else 1


def _eval_metrics(
    cfg: Config,
    remote: Remote,
    args: argparse.Namespace,
    meta: dict,
    ckpt: str,
    stamp: str,
    local: Path,
) -> int:
    """Run tools/replay_eval.py on the box: standing, straight, turning."""
    rd = run_dir(cfg, args.job)
    log = f"{rd}/eval/metrics-{stamp}.log"
    out = f"{rd}/eval/metrics-{stamp}.json"
    seconds = max(300, args.episodes * args.steps * 6)
    script = f"""
set -u
cd {_q(release_dir(cfg, meta['release']))}
MUJOCO_GL={_q(cfg.mujoco_gl)} timeout --foreground --signal=INT --kill-after=60 \
  {seconds} \
  {_q(venv_python(cfg))} tools/replay_eval.py \
  --task {_q(meta['task'])} --checkpoint {_q(ckpt)} \
  --episodes {args.episodes} --steps {args.steps} --seed {args.seed} \
  --scenarios {_q(args.scenarios)} --out {_q(out)} > {_q(log)} 2>&1
rc=$?
echo "METRICS_RC=$rc"
echo "METRICS_BEGIN"
cat {_q(out)} 2>/dev/null
echo "METRICS_END"
echo "TAIL_BEGIN"
tail -n 12 {_q(log)}
echo "TAIL_END"
"""
    proc = remote.run(script, timeout=seconds + 180, check=False)
    body = section(proc.stdout, "METRICS_BEGIN", "METRICS_END").strip()
    match = re.search(r"METRICS_RC=(\d+)", proc.stdout)
    rc = int(match.group(1)) if match else -1
    tail = section(proc.stdout, "TAIL_BEGIN", "TAIL_END")
    if not body:
        print(f"metrics eval failed (rc={rc}); log tail:")
        for line in tail.rstrip().splitlines()[-8:]:
            print(f"  {line}")
        return 1
    report = json.loads(body)
    (local / f"metrics-{stamp}.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    (local / f"metrics-{stamp}.log").write_text(tail, "utf-8")
    print(f"metrics {args.job}  checkpoint {Path(ckpt).name}  seed={args.seed} "
          f"episodes={args.episodes} steps<={args.steps}")
    print(f"{'scenario':<10}{'falls':>7}{'survival':>10}{'lin err':>9}{'yaw err':>9}{'speed':>8}")
    for name, row in report["scenarios"].items():
        print(
            f"{name:<10}{row['falls']}/{row['episodes']:<5}"
            f"{row['mean_survival_steps']:>10.0f}{row['mean_lin_error']:>9.3f}"
            f"{row['mean_yaw_error']:>9.3f}{row['mean_speed']:>8.3f}"
        )
    return 0


def resume_argv(
    original: list[str], checkpoint: str, max_iterations: int | None
) -> list[str]:
    """Rebuild a job's arguments for a resumed run.

    The value after `--max-iterations` and `--checkpoint` has to be dropped with
    the flag. Removing only the flag produced `--num_envs 64 20`, which argparse
    rejects -- and the failure is only visible in the job's log, after it has
    already been submitted.
    """
    cleaned: list[str] = []
    skip_next = {"--checkpoint", "--max-iterations"}
    index = 0
    while index < len(original):
        arg = original[index]
        if arg in skip_next:
            index += 2
            continue
        if arg == "--resume":
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    cleaned += ["--resume", "--checkpoint", checkpoint]
    if max_iterations is not None:
        cleaned += ["--max-iterations", str(max_iterations)]
    return cleaned


def cmd_resume(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    job = read_job(remote, cfg, args.job)
    if not job["checkpoints"]:
        raise RemoteTrainError(f"{args.job} has no checkpoint to resume from")
    ckpt = args.checkpoint or job["checkpoints"][-1]["path"]
    original = job["meta"]
    cleaned = resume_argv(original["argv"], ckpt, args.max_iterations)
    hours = args.max_hours or (original.get("max_seconds", 0) / 3600) or cfg.max_hours
    print(f"resuming {args.job} from {ckpt}")
    submit_job(
        cfg,
        remote=remote,
        task=original["task"],
        release=original["release"],
        argv=cleaned,
        max_hours=hours,
        dry_run=False,
        allow_busy=True,
        force=False,
        concurrent=False,
    )
    return 0


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def cmd_list(cfg: Config, args: argparse.Namespace) -> int:
    remote = Remote(cfg)
    for job_id in job_ids(remote, cfg, limit=50):
        try:
            job = read_job(remote, cfg, job_id)
        except RemoteTrainError:
            continue
        started = next((e.get("ts", "") for e in job["events"] if e["event"] == "start"), "")
        print(f"{job_id}  {job['status']:<9}  {job['meta'].get('task', '?'):<18} "
              f"{started}  ckpts={len(job['checkpoints'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote_train.py",
        description="Run scripts/train.py on a remote GPU box over ssh + rsync.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CONFIG_EXAMPLE,
    )
    parser.add_argument("-c", "--config", help="config file (default: .remote.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="connection, disk, GPU, python, tmux and imports").set_defaults(
        func=cmd_check
    )

    p = sub.add_parser("sync", help="copy this checkout to a new immutable release")
    p.add_argument("--rev", help="export a git revision instead of the working tree")
    p.add_argument("--label", help="release id (default: timestamp + commit)")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("bootstrap", help="create .venv on the box and install the project")
    p.add_argument("--release", help="release to install (default: newest)")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("submit", help="start a training job in tmux")
    p.add_argument("--task", required=True)
    p.add_argument("--release", required=True, help="release id from `sync`")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--max-hours", type=float)
    p.add_argument("--backend")
    p.add_argument("--device")
    p.add_argument("--tensorboard-port", type=int)
    p.add_argument("--seed", type=int, help="seed the environment, torch and numpy")
    p.add_argument("--resume-from", help="checkpoint path on the box (implies --resume)")
    p.add_argument("--extra", help="extra scripts/train.py arguments, quoted")
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument(
        "--concurrent", action="store_true",
        help="allow a second managed job (max two); default is exclusive",
    )
    p.add_argument("--force", action="store_true", help="start despite another live job")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="job state, progress, GPU and log tail")
    p.add_argument("job", nargs="?")
    p.add_argument("--json", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs", help="read or follow a job's log")
    p.add_argument("job")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=60)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("fetch", help="download results and verify SHA-256")
    p.add_argument("job")
    p.add_argument(
        "--all", action="store_true",
        help="download every checkpoint (default: final model, logs, config and eval)",
    )
    p.add_argument(
        "--post", action="store_true",
        help="after the copy verifies, run tools/postfetch on it: evaluate, record a "
             "short clip, and notify. The copy stays the contract; a pipeline failure "
             "is recorded, not raised",
    )
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("eval", help="load a checkpoint in headless replay for N steps")
    p.add_argument("job")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--checkpoint", help="absolute path on the box (default: newest)")
    p.add_argument("--metrics", action="store_true",
                   help="tracking metrics via tools/replay_eval.py (stand/straight/turn)")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios", default="stand,straight,turn")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("resume", help="continue from a job's newest checkpoint")
    p.add_argument("job")
    p.add_argument("--checkpoint", help="absolute path on the box (default: newest)")
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--max-hours", type=float)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("stop", help="SIGINT the job, then TERM/KILL if needed")
    p.add_argument("job")
    p.add_argument("--timeout", type=int, default=15)
    p.set_defaults(func=cmd_stop)

    sub.add_parser("list", help="summarise recent jobs").set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except RemoteTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        return args.func(cfg, args)
    except Unreachable as exc:
        print(f"connection error: {exc}", file=sys.stderr)
        return 2
    except RemoteTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
