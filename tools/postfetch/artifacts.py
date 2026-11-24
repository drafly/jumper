"""What a fetched run directory says about itself, and whether it can be run here.

The pipeline must not assume a job is `jumper.flat`. It reads the task from the
training metadata, refuses to measure a task the configuration has no profile
for, and records the trained code revision beside the local one so a mismatch is
a stated incompatibility rather than a mysterious load error. Loading itself is
checked before the evaluation starts: the actor's layer shapes in the checkpoint
are compared with this checkout's agent config, so "the checkpoint is from a
different network" fails as a compatibility error instead of as a shape error
halfway through a rollout.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class ArtifactError(RuntimeError):
    """The fetched run cannot be turned into a job (missing or inconsistent)."""


@dataclass
class FetchedJob:
    """A verified fetched run, with the facts the pipeline needs to decide."""

    source_dir: Path
    job_id: str
    task: str
    checkpoint: Path
    checkpoint_sha256: str
    git_head: str
    code: dict
    meta: dict
    fetch: dict
    supported: bool
    reason: str
    compatible: bool
    compatibility_reason: str
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def local_git_head(repo: Path = REPO) -> str:
    """The commit this checkout is on, or "" outside a git checkout."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
        )
    except (OSError, FileNotFoundError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def newest_checkpoint(root: Path) -> Path:
    """The newest `model_*.pt` under a fetched run's `tree/`.

    Sorted by the iteration in the name rather than by mtime: a copy can set
    every mtime to the same second, and `model_100` sorts before `model_99`
    lexicographically, so the number is read as an int.
    """
    checkpoints = list(root.rglob("model_*.pt"))
    if not checkpoints:
        raise ArtifactError(f"no model_*.pt under {root}")
    return max(checkpoints, key=_iteration_of)


def _iteration_of(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ArtifactError(f"missing {path.name} in {path.parent}")
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc


def _check_actor_shapes(agent_cfg, checkpoint: Path) -> tuple[bool, str]:
    """Compare the checkpoint's actor layers with this checkout's agent config.

    The history of this repository includes an actor whose observation width
    changed; loading it against the wrong config fails deep inside torch with a
    shape mismatch that reads like a code bug. Here it is a compatibility
    result, with both sides named.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the repo
        return True, ""
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint is the same answer
        return False, f"checkpoint unreadable: {type(exc).__name__}: {exc}"
    state = raw.get("actor_state_dict") or raw.get("model_state_dict") or {}
    layers = [
        (name, weight.shape)
        for name, weight in state.items()
        if name.endswith(".weight") and weight.dim() == 2 and "normalizer" not in name
    ]
    if not layers:
        return False, "checkpoint holds no actor weight matrices"
    try:
        layers.sort(key=lambda item: int(item[0].rsplit(".", 2)[-2]))
    except ValueError:
        pass
    actor = getattr(agent_cfg, "actor", None)
    expected = getattr(actor, "hidden_dims", None)
    if expected is None:
        return True, ""
    hidden = [int(shape[0]) for _, shape in layers[:-1]]
    if tuple(hidden) != tuple(int(width) for width in expected):
        return False, (
            f"checkpoint actor hidden layers {hidden} do not match this checkout's "
            f"{list(expected)}"
        )
    return True, ""


def read_job(job_dir: str | Path, config) -> FetchedJob:
    """Read one fetched run directory into a job the pipeline can act on.

    The SHA-256 recorded in `FETCH.json` is not re-derived here; the pipeline
    hashes the checkpoint file it actually holds, which is what the unit key
    must be about. `FETCH.json` is used for the download verdict, not for
    trusting bytes.
    """
    source = Path(job_dir).expanduser().resolve()
    if not source.is_dir():
        raise ArtifactError(f"not a directory: {source}")
    fetch = _load_json(source / "FETCH.json")
    meta = _load_json(source / "meta.json")

    if not fetch.get("verified") or fetch.get("missing"):
        raise ArtifactError(
            f"{source.name}: the download did not verify ({fetch.get('verified', 0)} files, "
            f"missing {fetch.get('missing', [])})"
        )

    tree = source / "tree"
    if not tree.is_dir():
        raise ArtifactError(f"{source.name}: no tree/ directory; fetch did not copy results")
    checkpoint = newest_checkpoint(tree)
    sha = sha256_file(checkpoint)

    job_id = str(fetch.get("job_id") or meta.get("job_id") or source.name)
    task = str(meta.get("task") or "")
    if not task:
        raise ArtifactError(f"{job_id}: meta.json names no task")

    code_path = source / "CODE_VERSION.json"
    code = json.loads(code_path.read_text("utf-8")) if code_path.is_file() else {}
    git_head = str(code.get("git_head") or meta.get("git_head") or "")

    warnings: list[str] = []
    if code.get("git_dirty"):
        warnings.append("trained from a dirty working tree (CODE_VERSION.json)")

    supported, reason = True, ""
    profile = config.profile_for(task)
    if profile is None:
        supported = False
        known = ", ".join(sorted(config.profiles)) or "(none)"
        reason = f"task {task!r} has no evaluation profile; configured profiles: {known}"
    else:
        try:
            import tasks

            tasks.get(task)
        except Exception as exc:  # noqa: BLE001 - unregistered is the answer
            supported = False
            reason = f"task {task!r} is not registered in this checkout: {exc}"

    local = local_git_head()
    if supported and git_head and local and git_head != local:
        if config.allow_code_mismatch:
            warnings.append(f"trained from {git_head[:12]}, local checkout is {local[:12]}")
        else:
            supported = False
            reason = (
                f"code version mismatch: trained from {git_head[:12]}, local checkout is "
                f"{local[:12]} (set watch.allow_code_mismatch to measure anyway)"
            )

    compatible, compatibility_reason = True, ""
    if supported:
        try:
            import tasks

            agent_cfg = tasks.load_agent_cfg(task)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            compatible = False
            compatibility_reason = f"cannot load {task} agent config: {exc}"
        else:
            compatible, compatibility_reason = _check_actor_shapes(agent_cfg, checkpoint)

    return FetchedJob(
        source_dir=source,
        job_id=job_id,
        task=task,
        checkpoint=checkpoint,
        checkpoint_sha256=sha,
        git_head=git_head,
        code=code,
        meta=meta,
        fetch=fetch,
        supported=supported,
        reason=reason,
        compatible=compatible,
        compatibility_reason=compatibility_reason,
        warnings=warnings,
    )
