"""Per-unit status, idempotency, and crash recovery.

A *unit* is one checkpoint measured under one evaluation configuration. Its key
is a hash of the job id, the checkpoint's own SHA-256 and the evaluation
configuration version, so:

- fetching the same job twice, or a `watch` loop seeing the same directory on
  every pass, finds the unit already processed and does nothing;
- a changed threshold or step count is a different unit (the measurement
  changed), while a changed chat or video setting is not (the measurement did
  not);
- a process that dies mid-way leaves the unit's per-stage status on disk, and
  the next start resumes from the first stage that is not terminal.

The file is JSON rather than SQLite on purpose: it is small, a human may need to
read it while debugging a stuck unit, and an atomic rename is the whole
concurrency story together with the lock file beside it.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

#: The stages, in the order they run. `download` is recorded by the trigger
#: (the fetch already verified the copy) so a resumed unit can still say which
#: step completed before it died.
STAGES: tuple[str, ...] = ("download", "eval", "video", "notify")

#: A stage in one of these needs no further work from the watch loop. `blocked`
#: and `failed` are left out because both can be fixed by configuration or a
#: later attempt; `unsupported` is terminal -- retrying a task with no profile
#: would return the same answer.
TERMINAL_STATUSES: frozenset[str] = frozenset({"ok", "skipped", "unsupported"})


def now_iso() -> str:
    """Local time with an offset, so a person can line it up with their day."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def unit_key(job_id: str, checkpoint_sha256: str, eval_version: str) -> str:
    """The idempotency key: one per (job, checkpoint bytes, measurement)."""
    import hashlib

    payload = f"{job_id}\n{checkpoint_sha256}\n{eval_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def new_entry(
    *,
    key: str,
    job_id: str,
    task: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    eval_version: str,
    source_dir: Path,
) -> dict:
    """A unit with every stage recorded as not yet run."""
    stamp = now_iso()
    return {
        "key": key,
        "job_id": job_id,
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "eval_version": eval_version,
        "source_dir": str(source_dir),
        "created_at": stamp,
        "updated_at": stamp,
        "stages": {name: {"status": "pending", "reason": "", "at": ""} for name in STAGES},
        "attempts": {name: 0 for name in STAGES},
        "artifacts": {},
    }


def stage_ok(entry: dict, name: str) -> bool:
    return entry["stages"].get(name, {}).get("status") == "ok"


def stage_terminal(entry: dict, name: str) -> bool:
    return entry["stages"].get(name, {}).get("status") in TERMINAL_STATUSES


def needs_work(entry: dict, max_attempts: int) -> bool:
    """True while some stage is neither terminal nor out of attempts.

    `blocked` and `failed` count as work: a webhook may be configured later and
    a transient network error is worth another pass.
    """
    for name in STAGES:
        status = entry["stages"].get(name, {}).get("status")
        if status in TERMINAL_STATUSES:
            continue
        if status == "pending":
            return True
        if entry.get("attempts", {}).get(name, 0) < max_attempts:
            return True
    return False


class Store:
    """The state file, its lock, and atomic replacement.

    Every mutation holds the lock, and every write replaces the file in one
    `os.replace`, so a `watch` process and a manual `run` cannot leave a
    half-written state behind. Two processes still race on *doing* the work;
    the pipeline re-reads the unit under the lock before each stage for that.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self) -> dict:
        if not self.path.is_file():
            return {"version": 1, "entries": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError(f"{self.path} is not a postfetch state file")
        return data

    def write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def get(self, key: str) -> dict | None:
        return self.read()["entries"].get(key)

    def entries(self) -> dict[str, dict]:
        return self.read()["entries"]

    def put(self, entry: dict) -> None:
        with self.locked():
            data = self.read()
            entry["updated_at"] = now_iso()
            data["entries"][entry["key"]] = entry
            self.write(data)

    def set_stage(
        self, key: str, name: str, status: str, reason: str = "", detail: dict | None = None
    ) -> dict:
        """Record one stage's outcome and return the updated entry."""
        with self.locked():
            data = self.read()
            entry = data["entries"].get(key)
            if entry is None:
                raise KeyError(f"no unit {key} in {self.path}")
            entry["stages"][name] = {"status": status, "reason": reason, "at": now_iso()}
            if detail:
                entry["artifacts"][name] = detail
            if status in {"failed", "blocked"}:
                entry["attempts"][name] = entry["attempts"].get(name, 0) + 1
            entry["updated_at"] = now_iso()
            self.write(data)
            return entry

    def reset_unit(self, key: str) -> None:
        """Put every post-download stage back to pending; used by `run --force`."""
        with self.locked():
            data = self.read()
            entry = data["entries"].get(key)
            if entry is None:
                raise KeyError(f"no unit {key} in {self.path}")
            for name in STAGES:
                if name == "download":
                    continue
                entry["stages"][name] = {"status": "pending", "reason": "", "at": ""}
                entry["attempts"][name] = 0
            entry["artifacts"] = {name: value for name, value in entry.get("artifacts", {}).items()
                                  if name == "download"}
            entry["updated_at"] = now_iso()
            self.write(data)

    def reset_attempts(self, key: str, stage: str | None = None) -> None:
        """Give a unit another chance; used by the manual `retry` command."""
        with self.locked():
            data = self.read()
            entry = data["entries"].get(key)
            if entry is None:
                raise KeyError(f"no unit {key} in {self.path}")
            names = [stage] if stage else list(STAGES)
            for name in names:
                entry["attempts"][name] = 0
                if entry["stages"].get(name, {}).get("status") in {"failed", "blocked"}:
                    entry["stages"][name] = {"status": "pending", "reason": "", "at": ""}
            entry["updated_at"] = now_iso()
            self.write(data)
