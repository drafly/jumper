"""Run the post-fetch stages as one unit, once, and keep it that way.

The order is `download -> eval -> video -> notify`, and each stage's status is
written before the next begins. That is what makes a killed process resumable
and a failed notification retryable without re-running the simulation: the
notification is a *stage*, not a step inside the evaluation.

Nothing here sends a message before the numeric result exists, and nothing
sends the same unit twice. A unit that has nothing left to do is detected by
`state.needs_work`, which is also what a restarted `watch` loop uses to pick up
work a sleeping Mac interrupted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import artifacts, evaluate, state, video
from .config import REPO, Config


#: Reasons get one line in the state file and in the message; a traceback would
#: be unreadable in a chat and is not a reason a human can act on.
def _short(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _entry_dir(cfg: Config, key: str) -> Path:
    return cfg.out_dir / "entries" / key


def _stage(entry: dict, name: str) -> dict:
    return entry["stages"].get(name, {"status": "pending", "reason": ""})


def _should_run(entry: dict, name: str, max_attempts: int) -> bool:
    stage = _stage(entry, name)
    status = stage.get("status", "pending")
    if status in {"ok", "skipped", "unsupported"}:
        return False
    if status == "pending":
        return True
    return entry.get("attempts", {}).get(name, 0) < max_attempts


def _verdict_text(ev: dict | None, video_result: dict | None) -> str:
    if ev is None:
        return "程序失败"
    if not ev.get("program_ok"):
        return "程序失败"
    return "达标" if ev.get("performance_pass") else "未达标"


def message_text(job: artifacts.FetchedJob, ev: dict | None, video_result: dict | None,
                 reason: str = "") -> str:
    """The chat summary: what was measured, whether it passed, and what is missing."""
    lines: list[str] = []
    lines.append(f"【训练结果】{job.task} / {job.job_id}")
    lines.append(f"checkpoint: {job.checkpoint.name} (sha256 {job.checkpoint_sha256[:12]})")
    if not job.supported:
        lines.append(f"评估: 不支持（{job.reason}）")
        lines.append("未套用行走指标；不给出性能结论。")
    elif ev is None:
        lines.append(f"评估: 程序失败（{reason or '见状态文件'}）")
    else:
        lines.append(f"评估: {_verdict_text(ev, video_result)}（程序成功）")
        for name in ev["config"]["scenarios"]:
            row = ev["scenarios"].get(name) or {}
            if row.get("missing"):
                lines.append(f"- {name}: 缺失")
                continue
            verdict = ev["judgement"].get(name, {})
            if name == "turn":
                metric = f"角速度误差 {row['mean_yaw_error']:.3f} rad/s"
            else:
                metric = f"速度误差 {row['mean_lin_error']:.3f} m/s"
            disp = row["mean_net_disp_heading_xy"][0]
            lines.append(
                f"- {name}: {metric}, 净位移 {disp:+.3f} m"
                f"{'' if verdict.get('pass') else '  [未通过]'}"
            )
        failed = [
            f"{name}.{check['name']}={check['value']}"
            for name, verdict in ev["judgement"].items()
            for check in verdict.get("checks", [])
            if not check["passed"]
        ]
        if failed:
            lines.append("未通过项: " + ", ".join(failed))
    if video_result is None:
        lines.append("录像: 未执行")
    elif video_result.get("status") == "ok":
        lines.append(
            f"录像: 已生成 {video_result['seconds']:.1f} s / "
            f"{video_result['bytes'] / 1e6:.1f} MB（仿真时间编码, {video_result['fps']} fps）"
        )
    else:
        lines.append(f"录像: 未生成（{video_result.get('reason', video_result.get('status'))}）")
    if reason and job.supported:
        lines.append(f"原因: {reason}")
    return "\n".join(lines)


def _write_report(entry_dir: Path, job: artifacts.FetchedJob, ev: dict | None,
                  video_result: dict | None, reason: str) -> None:
    """A one-file summary beside the machine-readable JSON."""
    lines = [f"# {job.task} / {job.job_id}", ""]
    lines.append(f"- checkpoint: `{job.checkpoint.name}` (`{job.checkpoint_sha256}`)")
    if not job.supported:
        lines.append(f"- 不支持: {job.reason}")
    elif ev is None:
        lines.append(f"- 程序失败: {reason}")
    else:
        lines.append(f"- 性能结论: {'达标' if ev['performance_pass'] else '未达标'}")
        lines.append(f"- 评估配置版本: `{ev['eval_version']}`")
        lines.append("- 详见同目录 `eval.json` 与 `report.md`")
    lines.append("")
    lines.append("## 录像")
    lines.append("")
    if video_result is None:
        lines.append("- 未执行")
    elif video_result.get("status") == "ok":
        lines.append(
            f"- `{Path(video_result['file']).name}`：{video_result['seconds']:.1f} s，"
            f"{video_result['bytes'] / 1e6:.1f} MB，{video_result['fps']} fps，"
            f"按仿真时间编码（{video_result['sampling_dt_s']:.2f} s/帧）"
        )
    else:
        lines.append(f"- 未生成：{video_result.get('reason', '')}")
    lines.append("")
    (entry_dir / "summary.md").write_text("\n".join(lines) + "\n", "utf-8")


def _notify(cfg: Config, job: artifacts.FetchedJob, ev: dict | None,
            video_result: dict | None, reason: str, entry_dir: Path) -> dict:
    """Send, or leave a preview when there is nowhere to send.

    A blocked notification is not a failure of the run: it is the documented
    outcome when no chat is configured, and the message is kept so a later
    configuration change can send it without re-evaluating anything.
    """
    from . import feishu

    text = message_text(job, ev, video_result, reason)
    if not cfg.feishu.enabled:
        return {"status": "skipped", "reason": "feishu disabled by configuration"}
    webhook = ""
    if cfg.feishu.webhook_env:
        import os

        webhook = os.environ.get(cfg.feishu.webhook_env, "").strip()
    overrides = {
        "app_id": cfg.feishu.app_id,
        "chat_id": cfg.feishu.chat_id,
        "webhook_url": webhook,
    }
    chat = feishu.discover(REPO, overrides=overrides)
    video_path = None
    if video_result and video_result.get("status") == "ok":
        video_path = Path(video_result["file"])
    result = feishu.send(chat, text, video=video_path)
    if result.get("status") == "blocked":
        pending = cfg.out_dir / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        path = feishu.preview(chat, text, pending)
        return {
            "status": "blocked",
            "reason": result.get("reason", "no notification target configured"),
            "preview": str(path),
        }
    if result.get("status") != "ok":
        return {"status": "failed", "reason": result.get("reason", "send failed")}
    detail = {"mode": result.get("mode", ""), "video": result.get("video", "")}
    if result.get("reason"):
        detail["note"] = result["reason"]
    return {"status": "ok", "reason": result.get("reason", ""), "detail": detail}


def process_job(job_dir: str | Path, cfg: Config, store: state.Store,
                force: bool = False) -> dict:
    """Advance one fetched job as far as it can go; safe to call repeatedly."""
    job = artifacts.read_job(job_dir, cfg)
    key = state.unit_key(job.job_id, job.checkpoint_sha256, cfg.eval_version(job.task))
    entry = store.get(key)
    if entry is None:
        entry = state.new_entry(
            key=key, job_id=job.job_id, task=job.task, checkpoint=job.checkpoint,
            checkpoint_sha256=job.checkpoint_sha256, eval_version=cfg.eval_version(job.task),
            source_dir=job.source_dir,
        )
        store.put(entry)
    if force:
        store.reset_unit(key)

    entry = store.get(key) or entry
    entry_dir = _entry_dir(cfg, key)
    entry_dir.mkdir(parents=True, exist_ok=True)

    # The fetch this job came from already verified the SHA-256; recording it is
    # what lets a resumed unit say the download stage had passed before it died.
    if _stage(entry, "download")["status"] == "pending":
        detail = {
            "verified_files": int(job.fetch.get("verified", 0)),
            "mode": job.fetch.get("mode", ""),
            "source": str(job.fetch.get("local", job.source_dir)),
        }
        store.set_stage(key, "download", "ok", detail=detail)
        entry = store.get(key) or entry

    # ── evaluation ──
    ev: dict | None = None
    if _should_run(entry, "eval", cfg.max_attempts):
        if not job.supported:
            store.set_stage(key, "eval", "unsupported", job.reason)
            entry = store.get(key) or entry
        elif not job.compatible:
            store.set_stage(key, "eval", "failed", job.compatibility_reason)
            entry = store.get(key) or entry
        else:
            try:
                ev = evaluate.evaluate(job, cfg, entry_dir)
            except Exception as exc:  # noqa: BLE001 - recorded, retried by watch
                store.set_stage(key, "eval", "failed", _short(exc))
                entry = store.get(key) or entry
            else:
                store.set_stage(
                    key, "eval", "ok",
                    detail={"performance_pass": ev["performance_pass"],
                            "eval_json": str(entry_dir / "eval.json"),
                            "report": str(entry_dir / "report.md")},
                )
                entry = store.get(key) or entry
    else:
        path = entry.get("artifacts", {}).get("eval", {}).get("eval_json")
        if path and Path(path).is_file():
            ev = json.loads(Path(path).read_text("utf-8"))

    # ── video ──
    video_result: dict | None = entry.get("artifacts", {}).get("video")
    if _should_run(entry, "video", cfg.max_attempts):
        if ev is None or not ev.get("program_ok"):
            store.set_stage(key, "video", "skipped", "no successful evaluation")
            entry = store.get(key) or entry
        else:
            video_result = video.render(job, cfg, entry_dir)
            status = "ok" if video_result.get("status") == "ok" else "failed"
            store.set_stage(key, "video", status, video_result.get("reason", ""),
                            detail=video_result)
            entry = store.get(key) or entry

    # ── notification ──
    if _should_run(entry, "notify", cfg.max_attempts):
        reason = ""
        eval_stage = _stage(entry, "eval")
        if eval_stage["status"] in {"failed", "unsupported"}:
            reason = eval_stage.get("reason", "")
        result = _notify(cfg, job, ev, video_result, reason, entry_dir)
        status = result["status"]
        store.set_stage(key, "notify", status, result.get("reason", ""),
                        detail={"preview": result.get("preview", ""),
                                **result.get("detail", {})})
        entry = store.get(key) or entry

    entry = store.get(key) or entry
    _write_report(entry_dir, job, ev, video_result, _stage(entry, "eval").get("reason", ""))
    return entry


def _find_fetched(cfg: Config, job_id: str | None = None) -> list[Path]:
    root = cfg.results_dir
    if not root.is_dir():
        return []
    if job_id:
        target = root / job_id
        return [target] if (target / "FETCH.json").is_file() else []
    return sorted(path.parent for path in root.glob("*/FETCH.json"))


def run_once(cfg: Config, store: state.Store, job_dir: str | Path | None = None,
             job_id: str | None = None, force: bool = False) -> list[dict]:
    """Process one job, or every fetched job that still has work."""
    targets = [Path(job_dir)] if job_dir else _find_fetched(cfg, job_id)
    done: list[dict] = []
    for target in targets:
        try:
            entry = process_job(target, cfg, store, force=force)
        except artifacts.ArtifactError as exc:
            print(f"skip {target.name}: {exc}")
            continue
        done.append(entry)
        print(_line(entry))
    return done


def _line(entry: dict) -> str:
    stages = " ".join(
        f"{name}={entry['stages'][name]['status']}" for name in state.STAGES
    )
    return f"{entry['job_id']}  {entry['task']}  {stages}"


def resume(cfg: Config, store: state.Store) -> list[dict]:
    """Finish every unit the last process left unfinished.

    This is the catch-up path after a crash or a Mac sleep: the state file is
    the queue, and a unit whose source directory is gone is reported and left
    alone rather than silently forgotten.
    """
    done: list[dict] = []
    for entry in list(store.entries().values()):
        if not state.needs_work(entry, cfg.max_attempts):
            continue
        source = Path(entry.get("source_dir", ""))
        if not (source / "FETCH.json").is_file():
            print(f"cannot resume {entry['job_id']}: {source} has no FETCH.json")
            continue
        try:
            done.append(process_job(source, cfg, store))
        except artifacts.ArtifactError as exc:
            print(f"cannot resume {entry['job_id']}: {exc}")
    return done


def watch(cfg: Config, store: state.Store, interval_s: float | None = None,
          once: bool = False) -> None:
    """Poll for new verified downloads and unfinished work.

    Each pass is the same two questions -- is there a fetched job not yet seen,
    and is there a unit with a retryable stage -- so a pass that could not run
    while the Mac slept is simply the next pass after it wakes.
    """
    delay = cfg.watch_interval_s if interval_s is None else interval_s
    while True:
        for entry in resume(cfg, store):
            print(_line(entry))
        for target in _find_fetched(cfg):
            try:
                job = artifacts.read_job(target, cfg)
            except artifacts.ArtifactError as exc:
                print(f"skip {target.name}: {exc}")
                continue
            key = state.unit_key(job.job_id, job.checkpoint_sha256, cfg.eval_version(job.task))
            if store.get(key) is not None:
                continue
            entry = process_job(target, cfg, store)
            print(_line(entry))
        if once:
            return
        time.sleep(delay)


def status(cfg: Config, store: state.Store) -> None:
    entries = store.entries()
    if not entries:
        print(f"no units in {store.path}")
        return
    print(f"{'job_id':<26}{'task':<16}{'eval':<12}{'video':<10}{'notify':<10}key")
    for entry in sorted(entries.values(), key=lambda item: item.get("updated_at", "")):
        stages = entry["stages"]
        print(
            f"{entry['job_id']:<26}{entry['task']:<16}"
            f"{stages['eval']['status']:<12}{stages['video']['status']:<10}"
            f"{stages['notify']['status']:<10}{entry['key']}"
        )
        for name in state.STAGES:
            reason = stages[name].get("reason", "")
            if reason:
                print(f"    {name}: {reason}")


def show_preview(cfg: Config) -> None:
    pending = cfg.out_dir / "pending"
    files = sorted(pending.glob("*.json")) if pending.is_dir() else []
    if not files:
        print(f"no pending notifications under {pending}")
        return
    for path in files:
        data = json.loads(path.read_text("utf-8"))
        print(f"--- {path}")
        print(data.get("text", ""))
