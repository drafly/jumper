"""The fixed walking sweep, and the difference between "ran" and "passed".

`tools/replay_eval.py` is the measurement; this module only decides what the
measurement is *of*: four pinned commands, the configured seeds and episode
length, and the thresholds that separate a program that ran from a policy that
performs. Both are reported, because a checkpoint whose evaluation completed is
a successful **program** even when every threshold is missed, and pretending
otherwise is how a broken policy gets a green check.

The original numbers are kept in `eval.json` next to the aggregate and the
configuration that produced them: a threshold comparison without the raw
metrics beside it cannot be re-read later when the threshold changes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from . import artifacts
from .config import Config

REPO = Path(__file__).resolve().parents[2]

#: The command each scenario names, so the report can print what the policy was
#: actually asked to do even if a tool's vocabulary changes later.
COMMANDS: dict[str, tuple[float, float, float]] = {
    "stand": (0.0, 0.0, 0.0),
    "straight": (0.3, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.5),
}


def _replay_eval():
    """Load `tools/replay_eval.py` by path, independent of the caller's cwd."""
    path = REPO / "tools" / "replay_eval.py"
    name = "tools.postfetch._replay_eval"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_sweep(
    task: str,
    checkpoint: Path,
    cfg: Config,
    scenarios: tuple[str, ...] | None = None,
) -> dict:
    """Run replay_eval for every configured seed and return the raw reports."""
    replay = _replay_eval()
    reports = {}
    for seed in cfg.seeds:
        namespace = SimpleNamespace(
            task=task,
            checkpoint=str(checkpoint),
            episodes=cfg.episodes,
            steps=cfg.steps,
            seed=seed,
            scenarios=",".join(scenarios or cfg.scenarios),
            out=None,
        )
        reports[seed] = replay.run(namespace)
    return reports


def aggregate(reports: dict[int, dict], cfg: Config) -> dict:
    """Pool each scenario's episodes across seeds into one measurable row."""
    pooled: dict[str, list[dict]] = {name: [] for name in cfg.scenarios}
    for report in reports.values():
        for name in cfg.scenarios:
            row = report["scenarios"].get(name)
            if row:
                pooled[name].extend(row["detail"])
    rows: dict[str, dict] = {}
    for name, episodes in pooled.items():
        if not episodes:
            rows[name] = {"episodes": 0, "missing": True}
            continue
        count = len(episodes)
        disp = [
            (float(episode["net_disp_heading_xy"][0]), float(episode["net_disp_heading_xy"][1]))
            for episode in episodes
        ]
        rows[name] = {
            "command": dict(zip(("vx", "vy", "wz"), COMMANDS[name], strict=True)),
            "episodes": count,
            "falls": sum(1 for episode in episodes if episode["fell"]),
            "fall_rate": sum(1 for episode in episodes if episode["fell"]) / count,
            "mean_survival_steps": sum(float(episode["steps"]) for episode in episodes) / count,
            "mean_lin_error": sum(float(episode["lin_error"]) for episode in episodes) / count,
            "mean_yaw_error": sum(float(episode["yaw_error"]) for episode in episodes) / count,
            "mean_speed": sum(float(episode["speed"]) for episode in episodes) / count,
            "mean_vx": sum(float(episode["mean_vx"]) for episode in episodes) / count,
            "mean_wz": sum(float(episode["mean_wz"]) for episode in episodes) / count,
            "mean_net_disp_heading_xy": [
                sum(item[0] for item in disp) / count,
                sum(item[1] for item in disp) / count,
            ],
            "min_survival_steps": min(float(episode["steps"]) for episode in episodes),
        }
    return rows


def _check(name: str, value: float, threshold: float, lower_is_better: bool) -> dict:
    passed = value <= threshold if lower_is_better else value >= threshold
    return {
        "name": name,
        "value": round(value, 4),
        "threshold": threshold,
        "rule": "max" if lower_is_better else "min",
        "passed": bool(passed),
    }


def judge(rows: dict, cfg: Config) -> dict:
    """Apply the configured thresholds, scenario by scenario.

    Every scenario also has to be survived for the whole episode and without a
    fall, because "tracks the command" while lying on its side is not a pass.
    """
    survival_ratio_floor = cfg.thresholds["min_survival_ratio"]
    fall_ceiling = cfg.thresholds["fall_rate_max"]
    judged: dict[str, dict] = {}
    for name, row in rows.items():
        if row.get("missing"):
            judged[name] = {"pass": False, "checks": [], "missing": True}
            continue
        checks = [
            _check("fall_rate", float(row["fall_rate"]), fall_ceiling, True),
            _check(
                "survival_ratio",
                float(row["mean_survival_steps"]) / cfg.steps,
                survival_ratio_floor,
                False,
            ),
        ]
        if name == "stand":
            checks.append(_check("lin_error", row["mean_lin_error"], cfg.thresholds["stand_lin_err"], True))
        elif name == "straight":
            checks.append(
                _check("lin_error", row["mean_lin_error"], cfg.thresholds["straight_lin_err"], True)
            )
            checks.append(
                _check(
                    "net_disp_heading_x",
                    row["mean_net_disp_heading_xy"][0],
                    cfg.thresholds["straight_disp_min"],
                    False,
                )
            )
        elif name == "backward":
            checks.append(
                _check("lin_error", row["mean_lin_error"], cfg.thresholds["backward_lin_err"], True)
            )
            checks.append(
                _check(
                    "net_disp_heading_x",
                    row["mean_net_disp_heading_xy"][0],
                    cfg.thresholds["backward_disp_max"],
                    False,
                )
            )
        elif name == "turn":
            checks.append(_check("yaw_error", row["mean_yaw_error"], cfg.thresholds["turn_yaw_err"], True))
        judged[name] = {"pass": all(check["passed"] for check in checks), "checks": checks}
    return judged


def _report_markdown(job: artifacts.FetchedJob, result: dict) -> str:
    verdict = "达标" if result["performance_pass"] else "未达标"
    lines = [
        f"# 评估报告：{result['task']} / {result['job_id']}",
        "",
        f"- checkpoint：`{job.checkpoint.name}`",
        f"- checkpoint SHA-256：`{job.checkpoint_sha256}`",
        f"- 代码版本：训练于 `{job.git_head[:12] or '未知'}`；本地 `{result['local_head'][:12] or '未知'}`",
        (
            f"- 评估配置版本：`{result['eval_version']}`（steps={result['config']['steps']}，"
            f"episodes={result['config']['episodes']}，seeds={result['config']['seeds']}）"
        ),
        f"- 程序执行：成功；策略性能：**{verdict}**",
        f"- 生成时间：{result['generated_at']}",
        "",
        (
            "| 场景 | 命令 (vx, vy, wz) | 回合 | 跌倒 | 存活步 | 速度误差 (m/s) | "
            "角速度误差 (rad/s) | 净位移(目标方向) | 通过 |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in result["config"]["scenarios"]:
        row = result["scenarios"].get(name, {})
        judged = result["judgement"].get(name, {})
        if row.get("missing"):
            lines.append(f"| {name} | - | - | - | - | - | - | 缺失 |")
            continue
        command = row["command"]
        lines.append(
            f"| {name} | ({command['vx']:g}, {command['vy']:g}, {command['wz']:g}) | "
            f"{row['episodes']} | {row['falls']} | {row['mean_survival_steps']:.0f} | "
            f"{row['mean_lin_error']:.3f} | {row['mean_yaw_error']:.3f} | "
            f"{row['mean_net_disp_heading_xy'][0]:+.3f} m | "
            f"{'是' if judged.get('pass') else '否'} |"
        )
    lines.extend(["", "## 阈值与判定", "", "| 场景 | 检查 | 实测 | 阈值 | 规则 | 结果 |", "|---|---|---|---|---|---|"])
    for name in result["config"]["scenarios"]:
        for check in result["judgement"].get(name, {}).get("checks", []):
            rule = "≤" if check["rule"] == "max" else "≥"
            lines.append(
                f"| {name} | {check['name']} | {check['value']} | {rule} {check['threshold']} | "
                f"{'通过' if check['passed'] else '未通过'} |"
            )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 原始指标、实际命令与评估配置都保存在同目录的 `eval.json` 中。",
            (
                "- 上表的速度误差是 `replay_eval` 的 `mean_lin_error`（|vx-cmd_x|+|vy-cmd_y| 的逐步均值，"
                "单位为 m/s）；净位移是每回合世界位移在**回合初始朝向**下的投影，前进为 +x。"
            ),
            "- 跌倒阈值：基座高度 < 0.06 m 或机身 up 轴 z < 0.4，或环境自身终止。",
            "- “程序执行成功”只表示评估跑完了；“策略性能达标”是上表阈值全部通过。",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(job: artifacts.FetchedJob, cfg: Config, entry_dir: Path) -> dict:
    """Measure, judge, write `eval.json` and `report.md`, return the result.

    A supported and compatible job is assumed; the caller has already recorded
    the unsupported cases. An exception here is a **program** failure and is
    raised for the pipeline to record with its reason.
    """
    reports = run_sweep(job.task, job.checkpoint, cfg)
    rows = aggregate(reports, cfg)
    judgement = judge(rows, cfg)
    result = {
        "task": job.task,
        "job_id": job.job_id,
        "checkpoint": str(job.checkpoint),
        "checkpoint_name": job.checkpoint.name,
        "checkpoint_sha256": job.checkpoint_sha256,
        "trained_from": job.git_head,
        "local_head": artifacts.local_git_head(),
        "code_warnings": list(job.warnings),
        "eval_version": cfg.eval_version(job.task),
        "profile": cfg.profile_for(job.task),
        "config": {
            "scenarios": list(cfg.scenarios),
            "steps": cfg.steps,
            "episodes": cfg.episodes,
            "seeds": list(cfg.seeds),
            "thresholds": dict(sorted(cfg.thresholds.items())),
            "commands": {name: COMMANDS[name] for name in cfg.scenarios},
        },
        "scenarios": rows,
        "judgement": judgement,
        "program_ok": True,
        "performance_pass": bool(judgement) and all(row.get("pass") for row in judgement.values()),
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    }
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")
    (entry_dir / "report.md").write_text(_report_markdown(job, result), "utf-8")
    return result
