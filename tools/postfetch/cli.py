"""Command line for the post-fetch pipeline.

    python -m tools.postfetch run --job-dir outputs/remote/<job>
    python -m tools.postfetch watch
    python -m tools.postfetch status
    python -m tools.postfetch retry <job_id> --stage notify
    python -m tools.postfetch preview

`remote_train.py fetch --post` calls `run` for the job it just verified, which
is the single execution path; `watch` is the continuous one.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

from . import config, pipeline, state


def _common() -> argparse.ArgumentParser:
    """Options accepted before or after the subcommand.

    `--webhook` exists to exercise the failure path by hand, and typing it after
    `run` is the natural order; a parent parser attached to the subparsers as
    well as the main parser accepts both without duplicating the definitions.
    """
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so the subparser does not overwrite what the main parser (or an
    # earlier position of the same option) already parsed: argparse applies a
    # subparser's defaults over the parent namespace, which silently discarded
    # `--config` typed before the subcommand.
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help="TOML config path (default: ~/.config/jumper/postfetch.toml)")
    common.add_argument(
        "--webhook", metavar="URL", default=argparse.SUPPRESS,
        help="override the Feishu webhook for this invocation only; intended for "
             "exercising the failure path, never stored",
    )
    common.add_argument(
        "--notify", choices=["on", "off"], default=argparse.SUPPRESS,
        help="turn the notification stage off for this invocation (default: on)",
    )
    return common


def _parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="postfetch",
        description="Evaluate a fetched checkpoint, record it, and notify Feishu.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process one job, or every job with work",
                         parents=[common])
    run.add_argument("--job-dir", help="a fetched run directory (holds FETCH.json)")
    run.add_argument("--job", help="a job id under the configured results directory")
    run.add_argument("--force", action="store_true",
                     help="reset THIS unit's attempts and run its stages again")

    watch = sub.add_parser("watch", help="keep watching for downloads and retries",
                           parents=[common])
    watch.add_argument("--interval", type=float, default=None,
                       help="seconds between passes (default: watch.interval_s)")
    watch.add_argument("--once", action="store_true", help="one pass, then exit")

    sub.add_parser("status", help="print each unit's stage statuses", parents=[common])
    sub.add_parser("preview", help="print notifications waiting for a target",
                   parents=[common])

    retry = sub.add_parser("retry", help="reset attempts so a stage runs again",
                           parents=[common])
    retry.add_argument("job", help="job id or the unit's 16-hex key")
    retry.add_argument("--stage", choices=list(state.STAGES),
                       help="only this stage (default: every failed/blocked stage)")
    return parser


def _store(cfg: config.Config) -> state.Store:
    return state.Store(cfg.state_path)


def _apply_overrides(args: argparse.Namespace, cfg: config.Config) -> config.Config:
    webhook = getattr(args, "webhook", None)
    if webhook:
        os.environ[cfg.feishu.webhook_env] = webhook
    if getattr(args, "notify", "on") == "off":
        cfg = dataclasses.replace(cfg, feishu=dataclasses.replace(cfg.feishu, enabled=False))
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = _apply_overrides(args, config.load(getattr(args, "config", None)))
    store = _store(cfg)
    try:
        if args.command == "run":
            if not args.job_dir and not args.job:
                raise SystemExit("run needs --job-dir or --job")
            entries = pipeline.run_once(cfg, store, job_dir=args.job_dir, job_id=args.job,
                                        force=args.force)
            if not entries:
                print("nothing to do")
            return 0
        if args.command == "watch":
            pipeline.watch(cfg, store, interval_s=args.interval, once=args.once)
            return 0
        if args.command == "status":
            pipeline.status(cfg, store)
            return 0
        if args.command == "preview":
            pipeline.show_preview(cfg)
            return 0
        if args.command == "retry":
            entries = store.entries()
            match = entries.get(args.job) or next(
                (entry for entry in entries.values() if entry["job_id"] == args.job), None
            )
            if match is None:
                raise SystemExit(f"no unit for {args.job!r} in {cfg.state_path}")
            store.reset_attempts(match["key"], args.stage)
            source = match.get("source_dir", "")
            if source:
                pipeline.run_once(cfg, store, job_dir=source)
            else:
                print(f"reset {match['key']}; no source directory to resume")
            return 0
    except KeyboardInterrupt:
        print("\nstopped")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
