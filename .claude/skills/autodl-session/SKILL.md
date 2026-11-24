---
name: autodl-session
description: Run a training job on the AutoDL box without leaving it powered on -- start the instance on demand, wait for the job, fetch and verify every result, then shut it down. Use this whenever a training run is requested on AutoDL, when the instance may currently be off, when someone asks "shut the box down after the run", "don't leave the GPU running", "what is still unsynced", or after a run finishes so the shutdown gate and the post-fetch pipeline can complete. The command is `python tools/autodl_lifecycle.py`; its whole point is that shutdown is refused until every terminal run is safely on this Mac.
---

# A training session that ends with the box off

`tools/remote_train.py` assumes a running box. This skill is the wrapper that makes
the box exist only for the run, and makes "done" mean **the evidence is here**, not
"the tmux session stopped".

## The one rule

**Never shut the box down because training finished.** Shut it down because every
run on it is fetched, SHA-256 verified, and its final checkpoint hashes to the box's
copy. That is what `down` checks; `--force` is the only way past it and the audit log
records that it was forced.

## Procedure

1. **Decide the start path** from `~/.config/jumper/autodl.toml` (template:
   `tools/autodl.example.toml`):

   | `autodl.api` | meaning |
   |---|---|
   | `manual` | default; you start the instance in the console, `up` waits for SSH |
   | `pro` | official API, needs a container instance Pro (`pro-...`) and `AUTODL_TOKEN` |
   | `console` | web-console JWT; standard instance, but the token expires |

   Standard container instances have **no official start API**; their shutdown,
   however, is AutoDL's documented in-container `/usr/bin/shutdown` and needs no
   credential. So a standard instance can always be stopped automatically even when
   it must be started by hand.

2. **Run** one of:

   ```bash
   python tools/autodl_lifecycle.py session --task jumper.flat          # the whole day
   python tools/autodl_lifecycle.py status                              # what is up
   python tools/autodl_lifecycle.py verify                              # gate only
   python tools/autodl_lifecycle.py down                                # gate, then off
   ```

   `session` syncs a fresh release, submits, polls the job's terminal status,
   fetches, runs the post-fetch pipeline when `--post` is given, checks the gate,
   and only then shuts down. Add `--dry-run` to see the plan without touching the
   box, `--fetch-missing` to make `down` fetch what is missing first, and
   `--all-checkpoints` (config `fetch_all`) when the whole trajectory matters.

3. **On failure, leave it running.** If a step before the gate fails, `session`
   stops with the box still on and says so. Do not "clean up" with
   `down --force` unless you have decided the unsynced run does not matter.

## What to check when it refuses

- `refusing to shut down, the Mac does not hold a verified copy` names the job ids
  and the reason: either `remote_train.py fetch <id>` was never run, the fetch
  reported `missing`, or the local checkpoint no longer hashes to the box's. Fetch,
  then `verify` again.
- `job(s) still running` means `status` still sees a live or queued job. Stop it
  with `remote_train.py stop <id>`; that is a training decision, not a cleanup.
- The audit trail is `outputs/autodl/lifecycle.jsonl` -- no token is ever in it.

## Reference

- [`docs/AUTODL_LIFECYCLE.md`](../../../docs/AUTODL_LIFECYCLE.md): the platform facts,
  the three start paths and their credential lifetimes, and the gate's definition.
- [`docs/REMOTE_TRAIN.md`](../../../docs/REMOTE_TRAIN.md): the job driver this wraps.
