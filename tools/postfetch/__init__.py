"""Post-fetch pipeline: evaluate a downloaded checkpoint, record it, notify.

`tools/remote_train.py fetch` copies a finished run back and checks its SHA-256.
What happens *after* that -- loading the checkpoint in replay, measuring whether
it tracks its commands, recording a short simulation video, and telling Feishu --
is this package's job.

The three pieces are deliberately separable and separately recorded, because
they fail differently and at different rates: a checkpoint can evaluate fine and
fail to render, and a rendered video can fail to reach a chat. The state file
keeps a status and the reason for each, so a retry re-runs only what failed.

    python tools/postfetch run --job-dir outputs/remote/<job_id>   # one run
    python tools/postfetch watch                                  # keep watching
    python tools/postfetch status                                 # what happened
    python tools/postfetch retry <job_id> --stage notify          # only the notify

The same checkpoint is processed once. A processed unit is keyed by the job id,
the checkpoint's own SHA-256 and the evaluation configuration version, so a
restarted process, a rerun of `watch`, or a second `fetch` of the same job does
not evaluate it again or send a second message.
"""

from __future__ import annotations

#: Bumped only for the package's own contract, not for configuration. The
#: configuration version that gates idempotency is derived from the eval block
#: by `config.Config.eval_version`.
__version__ = "0.1.0"

__all__ = ["__version__"]
