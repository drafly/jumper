# After the download: evaluate, record, notify

`tools/remote_train.py fetch` copies a finished run back and verifies its
SHA-256. This page is the next step: the same command can hand the verified
directory to `tools/postfetch`, which evaluates the checkpoint with
`tools/replay_eval.py`, records a short simulation clip, and sends one Feishu
message.

```bash
python tools/remote_train.py fetch <job_id> --post
```

`--post` is best effort by design: the verified copy is `fetch`'s contract, and
a pipeline that cannot run (no Feishu target, no working renderer) is recorded
in its own state file rather than failing the download.

## Commands

| command | what it does |
|---|---|
| `python -m tools.postfetch run --job-dir <dir>` | process one fetched directory |
| `python -m tools.postfetch run --job <job_id>` | the same, found under `paths.results` |
| `python -m tools.postfetch watch` | keep watching: new downloads, unfinished units |
| `python -m tools.postfetch status` | every unit and the status of each stage |
| `python -m tools.postfetch retry <job_id> [--stage notify]` | reset attempts and run the failed stage again |
| `python -m tools.postfetch preview` | print notifications waiting for a target |

`watch` is the continuous form of `run`. Each pass looks for a fetched
directory it has not seen and for a unit with a stage that can be retried, so a
pass that could not run while the Mac slept is simply the next pass after it
wakes.

## Configuration

Copy `tools/postfetch.example.toml` to `~/.config/jumper/postfetch.toml` (or
pass `--config`). Paths default to `outputs/remote` and `outputs/postfetch`.
Nothing in that file is a secret: the Feishu credential comes from the existing
DSH integration or the environment variable named by `feishu.webhook_env`.

## One unit, and only once

A unit is keyed by `job_id + checkpoint SHA-256 + evaluation configuration
version`. Fetching the same job again, restarting `watch`, or running a second
`run` finds the unit and does nothing. Changing a threshold or the step count
is a new unit because the measurement changed; changing the chat, the video
size or the policy name is not.

The version is a hash of the evaluation block only (`Config.eval_version`).
This is what keeps "the model was re-measured" and "the message went somewhere
else" from looking the same.

## Stages

`download -> eval -> video -> notify`, each with its own status and reason in
`outputs/postfetch/state.json`:

| stage | ok | skipped | failed/blocked |
|---|---|---|---|
| download | the fetch verified the copy | — | the fetch did not verify |
| eval | the sweep ran | the task has no profile | load error, unsupported task, code mismatch |
| video | an MP4 was written | no evaluation to show, video disabled | renderer or encoder failed |
| notify | the message was delivered | notification disabled | no target (blocked, preview kept) or API error |

A failure stops only its own stage. A video failure never hides a completed
evaluation, and a notification failure never re-runs the simulation: `retry
--stage notify` resets that stage alone. Automatic attempts are capped by
`watch.max_attempts`; `postfetch retry` resets the counter.

## Evaluation

The task is read from the fetched `meta.json`, never assumed. A task is
measured only when `[profiles]` names it; anything else is reported as
unsupported and no walking metrics are attached to it. Before loading, the
checkpoint is compared with this checkout: the actor's hidden layers against the
task's agent config, and the recorded training commit against local `HEAD`
(`watch.allow_code_mismatch = true` measures anyway and records a warning).

The sweep is fixed: `stand`, `straight` (+0.3 m/s), `backward` (-0.3 m/s) and
`turn` (+0.5 rad/s), two seeds, two episodes of 250 steps. Reported per
scenario: tracking error, net displacement along the heading the episode
started with, falls, episodes and survival. Thresholds live under
`[eval.thresholds]`.

Two different answers are kept apart: **program success** (the sweep ran) and
**performance pass** (every configured threshold met). A unit that runs but
misses a threshold is not "failed"; it is reported with both facts and the raw
metrics are in `eval.json` beside the report.

## Video

On macOS the clip is rendered off screen with MuJoCo's CGL renderer (`EGL`
elsewhere), encoded at 25 fps where one frame is 0.04 s of simulated time, so a
slow renderer cannot make the policy look slow. Frames are annotated with the
checkpoint, the command and the simulated time; `video.json` records that the
encoding is by simulation time. Duration, resolution and file size are capped in
`[video]`; a failure is reported in the message, not raised.

`video.backend = "server"` is declared but not implemented: it would need
libEGL (or osmesa) and ffmpeg on the box plus a copy of the checkpoint there.
Choosing it produces that explanation and the numeric evaluation still
completes -- this tool does not change a server's environment.

## Feishu

The existing DSH Feishu bot is reused: the bot in
`~/.dsh/integrations/dsh-feishu/config.json`, its secret from the credentials
store, and the chat mapped to this workspace in `workspaces.json`. A different
target can be given by `[feishu] app_id`/`chat_id` or
`JUMPER_FEISHU_APP_ID`/`JUMPER_FEISHU_CHAT_ID`.

- App mode sends the text summary, then uploads the clip as a media message.
  A video the app is not permitted to upload is tried once and never blocks the
  already-sent summary; the result says so.
- Webhook mode (`JUMPER_FEISHU_WEBHOOK`) can send text but cannot upload files.
- With no target at all the message is written to
  `outputs/postfetch/pending/` and named as blocked; `postfetch preview` prints
  it. Nothing is published anywhere on its own.
- The app secret and webhook URL never appear in the state file, a reason or a
  preview; URLs in errors are masked.

## Verifying it

```bash
python -m pytest tests/test_postfetch.py tests/test_feishu_notify.py -q
python -m tools.postfetch status
cat outputs/postfetch/entries/<key>/report.md
```

Expected artifacts per unit: `eval.json`, `report.md`, `video/replay.mp4`,
`video/video.json`, `summary.md`, and (when no target is configured)
`outputs/postfetch/pending/<key>.json`.
