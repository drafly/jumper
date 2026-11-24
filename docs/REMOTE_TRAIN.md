# Remote training over ssh

[Home](../README.md) · [中文](REMOTE_TRAIN.zh.md)

A laptop with no GPU can still train: this page is the driver that copies a
checkout to a GPU box, runs the repository's own `scripts/train.py` there, and
brings the checkpoint and its evidence back.

```bash
python tools/remote_train.py check
python tools/remote_train.py sync
python tools/remote_train.py submit --task jumper.flat --release <id> --num-envs 64 --max-iterations 20
python tools/remote_train.py status <job_id>
python tools/remote_train.py fetch <job_id>
```

The driver is `tools/remote_train.py`. It is **not** a second training entry
point: it builds an argument list for `scripts/train.py` and never imports a task,
a simulator or torch. Anything it learns about a run, it learns from the files the
run leaves behind.

## What it is responsible for, and what it is not

| It does | It does not |
|---|---|
| Copy a release and prove every byte arrived | Sync automatically before every job (a job runs one named release) |
| Start `train.py` in tmux and survive ssh disconnects | Survive a reboot of the box; tmux cannot do that, checkpoints can |
| Report `succeeded` from exit code **and** a checkpoint on disk | Treat a vanished tmux session as success |
| Fetch results and verify SHA-256 | Delete results on the box or release the instance |
| Refuse when another job already owns the GPU | Kill a process it did not start |
| Stop a job with SIGINT → TERM → KILL | Promise that stopping saves a checkpoint (see [Limits](#limits)) |

## Requirements

Local (the Mac):

- an ssh key accepted by the box — the driver runs `ssh` with `BatchMode=yes`, so a
  password prompt fails instead of hanging;
- this repository's `.venv` (any Python 3.10–3.13 with the `dev` extra is enough to
  *run* the driver: it uses only the standard library);
- `tar` (bsdtar on macOS) for the upload and `rsync` for the download.

On the box:

- `tmux`, `rsync`, a Python 3.10–3.13 and an NVIDIA driver that `nvidia-smi` reports;
- a writable work directory. On AutoDL use the data disk, `/root/autodl-tmp/...`,
  not `/root` — the system disk is small and a `.venv` is several GB.

On AutoDL specifically: `python3` is often **not on the non-interactive PATH** even
though the image has one at `/root/miniconda3/bin/python`. Set `python` in the
config to that path. Install `tmux` once with `apt-get install -y tmux`. The
image's torch (2.8+cu128 in the instance this was tested on) is reused through
`--system-site-packages`, which avoids a multi-GB download; `check` verifies that
torch sees CUDA before anything is submitted.

## Configure

```bash
cp tools/remote_train.example.toml .remote.toml   # git-ignored
```

```toml
host = "autodl"                    # ssh alias, or:
# hostname = "connect.xxx.seetacloud.com"
# port = 24227
# user = "root"
# identity = "~/.ssh/id_ed25519"

workdir = "/root/autodl-tmp/jumper"
python = "/root/miniconda3/bin/python"
results = "~/jumper-runs"
max_hours = 6.0
# AutoDL: the default PyPI is slow from China. `network_turbo` proxies GitHub and
# makes pip *slower*, so a mirror is configured explicitly instead.
pip_index = "https://pypi.tuna.tsinghua.edu.cn/simple"

backend = "warp"
device = "cuda:0"
num_envs = 64
tensorboard_port = 6006
# mjlab sets MUJOCO_GL=egl by default. A box without libEGL (this RTX 3080 Ti
# instance) fails at import, headless or not; "disabled" trains without rendering.
mujoco_gl = "egl"
```

The search order is `--config`, `$JUMPER_REMOTE_CONFIG`, `./.remote.toml`, then
`~/.config/jumper/remote.toml`. **Nothing in this file is a secret**: `identity` is
a path. Do not put a password or an API token in it, and do not put one in `.env`
— `.env` is committed in this repository.

If you have a password rather than a key, install the key once yourself and never
give the password to the tool:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub -p 24227 root@connect.xxx.seetacloud.com
```

## Commands

| Command | What it answers |
|---|---|
| `check` | Can I reach the box, is the disk big enough, is the GPU idle, does torch see CUDA, do `mjrl`/`mjlab`/`tasks` import? Exit code 1 means not ready, 2 means unreachable. |
| `sync [--rev REF] [--label ID]` | Copy this checkout (or a git revision) into `releases/<id>` and verify the copy file by file. |
| `bootstrap --release ID` | Create `.venv` on the box and `pip install -e` the release. |
| `submit --task T --release ID` | Start a job in tmux; prints the `job_id`. |
| `status [JOB] [--json]` | State, iteration, newest checkpoint, GPU memory, log tail. |
| `logs JOB [-f] [-n N]` | Read or follow the run's log. Ctrl-C stops following, not training. |
| `fetch JOB [--all]` | Download the evidence: final checkpoint, `meta.json`, status, log, eval and events, and verify SHA-256. `--all` takes every checkpoint. Resumable (`--partial`), so a re-run continues. |
| `eval JOB [--steps N] [--metrics]` | Load the newest checkpoint in headless replay for N steps (`play.py --steps`); `--metrics` runs `tools/replay_eval.py` for standing, straight and turning, with fall rate, survival and tracking error. |
| `resume JOB [--max-iterations N]` | New job from the same release and checkpoint, continuing the iteration counter. |
| `stop JOB` | SIGINT the process group, wait, then TERM, then KILL. |
| `list` | Summarise recent jobs. |

A first run, end to end:

```bash
python tools/remote_train.py check
python tools/remote_train.py sync
python tools/remote_train.py bootstrap --release <id from sync>
python tools/remote_train.py submit --task jumper.flat --release <id> \
    --num-envs 64 --max-iterations 20 --max-hours 1
# close the terminal, come back later
python tools/remote_train.py status <job_id>
python tools/remote_train.py fetch <job_id>
python tools/remote_train.py eval <job_id> --steps 200
```

`submit --dry-run` prints the job metadata and the generated `job.sh` without
touching the box.

## Where things land

On the box, under `workdir`:

```
releases/<release_id>/     immutable code snapshot, plus CODE_VERSION.json, CODE.patch,
                           MANIFEST.sha256; its logs/ is a symlink to the running job
runs/<job_id>/             meta.json, job.sh, status.tsv, pid, logs/train.log,
                           tree/ (the run's logs/<model>/<task>/<timestamp>), eval/,
                           root/ (symlinks to the release, with its own logs/)
.venv/                     one environment, re-pointed at the release being run
```

Locally, under `results`:

```
<results>/<job_id>/        the run directory as it was, plus SHA256SUMS and FETCH.json
<results>/<job_id>/eval/   eval_report.json and the replay log
```

**Two jobs never share a log target.** Each job runs in `runs/<id>/root/`, a
directory of symlinks into the release with a real `logs/` beside them; the
release itself is no longer given a `logs` symlink that a second submit could
re-point. That is what makes `--concurrent` safe for the first job's
checkpoints, and why its `status`, `logs` and `stop` stay separate.

**A job's code cannot change under it.** `submit` installs the release into the
shared `.venv` and then checks where `mjrl`, `mjlab`, `rsl_rl` and `tasks` actually
imported from; a mismatch aborts the submit. A later `sync` writes a *new*
release, so a running job keeps the bytes it started with.

## Statuses

| Status | Meaning |
|---|---|
| `queued` | Job directory exists, no start record yet. |
| `running` | The wrapper's pid is alive and there is no end record. |
| `succeeded` | End record with `rc=0` **and** at least one `model_*.pt` on disk. |
| `failed` | A non-zero exit, a timeout (rc 124), or `rc=0` with no checkpoint. |
| `stopped` | An end record after `stop` marked the job. |
| `unknown` | A start record, no end record, and the process gone — a reboot, a crash, or `kill -9`. Never reported as success. |

An unreachable ssh prints `connection error` and exits 2. It never writes `failed`,
because the training may be running perfectly on the other side of a broken link.

## Reproducibility

- The release id is the timestamp plus the git commit; a dirty tree adds a hash of
  the patch (`-dirty-<8 hex>`).
- `CODE_VERSION.json` records the commit, branch, `git status`, the untracked file
  list and the patch hash. `CODE.patch` is the actual diff.
- `MANIFEST.sha256` is computed on the box after the copy and checked against this
  checkout; a missing or differing file aborts the sync.
- `FETCH.json` and `SHA256SUMS` record what was downloaded and its digest.

## Limits

- **Stopping is not a graceful save.** rsl_rl writes a checkpoint every
  `save_interval` iterations (50 by default) and once at the end of a completed
  run; it does not save on SIGINT. `stop` loses the iterations since the last
  checkpoint and says so. A shorter `save_interval` in the task's `rl_cfg.py` is
  the way to make stopping cheaper.
- **One job per GPU by default.** `submit` refuses while another job's pid is
  alive, and refuses when unknown processes use more than 512 MiB, without
  stopping them. `--concurrent` allows a second managed job (a hard cap of two);
  `--force` / `--allow-busy` are the explicit overrides.
- **A full download is many small streams.** Each rsync is one SSH connection;
  on the link measured here one stream moved 0.36–0.49 MB/s and four moved
  1.48 MB/s together, so a whole trajectory is faster fetched in parallel (or
  through the instance's netdisk) than as one long rsync. The default `fetch` is
  selective for the same reason.
- **No automatic shutdown or cleanup.** The instance stays up and results stay on
  it; `fetch` only copies. Turn the machine off yourself.
- **A reboot ends the run.** tmux does not survive it; the job shows `unknown` and
  the last checkpoint is the resume point.
- **No video in this phase.** `eval` loads the checkpoint and replays N steps
  headless; it reports whether that happened, not how well the robot walks. Task
  media rendering stays with `tools/readme_media.py` and a committed export.
- **Upload is tar, download is rsync.** macOS's `/usr/bin/rsync` is openrsync
  2.6.9 and fails on this checkout with `mkpath: Operation not permitted` before
  sending a byte; the upload is therefore one tar stream over ssh. The download
  still uses rsync, where resumability matters.
- **`fetch` skips checkpoints written in the last 10 seconds** and says which; a
  `torch.save` in progress is not a checkpoint.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Host key verification failed` | rsync's own ssh did not get the pinned `known_hosts`; the driver passes `-e` for this reason. |
| `mkpath: Operation not permitted` | openrsync 2.6.9 on macOS; use the tool's tar upload rather than calling `rsync` by hand. |
| `no .venv on the box` | run `bootstrap --release <id>` after `sync`. |
| `python3: command not found` | point `python` at the image's interpreter, e.g. `/root/miniconda3/bin/python`. |
| `torch cannot see CUDA` | the image's torch is CPU-only or the driver is too old; install a matching cu wheel into `.venv`. |
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | no libEGL on the box: set `mujoco_gl = "disabled"` (training) or `"osmesa"` (rendering). |
| `the GPU already has processes` | something else is training; the tool will not kill it. |
| `a job is already running` | `status` shows it; `stop` it or use another instance. |
