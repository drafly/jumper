# AutoDL lifecycle: start on demand, shut down after the sync

The training driver (`tools/remote_train.py`) assumes the GPU box is running.
This tool owns the rest of the day: it can start the instance when a training
run is due, waits for the run to reach a terminal status, fetches and verifies
its results, and only then powers the instance off.

```
tools/remote_train.py fetch  ->  what came back
tools/postfetch/            ->  evaluate, record, notify
tools/autodl_lifecycle.py   ->  the box exists only around that
```

## Commands

```bash
python tools/autodl_lifecycle.py status     # API status, SSH, running jobs
python tools/autodl_lifecycle.py up         # power on and wait for SSH
python tools/autodl_lifecycle.py verify     # the sync gate only
python tools/autodl_lifecycle.py down       # sync gate, then shut down
python tools/autodl_lifecycle.py session --task jumper.flat
```

`session` is the one-call form: `up -> sync -> submit -> wait -> fetch -> verify
-> down`. `--dry-run` prints every step and touches nothing. `down --fetch-missing`
fetches any terminal run the Mac does not hold before allowing the shutdown.

## Container instances Pro: unattended create

The standard container API has no unattended create (the console request carries
an Aliyun captcha). The official Pro API does, and it is the only route where a
new GPU instance can be created without a human:

```toml
[autodl]
api = "pro"
instance_uuid = ""            # empty: find by name, or create when --rent-new
token_env = "AUTODL_TOKEN"    # or token_file = an ignored 600 file
pro_gpu_spec = "v-32g-p"      # 4080S-32G; see the docs appendix for the rest
pro_cuda_from = 128
pro_image_uuid = "base-image-l2t43iu6uk"
pro_instance_name = "jumper-pro"
pro_system_disk_gb = 50
```

```bash
python tools/autodl_lifecycle.py pro-plan                 # read-only payload preview
python tools/autodl_lifecycle.py up --rent-new --dry-run  # the same, in the up path
```

Three things to know before using it:

- **There is no RTX 3080 Ti in the Pro catalogue.** The documented specs are
  3090-48G (`v-48g-350w`), 4080S-32G (`v-32g-p`), 4090-48G (`v-48g`), 4090D,
  PRO6000-96G (`pro6000-p`), 5090-32G and H800. Pick a spec explicitly.
- **The public Pro images stop at PyTorch 2.0 / CUDA 11.8**, while this
  repository needs torch>=2.7. The first bootstrap therefore installs torch 2.8;
  once that instance is set up, `pro/image/save` turns it into a private image
  and later creates start from that instead.
- **Pro instances have a system disk, not a data disk.** The workdir lives under
  `/root`, and `pro_system_disk_gb` expands the system disk at create time.
- The token is a Developer Token (console -> account -> settings -> developer
  token), not the browser JWT; it is long-lived, so this path has no re-login.

A shutdown can still return the card to the pool, but unlike the standard path
the official API can create a replacement unattended; the practical model is
"create per training campaign (fast once a private image exists), release or
shut down when done".

## When the old host has no card: scan the market

A shut-down instance has no GPU reserved; the host may hand its card to somebody
else, and `up` then reports `机器空闲GPU不足`. Two answers, and the cheap one
first:

```bash
# read-only: which machines have a free card, at what price (no cost)
python tools/autodl_lifecycle.py market

# what a replacement would look like: old-host retry first, then a market plan
python tools/autodl_lifecycle.py up --rent-new --dry-run
```

`market` asks the console's own machine list for the configured
`autodl.gpu_types` and prints idle counts and ¥/hour. The plan prefers the
configured GPU order, then the machine with most idle cards, then the cheapest.
Creating a replacement instance is deliberately **not enabled**: it costs money
and needs an image (`autodl.image_type` = base/community/private plus the
matching URL or UUID). Until an image is configured the plan says exactly what is
missing. The old instance and its data disk are never deleted.

## Keep the instance running

A low-end GPU is hard to rent again once it is released, so `session` does **not**
shut the instance down when a run finishes. It fetches, verifies the sync gate and
leaves the box running; say so and run `down` when you actually want it stopped.
`--auto-shutdown` (or `auto_shutdown = true`) brings back the old behaviour.
`down` still refuses to power off a box whose runs are not fetched and verified.

## The shutdown gate

`down` refuses when either is true:

1. **A job is still running or queued.** `--force` is the explicit override.
2. **A terminal run is not safe on this Mac.** Every run the box has marked
   `succeeded`, `failed`, `stopped` or `unknown` must have a local
   `outputs/remote/<job_id>/FETCH.json` from `remote_train.py fetch` with
   `verified > 0`, no `missing` entries, and the final checkpoint must still
   hash to the box's copy. `--force` overrides this too, and the audit log
   records that it was forced.

This is the step that makes automatic shutdown safe: a machine that powers off
between the training and the download is how a run's only copy is lost, and a
`tmux` session that died is not "finished".

## Two platform facts this is built on

**Stopping needs no credential.** AutoDL documents stopping from inside the
container in [省钱绝招, 场景三](https://www.autodl.com/docs/save_money/):
`python train.py; /usr/bin/shutdown`. That is what `shutdown = "ssh"` runs, after
the gate. It works for a standard container instance.

**Starting does need one of three things.** AutoDL's official API
([container instance Pro](https://www.autodl.com/docs/instance_pro_api/)) is
documented for Pro instances and wants a developer token from
console -> account -> settings -> developer token. A standard container
instance has no official start API; the web console's own JSON API can start it
with a browser JWT, but that token expires. So:

| `autodl.api` | instance | credential | longevity |
|---|---|---|---|
| `pro` | container instance Pro (`pro-...`) | developer token | stable |
| `console` | standard container instance | web-console JWT | expires; not official |
| `browser` | standard container instance | Playwright + logged-in Chrome profile | session lasts until the console logs out |
| `manual` | any | none | you start it; everything else is automatic |

The default is `manual`, because the safe half (wait, gate, shutdown) works
before any credential is configured.

### Using the browser backend (standard container instance)

This is the option for a normal container instance when capturing a JWT by hand
is not wanted. Playwright opens the console with a **persistent Chrome profile**;
the profile is logged in once by a human (AutoDL offers WeChat/SMS/sub-account
login, so there is no unattended login), and from then on every command captures
a fresh `Authorization` header from the console's own XHR headlessly and feeds it
to the same console API the `console` backend uses. Power actions therefore do
not depend on clicking 开机/关机 buttons, which is the part of the page most
likely to change.

The lifecycle file is found in this order: `--config`, `$JUMPER_AUTODL_CONFIG`,
`.autodl.toml` in the repository checkout, `~/.config/jumper/autodl.toml`. The
checkout's `.autodl.toml` is git-ignored; the browser profile it names lives
under `outputs/autodl/` (also ignored) unless given as an absolute path.

```bash
pip install -e ".[browser]"          # Playwright only; the system Chrome is used
python tools/autodl_lifecycle.py browser-auth --wait-seconds 180
#   a Chrome window opens: scan the WeChat QR / enter the SMS code, then wait
python tools/autodl_lifecycle.py discover
python tools/autodl_lifecycle.py up
```

```toml
[autodl]
api = "browser"
instance_uuid = ""     # discovered by SSH port
browser_profile = "~/.config/jumper/autodl-browser-profile"
browser_headless = true
```

**The profile is a full account session**, not a scoped token: it is created
owner-only (700), it must stay outside the repository, and it must never be
copied into an artifact, a log or a message. If the console logs the session out,
`browser-auth` is the only way back in -- there is no unattended re-login.
`browser-auth` also writes a `storage_state` backup (600) beside it; the tool
never reads that file, it only keeps the persistent profile.

If Chrome is not installed, the profile falls back to Playwright's own Chromium
(`playwright install chromium`, about 170 MB); with Chrome present nothing is
downloaded beyond the small Playwright package.

### Using the console JWT (standard container instance)

1. Log in to <https://www.autodl.com>, open a console page that talks to the
   API, open the browser devtools (F12) -> Network, and copy the whole
   `Authorization` request header from any XHR request. It is a JWT, not the
   developer token.
2. Keep it out of the repository and out of the config file:

   ```bash
   mkdir -p ~/.config/jumper
   printf '%s' '<the JWT>' > ~/.config/jumper/autodl-token
   chmod 600 ~/.config/jumper/autodl-token
   ```

3. Point the config at it and let the tool find the instance by its SSH port:

   ```toml
   [autodl]
   api = "console"
   token_file = "~/.config/jumper/autodl-token"
   instance_uuid = ""        # empty: discovered from the SSH port
   ```

   ```bash
   python tools/autodl_lifecycle.py discover   # prints every instance and the match
   python tools/autodl_lifecycle.py up         # powers it on
   ```

The JWT expires. When it does, every API call answers `AuthorizeFailed` and the
tool says so instead of pretending the box is off; capture a new one. That
short lifetime is the reason `pro` + developer token is the stable path if the
instance can be a Pro one.

## Configuration

Copy `tools/autodl.example.toml` to `~/.config/jumper/autodl.toml`. SSH settings
come from `.remote.toml`; the lifecycle file only adds the power stage. The token
is never in that file: `AUTODL_TOKEN` in the environment, or `token_file` (mode
600, outside the repository). The tool warns on a token file other users can
read and never writes the token to the audit log.

## Audit

`outputs/autodl/lifecycle.jsonl` records each `power_on` / `power_off` / `up`
action with time, instance identifier and host. It contains no credential.

## Limits

- The gate covers the run directories (`workdir/runs`). It does not decide what
  else on the data disk matters; that is a human decision. Nothing is deleted.
- A shutdown over SSH is observed by SSH stopping to answer, which is what
  AutoDL does; with an API configured the status is polled as well. The instance
  can take a minute or two to disappear from the console.
- `session` leaves the instance running if a step before the gate fails. A
  machine in an unknown state is left for a human rather than shut down.
- **A shutdown returns the GPU to the pool.** The instance's GPU is its
  specification, not a reservation: once the instance is shut down, AutoDL can
  rent that card to somebody else, and the next start must find a free card of
  the same model on the host. When none is free, the API answers
  `BadRequest: 机器空闲GPU不足` (see
  [省钱绝招](https://www.autodl.com/docs/save_money/): "正常开机时如果GPU被其他用户租用，
  可能出现空闲GPU不足"). This is the platform, not the request: `up
  --retry-minutes N` / `session --retry-minutes N` retry with a growing delay,
  or clone/migrate the instance to a host that has a free card, or leave it
  running while a training campaign is in progress.
- Shutting down on AutoDL keeps the data disk; the run directories stay on the
  box and can be fetched after the next start. "Abandoning unsynced runs" with
  `--force` means they were not downloaded now, not that they were deleted.
