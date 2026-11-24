"""Configuration for the post-fetch pipeline.

One TOML file holds every decision made after a download: where fetched jobs
live, which tasks may be evaluated and against which thresholds, how the video
is produced, and where a notification goes. `tools/postfetch.example.toml` is
the template; copy it to `~/.config/jumper/postfetch.toml` (git-ignored) and
edit that.

**Nothing in this file is a secret.** A Feishu webhook URL or app secret is
read from the environment or from the existing DSH Feishu integration; this
file only decides whether to use one. `secrets` is not a configuration key.

Two rules are worth knowing before changing a number here:

- The **evaluation configuration version** is a hash of everything that changes
  what is measured -- the scenarios, steps, episodes, seeds, task profile and
  thresholds. Changing a threshold therefore creates a new unit of work rather
  than quietly reusing an old "pass"; `Config.eval_version` is the only place
  that hash is computed.
- A task is evaluated only if a **profile** names it. A task with no profile is
  reported as unsupported instead of being measured with the walking profile,
  because the walking metrics on another task would look like a number and mean
  nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[2]

#: Copy of `tools/postfetch.example.toml`. It is not in the repository and it
#: never holds a secret; it names the paths and the thresholds.
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "jumper" / "postfetch.toml"

#: The walking sweep, in the order it is reported. `straight` is replay_eval's
#: name for the forward command and is kept so one tool's vocabulary does not
#: need a translation table into the other's.
SCENARIOS: tuple[str, ...] = ("stand", "straight", "backward", "turn")

#: Thresholds that define "the policy performs". A run that clears every one of
#: them is a pass; a run that completes but misses one is still a program
#: success, and the report says both.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "stand_lin_err": 0.06,      # m/s, |v| while commanded to stand still
    "straight_lin_err": 0.10,   # m/s, |vx - cmd| while commanded forward
    "straight_disp_min": 0.50,  # m, net displacement along the commanded heading
    "backward_lin_err": 0.10,   # m/s
    "backward_disp_max": -0.30,  # m, negative because backward is negative x
    "turn_yaw_err": 0.15,       # rad/s, |wz - cmd|
    "fall_rate_max": 0.0,       # falls / episodes
    "min_survival_ratio": 1.0,  # mean survival steps / max steps
}


@dataclass(frozen=True)
class VideoConfig:
    """How the short replay clip is produced.

    `backend = "local"` renders on this machine through MuJoCo's off-screen
    renderer. `backend = "server"` is a declared option with no implementation
    yet: it names what a box would need (EGL and ffmpeg) in the report instead
    of changing the server's environment, which is not this tool's to change.
    """

    enabled: bool = True
    backend: str = "local"
    width: int = 480
    height: int = 360
    fps: int = 25
    seconds: float = 4.0
    max_bytes: int = 8_000_000
    command: tuple[float, float, float] = (0.3, 0.0, 0.0)


@dataclass(frozen=True)
class FeishuConfig:
    """Where the notification goes, without holding the credential.

    `app_id`/`chat_id` may be empty, in which case the existing DSH Feishu
    integration is asked; `webhook_env` names the environment variable that
    would hold a group-bot webhook. `public_base_url` is only for a link that
    the receiver can actually open -- a Mac path is never turned into one.
    """

    enabled: bool = True
    app_id: str = ""
    chat_id: str = ""
    webhook_env: str = "JUMPER_FEISHU_WEBHOOK"
    public_base_url: str = ""


@dataclass(frozen=True)
class Config:
    """Everything the pipeline reads, resolved once and then immutable."""

    results_dir: Path
    out_dir: Path
    state_path: Path
    steps: int
    episodes: int
    seeds: tuple[int, ...]
    scenarios: tuple[str, ...]
    thresholds: dict[str, float]
    profiles: dict[str, str]
    video: VideoConfig
    feishu: FeishuConfig
    watch_interval_s: float
    max_attempts: int
    allow_code_mismatch: bool
    source: Path | None = field(default=None, compare=False)

    def profile_for(self, task: str) -> str | None:
        """The evaluation profile for a task, or None when it has none."""
        return self.profiles.get(task)

    def eval_version(self, task: str) -> str:
        """A short hash of everything that decides what this run measures.

        Deliberately excludes the video and notification settings: a change to
        where a message goes must not make an already-evaluated checkpoint look
        like new work, and a change to a threshold must.
        """
        payload = {
            "profile": self.profile_for(task),
            "scenarios": list(self.scenarios),
            "steps": self.steps,
            "episodes": self.episodes,
            "seeds": list(self.seeds),
            "thresholds": dict(sorted(self.thresholds.items())),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _section(document: dict, name: str) -> dict:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a table")
    return value


def _path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load(path: str | Path | None = None) -> Config:
    """Read the TOML file, falling back to the documented defaults.

    The search order is `--config`, `$JUMPER_POSTFETCH_CONFIG`, then
    `~/.config/jumper/postfetch.toml`. A missing file is not an error: the
    defaults point at `outputs/remote` and `outputs/postfetch`, which is what a
    checkout that has just run `remote_train.py fetch` holds.
    """
    explicit = path or os.environ.get("JUMPER_POSTFETCH_CONFIG")
    source = Path(explicit).expanduser() if explicit else DEFAULT_CONFIG_PATH
    document: dict = {}
    if source.is_file():
        with source.open("rb") as handle:
            document = tomllib.load(handle)
    elif explicit:
        raise FileNotFoundError(f"postfetch config not found: {source}")

    base = source.parent if source.is_file() else REPO
    paths = _section(document, "paths")
    eval_doc = _section(document, "eval")
    video_doc = _section(document, "video")
    feishu_doc = _section(document, "feishu")
    watch_doc = _section(document, "watch")

    thresholds = dict(DEFAULT_THRESHOLDS)
    for key, value in (_section(eval_doc, "thresholds")).items():
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"eval threshold {key!r} is not a number") from exc

    profiles = {str(key): str(value) for key, value in _section(document, "profiles").items()}
    if "profiles" not in document:
        profiles = {"jumper.flat": "walking"}

    steps = int(eval_doc.get("steps", 250))
    episodes = int(eval_doc.get("episodes", 2))
    seeds = tuple(int(seed) for seed in eval_doc.get("seeds", [0, 1]))
    scenarios = tuple(str(name) for name in eval_doc.get("scenarios", list(SCENARIOS)))
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"eval scenarios {unknown} are not part of the walking sweep")
    if steps <= 0 or episodes <= 0 or not seeds:
        raise ValueError("eval steps, episodes and seeds must all be positive")

    command = video_doc.get("command", [0.3, 0.0, 0.0])
    video = VideoConfig(
        enabled=bool(video_doc.get("enabled", True)),
        backend=str(video_doc.get("backend", "local")),
        width=int(video_doc.get("width", 480)),
        height=int(video_doc.get("height", 360)),
        fps=int(video_doc.get("fps", 25)),
        seconds=float(video_doc.get("seconds", 4.0)),
        max_bytes=int(video_doc.get("max_bytes", 8_000_000)),
        command=tuple(float(value) for value in command),  # type: ignore[arg-type]
    )
    if len(video.command) != 3:
        raise ValueError("video.command must be exactly [vx, vy, wz]")
    if video.fps <= 0 or video.width < 64 or video.height < 64 or video.seconds <= 0:
        raise ValueError("video fps, width, height and seconds must be positive")

    feishu = FeishuConfig(
        enabled=bool(feishu_doc.get("enabled", True)),
        app_id=str(feishu_doc.get("app_id", "")),
        chat_id=str(feishu_doc.get("chat_id", "")),
        webhook_env=str(feishu_doc.get("webhook_env", "JUMPER_FEISHU_WEBHOOK")),
        public_base_url=str(feishu_doc.get("public_base_url", "")),
    )

    out_dir = _path(paths.get("out", "outputs/postfetch"), base)
    results_dir = _path(paths.get("results", "outputs/remote"), base)
    state_path = _path(paths.get("state", out_dir / "state.json"), base)

    return Config(
        results_dir=results_dir,
        out_dir=out_dir,
        state_path=state_path,
        steps=steps,
        episodes=episodes,
        seeds=seeds,
        scenarios=scenarios,
        thresholds=thresholds,
        profiles=profiles,
        video=video,
        feishu=feishu,
        watch_interval_s=float(watch_doc.get("interval_s", 60.0)),
        max_attempts=int(watch_doc.get("max_attempts", 3)),
        allow_code_mismatch=bool(watch_doc.get("allow_code_mismatch", False)),
        source=source if source.is_file() else None,
    )
