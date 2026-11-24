"""Send the post-fetch evaluation result to Feishu -> one message, best effort.

This module is deliberately standalone: the post-fetch pipeline imports it and it
imports nothing else from `tools/postfetch/`, so it can be developed and tested on
its own. It reuses the Feishu connection that already exists on this machine --
the bot in the DSH integration and the workspace-to-chat mapping in
`~/.dsh/integrations/dsh-feishu/workspaces.json` -- instead of creating a new bot.

Configuration is resolved field by field: explicit overrides passed to
:func:`discover` > `JUMPER_FEISHU_*` environment variables > the DSH
configuration. A workspace that is not registered in the mapping produces an
empty chat id rather than a guessed group.

Delivery has one payload, the Chinese text summary; the video is an optional
attachment on top of it. A rejected video upload (usually a permission the bot
was never granted) is never retried -- the summary has already been delivered,
and the result says the video failed. Webhook mode cannot attach files at all and
says so in its result. The app secret and the webhook URL never leave the
process: neither appears in the config's `repr()`, in the returned `details`, in
an error reason (URLs are masked, and the configured webhook is replaced whole)
or in the preview queue file.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["FeishuConfig", "build_summary", "discover", "preview", "send"]

FEISHU_API = "https://open.feishu.cn/open-apis"
_TOKEN_URL = f"{FEISHU_API}/auth/v3/tenant_access_token/internal"
_MESSAGES_URL = f"{FEISHU_API}/im/v1/messages?receive_id_type=chat_id"
_FILES_URL = f"{FEISHU_API}/im/v1/files"

_VIDEO_FILE_TYPE = "mp4"
_VIDEO_MIME = "video/mp4"
_JSON_HEADERS = {"Content-Type": "application/json; charset=utf-8"}
#: Feishu answers a permission the app does not have, or a malformed upload,
#: with one of these codes; an HTTP 400/403 means the same thing.
_PERMISSION_CODES = frozenset({99991672, 99991663, 234001, 234006})
_DEFAULT_TIMEOUT_S = 10.0

_URL_RE = re.compile(r"https?://[^\s'\"<>]+")
#: Absolute local paths are not actionable in a chat message, and a Mac path in
#: one is a promise the reader cannot keep.
_LOCAL_PATH_RE = re.compile(
    r"(?<![\w:/])/(?:Users|home|private|tmp|var|root|opt|mnt|Volumes|srv|data)/"
    r"[^\s，。；,;:)]*"
)

_VERDICT_LABELS = {
    "pass": "达标",
    "passed": "达标",
    "ok": "达标",
    "success": "达标",
    "达标": "达标",
    "fail": "未达标",
    "failed": "未达标",
    "failure": "未达标",
    "未达标": "未达标",
    "unsupported": "不支持",
    "not_supported": "不支持",
    "skip": "不支持",
    "skipped": "不支持",
    "不支持": "不支持",
    "error": "程序失败",
    "program_error": "程序失败",
    "crash": "程序失败",
    "exception": "程序失败",
    "程序失败": "程序失败",
}

_VIDEO_LABELS = {
    "ok": "已录制",
    "sent": "已录制",
    "recorded": "已录制",
    "success": "已录制",
    "已录制": "已录制",
    "failed": "录制失败",
    "failure": "录制失败",
    "error": "录制失败",
    "录制失败": "录制失败",
    "skipped": "未录制",
    "not_attempted": "未录制",
    "unsupported": "不支持",
    "none": "未录制",
    "未录制": "未录制",
}


@dataclass
class FeishuConfig:
    """Where a notification can go and how to reach it.

    `app_secret` and `webhook_url` are left out of `repr()` on purpose: a config
    object is the kind of thing that ends up in a traceback or a log line, and
    neither value may appear there.
    """

    mode: str
    app_id: str = ""
    app_secret: str = field(default="", repr=False)
    chat_id: str = ""
    webhook_url: str = field(default="", repr=False)
    public_base_url: str = ""
    timeout_s: float = 10.0


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _text(value: object) -> str:
    """A stripped string, or "" for anything that is not a string."""
    return value.strip() if isinstance(value, str) else ""


def _one_line(value: object) -> str:
    """A single line for a chat message; None becomes empty, never "None"."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _first(*values: object) -> str:
    """The first non-empty string, in priority order."""
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _timeout(cfg: FeishuConfig) -> float:
    try:
        timeout = float(cfg.timeout_s)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return timeout if timeout > 0 else _DEFAULT_TIMEOUT_S


def _mask_url(match: re.Match[str]) -> str:
    """Keep the host, drop the path: a token in a URL is the secret."""
    try:
        parts = urlsplit(match.group(0))
    except ValueError:
        return "<redacted url>"
    host = parts.hostname or ""
    if not host:
        return "<redacted url>"
    return f"{parts.scheme}://{host}/<redacted>"


def _scrub(value: object, cfg: FeishuConfig) -> str:
    """Make one string safe to log, return or print next to a config."""
    text = str(value)
    if cfg.webhook_url:
        text = text.replace(cfg.webhook_url, "<redacted webhook>")
    if cfg.app_secret:
        text = text.replace(cfg.app_secret, "<redacted secret>")
    return _URL_RE.sub(_mask_url, text)


def _scrub_obj(value: object, cfg: FeishuConfig) -> object:
    """Recursively scrub every string in a JSON-shaped value."""
    if isinstance(value, str):
        return _scrub(value, cfg)
    if isinstance(value, Mapping):
        return {str(key): _scrub_obj(item, cfg) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_obj(item, cfg) for item in value]
    return value


def _result(
    status: str,
    cfg: FeishuConfig,
    *,
    reason: str = "",
    video: str = "not_attempted",
    details: Mapping[str, object] | None = None,
) -> dict:
    """The one shape `send` returns on every path."""
    return {
        "status": status,
        "mode": cfg.mode,
        "reason": _scrub(reason, cfg),
        "video": video,
        "details": _scrub_obj(dict(details or {}), cfg),
    }


def _initial_video_state(video: Path | None) -> str:
    return "not_attempted" if video is None else "skipped"


def _missing_fields(cfg: FeishuConfig) -> list[str]:
    """Name the settings that keep this configuration from sending anything."""
    if cfg.mode == "webhook":
        return [] if cfg.webhook_url else ["webhook_url"]
    missing: list[str] = []
    if not cfg.app_id:
        missing.append("app_id")
    if not cfg.app_secret:
        missing.append("app_secret")
    if not cfg.chat_id:
        missing.append("chat_id")
    return missing


def _exception_text(exc: BaseException) -> str:
    text = _one_line(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# --------------------------------------------------------------------------
# discovering the existing connection
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    """Read a JSON object, or return {} for anything unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_ref(dsh_dir: Path, ref: str) -> str:
    """Resolve one `refs.<name>` entry of the DSH credentials file.

    An unreadable or malformed file only means the secret is absent (and `send`
    reports that); it must never make discovery raise or echo the file.
    """
    if not ref:
        return ""
    try:
        import yaml

        data = yaml.safe_load((dsh_dir / ".credentials.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absence of a credential is a normal state
        return ""
    refs = data.get("refs") if isinstance(data, dict) else None
    if not isinstance(refs, Mapping):
        return ""
    return _text(refs.get(ref))


def _dsh_bot(dsh_dir: Path) -> dict:
    """The DSH Feishu bot that is already connected on this machine."""
    data = _read_json(dsh_dir / "integrations" / "dsh-feishu" / "config.json")
    bots = data.get("bots")
    if not isinstance(bots, list):
        return {}
    for bot in bots:
        if isinstance(bot, dict) and _text(bot.get("appId")):
            return bot
    return {}


def _norm_path(path: object) -> str:
    """Compare workspace paths the way the filesystem does (symlinks, case)."""
    try:
        return os.path.normcase(str(Path(str(path)).expanduser().resolve()))
    except (OSError, ValueError, RuntimeError):
        return os.path.normcase(str(path))


def _chat_id_from_key(key: str) -> str:
    """Extract a chat id from a `group:<id>` / `<id>` conversation key."""
    candidate = key
    if ":" in candidate:
        prefix, rest = candidate.split(":", 1)
        if prefix in {"group", "chat", "p2p", "dm", "conversation"}:
            candidate = rest
    candidate = candidate.strip()
    return candidate if candidate.startswith("oc_") else ""


def _chat_for_workspace(data: dict, bot_id: str, workspace: Path) -> str:
    """Find the group whose registered workspace is this one.

    Never guess: when the workspace is not in the mapping, the chat id is empty
    and `send` reports it as blocked. An exact match wins; otherwise the deepest
    registered ancestor wins, so running from a subdirectory of the repository
    still reaches the same group.
    """
    conversation = data.get("conversationWorkspaces")
    if not isinstance(conversation, Mapping):
        return ""
    mappings: list[Mapping] = []
    preferred = conversation.get(bot_id)
    if isinstance(preferred, Mapping):
        mappings.append(preferred)
    for key, mapping in conversation.items():
        if key != bot_id and isinstance(mapping, Mapping):
            mappings.append(mapping)

    target = _norm_path(workspace)
    found: list[tuple[int, int, str]] = []
    for mapping in mappings:
        for conv_key, mapped in mapping.items():
            if not isinstance(conv_key, str) or not isinstance(mapped, str):
                continue
            registered = _norm_path(mapped)
            if registered == target:
                rank = 0
            elif target.startswith(registered + os.sep):
                rank = 1
            else:
                continue
            chat_id = _chat_id_from_key(conv_key)
            if chat_id:
                found.append((rank, -len(registered), chat_id))
    return min(found)[2] if found else ""


def discover(
    workspace: Path,
    overrides: dict | None = None,
    *,
    dsh_dir: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> FeishuConfig:
    """Resolve where a notification for `workspace` should go.

    `overrides` may hold `app_id`, `app_secret`, `chat_id`, `webhook_url`,
    `public_base_url` and `timeout_s`; it may also hold `dsh_dir`, which is the
    same thing as the keyword argument. The environment variables
    `JUMPER_FEISHU_APP_ID`, `JUMPER_FEISHU_APP_SECRET`, `JUMPER_FEISHU_CHAT_ID`,
    `JUMPER_FEISHU_WEBHOOK` and `JUMPER_FEISHU_PUBLIC_BASE_URL` sit between the
    overrides and the DSH configuration, and `env` replaces `os.environ` so a
    test can isolate the lookup.

    `dsh_dir` defaults to `~/.dsh`; the bot's app id and secret reference come
    from its integration config, the secret from `.credentials.yaml`, and the
    chat id from `workspaces.json` by workspace path. Values are resolved field
    by field, but the DSH secret is only used when the DSH app id is the one in
    effect, because a secret belongs to an app rather than to a machine.

    The mode is `"app"` only when app id, secret and chat id are all present,
    `"webhook"` when a webhook is configured (it cannot carry a video), and
    `"none"` otherwise. With nothing found at all every field stays empty; a
    partial discovery keeps what it found so `send` can name exactly what is
    missing.
    """
    overrides = dict(overrides or {})
    environ: Mapping[str, str] = os.environ if env is None else env
    if dsh_dir is None:
        dsh_dir = overrides.get("dsh_dir")
    root = Path(dsh_dir).expanduser() if dsh_dir is not None else Path.home() / ".dsh"

    app_id = _first(overrides.get("app_id"), environ.get("JUMPER_FEISHU_APP_ID"))
    app_secret = _first(overrides.get("app_secret"), environ.get("JUMPER_FEISHU_APP_SECRET"))
    chat_id = _first(overrides.get("chat_id"), environ.get("JUMPER_FEISHU_CHAT_ID"))
    webhook_url = _first(overrides.get("webhook_url"), environ.get("JUMPER_FEISHU_WEBHOOK"))
    public_base_url = _first(
        overrides.get("public_base_url"), environ.get("JUMPER_FEISHU_PUBLIC_BASE_URL")
    )

    bot = _dsh_bot(root)
    if bot:
        dsh_app_id = _text(bot.get("appId"))
        app_id = app_id or dsh_app_id
        # A secret belongs to an app, not to a machine: only fill it from the
        # bot whose app id is the one actually in effect.
        if not app_secret and app_id == dsh_app_id:
            app_secret = _read_ref(root, _text(bot.get("secretRef")))
        if not chat_id:
            chat_id = _chat_for_workspace(
                _read_json(root / "integrations" / "dsh-feishu" / "workspaces.json"),
                _text(bot.get("id")),
                Path(workspace),
            )

    timeout = overrides.get("timeout_s")
    try:
        timeout_s = float(timeout) if timeout not in (None, "") else _DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        timeout_s = _DEFAULT_TIMEOUT_S

    if webhook_url:
        mode = "webhook"
    elif app_id and app_secret and chat_id:
        mode = "app"
    else:
        mode = "none"
    return FeishuConfig(
        mode=mode,
        app_id=app_id,
        app_secret=app_secret,
        chat_id=chat_id,
        webhook_url=webhook_url,
        public_base_url=public_base_url,
        timeout_s=timeout_s,
    )


# --------------------------------------------------------------------------
# the Chinese summary
# --------------------------------------------------------------------------


def build_summary(
    task: str,
    job_id: str,
    checkpoint: str,
    verdict: str,
    metrics: list[tuple[str, str]],
    video_status: str,
    reason: str = "",
) -> str:
    """Compose the Chinese chat summary for one evaluated checkpoint.

    The verdict is mapped to one of 达标 / 未达标 / 不支持 / 程序失败 (English
    spellings and the Chinese labels themselves are accepted). Only the
    checkpoint's file name is printed -- never its directory -- and an absolute
    local path in `reason` is masked, because a chat reader cannot open either.
    """
    verdict_label = _VERDICT_LABELS.get(_text(verdict).lower())
    if verdict_label is None:
        verdict_label = _one_line(verdict) or "未知"
    checkpoint_name = Path(_one_line(checkpoint)).name if _one_line(checkpoint) else "（未提供）"
    lines = [
        "【Jumper 自动评估】",
        f"任务：{_one_line(task) or '未知'}",
        f"Job：{_one_line(job_id) or '未知'}",
        f"Checkpoint：{checkpoint_name}",
        f"评估结果：{verdict_label}",
    ]
    if metrics:
        lines.append("关键指标：")
        for name, value in metrics:
            lines.append(f"- {_one_line(name) or '指标'}：{_one_line(value) or '—'}")
    else:
        lines.append("关键指标：无")
    video_label = _VIDEO_LABELS.get(_text(video_status).lower())
    lines.append(f"录像：{video_label or _one_line(video_status) or '未知'}")
    clean_reason = _LOCAL_PATH_RE.sub("<path>", _one_line(reason))
    if clean_reason:
        lines.append(f"原因：{clean_reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the queued preview
# --------------------------------------------------------------------------


def preview(cfg: FeishuConfig, text: str, out_dir: Path) -> Path:
    """Queue the message to be sent later, as JSON, without any secret.

    The file records what the message says, how it would have travelled, what is
    missing, and when it was written. It deliberately does not record the app
    secret or the webhook URL, so a queued notification is safe to keep and to
    show; `pipeline.show_preview` prints its `text`.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    target = directory / f"feishu-{created.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    payload = {
        "version": 1,
        "kind": "feishu-notify",
        "created_at": created.isoformat(),
        "mode": cfg.mode,
        "text": text,
        "missing": _missing_fields(cfg),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# sending
# --------------------------------------------------------------------------


def _post_json(
    url: str,
    payload: Mapping[str, object],
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> object:
    """POST one JSON body; the real transport. Monkeypatch this in a test."""
    import requests

    return requests.post(url, json=payload, headers=dict(headers or {}), timeout=timeout)


def _post_multipart(
    url: str,
    *,
    data: Mapping[str, str],
    files: Mapping[str, object],
    headers: Mapping[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> object:
    """POST one multipart form; the real transport. Monkeypatch this in a test."""
    import requests

    return requests.post(url, data=data, files=files, headers=dict(headers or {}), timeout=timeout)


def _unavailable(name: str) -> Callable[..., object]:
    """A transport slot that reports itself when the video needs it."""

    def call(*args: object, **kwargs: object) -> object:
        raise RuntimeError(f"transport does not implement {name}")

    return call


def _resolve_transport(
    transport: object | None,
) -> tuple[Callable[..., object], Callable[..., object]]:
    """Pick the two POST callables, allowing a fake to be injected.

    A test transport exposes `post_json` and/or `post_multipart` with the same
    keyword arguments as the module functions above; a `requests`-like object
    with a single `post` is adapted. A missing half is replaced by a callable
    that raises only if reached, so a text-only path never needs the upload half.
    """
    if transport is None:
        return _post_json, _post_multipart
    post_json = getattr(transport, "post_json", None)
    post_multipart = getattr(transport, "post_multipart", None)
    if callable(post_json) or callable(post_multipart):
        return (
            post_json if callable(post_json) else _unavailable("post_json"),
            post_multipart if callable(post_multipart) else _unavailable("post_multipart"),
        )
    post = getattr(transport, "post", None)
    if callable(post):
        def json_via_post(url: str, payload: Mapping[str, object], *,
                          headers: Mapping[str, str] | None = None,
                          timeout: float = _DEFAULT_TIMEOUT_S) -> object:
            return post(url, json=payload, headers=dict(headers or {}), timeout=timeout)

        def multipart_via_post(url: str, *, data: Mapping[str, str],
                               files: Mapping[str, object],
                               headers: Mapping[str, str] | None = None,
                               timeout: float = _DEFAULT_TIMEOUT_S) -> object:
            return post(url, data=data, files=files, headers=dict(headers or {}),
                        timeout=timeout)

        return json_via_post, multipart_via_post
    raise TypeError("transport must expose post_json/post_multipart or a requests-like post")


def _decode(reply: object) -> tuple[int, dict]:
    """Normalize a transport reply to (HTTP status, JSON body)."""
    if isinstance(reply, Mapping):
        return 200, dict(reply)
    try:
        status = int(getattr(reply, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status = 0
    try:
        payload = reply.json()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - a non-JSON body is an interface failure
        payload = {}
    return status, dict(payload) if isinstance(payload, Mapping) else {}


def _reply_code(payload: Mapping[str, object]) -> tuple[object, str]:
    """The status code and message from a Feishu reply."""
    if "code" in payload:
        return payload["code"], _one_line(payload.get("msg"))
    if "StatusCode" in payload:
        return payload["StatusCode"], _one_line(payload.get("StatusMessage"))
    return None, ""


def _describe(status: int, code: object, message: str) -> str:
    base = f"HTTP {status}, code {code}"
    return f"{base}: {message}" if message else base


def _nested_string(payload: Mapping[str, object], *keys: str) -> str:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return _text(current)


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}


def _upload_headers(token: str) -> dict[str, str]:
    """Authorization only: requests sets the multipart boundary itself.

    Sending `_auth_headers` here set `Content-Type: application/json` on the
    multipart body; Feishu answered `234001 Invalid request param` for an upload
    that was otherwise fine, and the clip was dropped with the summary already
    sent. That is silent enough to deserve its own function.
    """
    return {"Authorization": f"Bearer {token}"}


def send(
    cfg: FeishuConfig,
    text: str,
    video: Path | None = None,
    transport: object | None = None,
) -> dict:
    """Send one message, and attach `video` when the app API accepts it.

    One attempt per request, no retry and no state: `status` is `"ok"`,
    `"failed"`, `"blocked"` (no usable configuration, `reason` names what is
    missing) or `"skipped"` (an empty message), and `video` reports
    independently as `"sent"`, `"failed"`, `"skipped"` or `"not_attempted"`.

    A video that Feishu rejects -- usually a permission the bot does not have,
    HTTP 400/403 or one of the documented codes -- is not retried and does not
    turn the delivered summary into a failure: `status` stays `"ok"` and
    `video` says `"failed"`. Webhook mode sends the text only. `transport` is
    the test seam described in `_resolve_transport`; every return path scrubs
    the secret and the webhook URL from `reason` and `details`.
    """
    video_path = Path(video) if video is not None else None
    message = "" if text is None else str(text)

    if cfg.mode == "none":
        missing = _missing_fields(cfg)
        if missing:
            reason = (
                "Feishu is not configured: missing " + ", ".join(missing)
                + " (set JUMPER_FEISHU_APP_ID / JUMPER_FEISHU_APP_SECRET / "
                "JUMPER_FEISHU_CHAT_ID, set JUMPER_FEISHU_WEBHOOK, or register this "
                "workspace with the DSH Feishu integration)"
            )
        else:
            reason = "Feishu notifications are disabled (mode is none)"
        return _result(
            "blocked", cfg, reason=reason,
            video=_initial_video_state(video_path), details={"missing": missing},
        )
    if cfg.mode not in {"app", "webhook"}:
        return _result(
            "failed", cfg, reason=f"unknown Feishu mode: {cfg.mode}",
            video=_initial_video_state(video_path),
        )
    if not message.strip():
        return _result(
            "skipped", cfg, reason="nothing to send: the message text is empty",
            video=_initial_video_state(video_path),
        )
    if cfg.mode == "app":
        return _send_app(cfg, message, video_path, transport)
    return _send_webhook(cfg, message, video_path, transport)


def _send_app(
    cfg: FeishuConfig, text: str, video: Path | None, transport: object | None
) -> dict:
    """Token, then the text summary, then (optionally) the video as media."""
    post_json, post_multipart = _resolve_transport(transport)
    timeout = _timeout(cfg)
    details: dict[str, object] = {}

    try:
        reply = post_json(
            _TOKEN_URL,
            {"app_id": cfg.app_id, "app_secret": cfg.app_secret},
            headers=dict(_JSON_HEADERS),
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - a transport error must not escape send()
        details["token"] = "request_error"
        return _result(
            "failed", cfg,
            reason=f"tenant access token request failed: {_exception_text(exc)}",
            details=details,
        )
    status, payload = _decode(reply)
    code, message = _reply_code(payload)
    token = _nested_string(payload, "tenant_access_token")
    if status != 200 or code != 0 or not token:
        details.update({"token": "failed", "http": status, "code": code})
        return _result(
            "failed", cfg,
            reason=f"tenant access token rejected ({_describe(status, code, message)})",
            details=details,
        )
    details["token"] = "ok"

    body = {
        "receive_id": cfg.chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False),
    }
    try:
        reply = post_json(_MESSAGES_URL, body, headers=_auth_headers(token), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        details["text_message"] = "request_error"
        return _result(
            "failed", cfg,
            reason=f"text message request failed: {_exception_text(exc)}",
            details=details,
        )
    status, payload = _decode(reply)
    code, message = _reply_code(payload)
    if status != 200 or code != 0:
        details.update({"text_message": "failed", "http": status, "code": code})
        return _result(
            "failed", cfg,
            reason=f"text message rejected ({_describe(status, code, message)})",
            details=details,
        )
    details["text_message"] = "ok"
    message_id = _nested_string(payload, "data", "message_id")
    if message_id:
        details["message_id"] = message_id

    if video is None:
        return _result("ok", cfg, details=details)
    return _send_video(cfg, video, token, post_json, post_multipart, timeout, details)


def _send_video(
    cfg: FeishuConfig,
    video: Path,
    token: str,
    post_json: Callable[..., object],
    post_multipart: Callable[..., object],
    timeout: float,
    details: dict[str, object],
) -> dict:
    """Upload the video and send it as a media message; never retry a rejection."""
    name = video.name or "video.mp4"
    try:
        handle = video.open("rb")
    except OSError:
        details["video"] = "unreadable"
        return _result(
            "ok", cfg,
            reason="text message sent; the video file could not be read",
            video="failed", details=details,
        )
    with handle:
        try:
            reply = post_multipart(
                _FILES_URL,
                data={"file_type": _VIDEO_FILE_TYPE, "file_name": name},
                files={"file": (name, handle, _VIDEO_MIME)},
                headers=_upload_headers(token),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to the text already sent
            details["video_upload"] = "request_error"
            return _result(
                "ok", cfg,
                reason=f"text message sent; video upload failed: {_exception_text(exc)}",
                video="failed", details=details,
            )
    status, payload = _decode(reply)
    code, message = _reply_code(payload)
    if status != 200 or code != 0:
        denied = status in (400, 403) or code in _PERMISSION_CODES
        details.update({
            "video_upload": "not_permitted" if denied else "failed",
            "http": status,
            "code": code,
        })
        outcome = "not permitted" if denied else "failed"
        return _result(
            "ok", cfg,
            reason=(
                f"text message sent; video upload {outcome} "
                f"({_describe(status, code, message)}); no retry"
            ),
            video="failed", details=details,
        )
    file_key = _nested_string(payload, "data", "file_key")
    if not file_key:
        details["video_upload"] = "missing_file_key"
        return _result(
            "ok", cfg,
            reason="text message sent; video upload returned no file_key",
            video="failed", details=details,
        )
    details["video_upload"] = "ok"

    media = {
        "receive_id": cfg.chat_id,
        "msg_type": "media",
        "content": json.dumps({"file_key": file_key}, ensure_ascii=False),
    }
    try:
        reply = post_json(_MESSAGES_URL, media, headers=_auth_headers(token), timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        details["video_message"] = "request_error"
        return _result(
            "ok", cfg,
            reason=f"text message sent; video message failed: {_exception_text(exc)}",
            video="failed", details=details,
        )
    status, payload = _decode(reply)
    code, message = _reply_code(payload)
    if status != 200 or code != 0:
        denied = status in (400, 403) or code in _PERMISSION_CODES
        details.update({
            "video_message": "not_permitted" if denied else "failed",
            "http": status,
            "code": code,
        })
        outcome = "not permitted" if denied else "failed"
        return _result(
            "ok", cfg,
            reason=(
                f"text message sent; video message {outcome} "
                f"({_describe(status, code, message)}); no retry"
            ),
            video="failed", details=details,
        )
    details["video_message"] = "ok"
    return _result("ok", cfg, video="sent", details=details)


def _send_webhook(
    cfg: FeishuConfig, text: str, video: Path | None, transport: object | None
) -> dict:
    """Send the text through a group-bot webhook; it cannot carry the video."""
    post_json, _ = _resolve_transport(transport)
    video_state = _initial_video_state(video)
    try:
        reply = post_json(
            cfg.webhook_url,
            {"msg_type": "text", "content": {"text": text}},
            headers=dict(_JSON_HEADERS),
            timeout=_timeout(cfg),
        )
    except Exception as exc:  # noqa: BLE001 - a transport error must not escape send()
        return _result(
            "failed", cfg,
            reason=f"webhook request failed: {_exception_text(exc)}",
            video=video_state, details={"webhook": "request_error"},
        )
    status, payload = _decode(reply)
    code, message = _reply_code(payload)
    if status != 200 or code not in (0, None):
        return _result(
            "failed", cfg,
            reason=f"webhook rejected the message ({_describe(status, code, message)})",
            video=video_state,
            details={"webhook": "failed", "http": status, "code": code},
        )
    note = ""
    if video is not None:
        note = "webhook mode cannot attach videos; the text summary was sent"
    return _result(
        "ok", cfg, reason=note, video=video_state,
        details={"webhook": "ok", "http": status},
    )
