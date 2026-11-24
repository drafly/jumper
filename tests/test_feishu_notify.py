"""`tools/postfetch/feishu.py` -- what reaches a chat, and what must not.

The notification stage is the one failure nobody finds by reading a log: a
message that went to the wrong group looks like success, an error reason with a
webhook URL in it leaks a token into the state file, and a video the bot is not
allowed to upload would, if treated as the whole failure, discard the summary
that was already delivered. These tests pin the properties that are invisible
afterwards:

1. the existing DSH bot and the group registered for this workspace are reused,
   and a missing configuration becomes `blocked` naming the missing keys;
2. token, message and webhook failures are `failed`, never raised;
3. a video upload Feishu denies leaves the text summary delivered, marks the
   video failed, and retries nothing;
4. no app secret, webhook URL or local path appears in a result, a reason or the
   preview queue file;
5. the Chinese summary maps every verdict and prints only a checkpoint basename.

No test touches the network and no test reads the real `~/.dsh`: each send gets a
fake transport and each discover gets a fabricated DSH directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SECRET = "test-app-secret-0123456789abcdef"
WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/webhook-token-0123456789"
CHAT_ID = "oc_2db3cf588fbdd5582a281850efe644b0"
TOKEN_OK = {"code": 0, "msg": "ok", "tenant_access_token": "t-tenant-token"}
MESSAGE_OK = {"code": 0, "msg": "success", "data": {"message_id": "om_1"}}


def _load() -> object:
    spec = importlib.util.spec_from_file_location(
        "feishu_notify_under_test", REPO / "tools" / "postfetch" / "feishu.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while decorating.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


feishu = _load()


# --------------------------------------------------------------------------
# fakes: no test may open a socket
# --------------------------------------------------------------------------


@dataclass
class FakeResponse:
    """Enough of a `requests.Response` for the JSON paths this module reads."""

    status_code: int = 200
    payload: dict | None = None

    def json(self) -> dict:
        return {} if self.payload is None else self.payload


class FakeTransport:
    """Records calls and replays scripted replies or exceptions, in order."""

    def __init__(self, *, posts: list | None = None, uploads: list | None = None) -> None:
        self._posts = list(posts or [])
        self._uploads = list(uploads or [])
        self.json_calls: list[dict] = []
        self.multipart_calls: list[dict] = []

    def post_json(self, url, payload, *, headers=None, timeout=10.0):
        self.json_calls.append({"url": url, "payload": payload,
                                "headers": dict(headers or {})})
        return self._next(self._posts, "post_json")

    def post_multipart(self, url, *, data, files, headers=None, timeout=10.0):
        self.multipart_calls.append({"url": url, "data": data,
                                     "headers": dict(headers or {})})
        return self._next(self._uploads, "post_multipart")

    @staticmethod
    def _next(queue: list, name: str):
        if not queue:
            raise AssertionError(f"unexpected {name} call")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class RequestsLikeTransport:
    """A fake with a single `post`, like the `requests` module itself."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def post(self, url, *, json=None, data=None, files=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "data": data})
        return self._replies.pop(0)


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _write_dsh(dsh: Path, workspace: Path) -> None:
    """A fabricated ~/.dsh with the same shape as the real Feishu integration."""
    integration = dsh / "integrations" / "dsh-feishu"
    integration.mkdir(parents=True)
    ref = "DSH_FEISHU_APP_SECRET_TEST"
    (integration / "config.json").write_text(json.dumps({
        "version": 1,
        "bots": [{"id": "bot_test", "appId": "cli_test", "secretRef": ref,
                  "domain": "feishu"}],
    }), "utf-8")
    (integration / "workspaces.json").write_text(json.dumps({
        "version": 1,
        "conversationWorkspaces": {"bot_test": {f"group:{CHAT_ID}": str(workspace)}},
    }), "utf-8")
    (dsh / ".credentials.yaml").write_text(
        f"version: 1\nrefs:\n  {ref}: {SECRET}\n", "utf-8"
    )


def _app_config(secret: str = SECRET) -> feishu.FeishuConfig:
    return feishu.FeishuConfig(
        mode="app", app_id="cli_test", app_secret=secret, chat_id=CHAT_ID,
        timeout_s=5.0,
    )


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_discover_reuses_the_existing_dsh_bot_and_group(tmp_path: Path) -> None:
    """A new bot must not be created when one is already connected."""
    workspace = tmp_path / "jumper-handoff"
    workspace.mkdir()
    dsh = tmp_path / "dsh"
    _write_dsh(dsh, workspace)

    cfg = feishu.discover(workspace, dsh_dir=dsh, env={})

    assert cfg.mode == "app"
    assert cfg.app_id == "cli_test"
    assert cfg.app_secret == SECRET
    assert cfg.chat_id == CHAT_ID
    assert cfg.webhook_url == ""
    # The config object is the thing that ends up in a traceback or a log line.
    assert SECRET not in repr(cfg)


def test_discover_defaults_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_dsh(tmp_path / "home" / ".dsh", workspace)

    cfg = feishu.discover(workspace, env={})

    assert cfg.mode == "app"
    assert cfg.chat_id == CHAT_ID


def test_explicit_and_environment_values_win_field_by_field(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    dsh = tmp_path / "dsh"
    _write_dsh(dsh, workspace)
    env = {
        "JUMPER_FEISHU_APP_ID": "cli_env",
        "JUMPER_FEISHU_APP_SECRET": "env-secret",
        "JUMPER_FEISHU_CHAT_ID": "oc_env",
    }

    from_env = feishu.discover(workspace, dsh_dir=dsh, env=env)
    assert (from_env.app_id, from_env.app_secret, from_env.chat_id) == (
        "cli_env", "env-secret", "oc_env",
    )

    explicit = feishu.discover(workspace, {"app_id": "cli_explicit"}, dsh_dir=dsh, env=env)
    assert explicit.app_id == "cli_explicit"
    assert explicit.chat_id == "oc_env"


def test_an_unmapped_workspace_yields_no_chat_and_a_blocked_send(tmp_path: Path) -> None:
    """The group is looked up by path; it is never hardcoded."""
    mapped = tmp_path / "mapped"
    mapped.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    dsh = tmp_path / "dsh"
    _write_dsh(dsh, mapped)

    cfg = feishu.discover(other, dsh_dir=dsh, env={})

    assert cfg.mode == "none"
    assert cfg.chat_id == ""
    # Empty scripted queues: a transport call here would raise.
    result = feishu.send(cfg, "摘要", transport=FakeTransport())
    assert result["status"] == "blocked"
    assert "chat_id" in result["reason"]


def test_no_configuration_at_all_names_what_is_missing(tmp_path: Path) -> None:
    cfg = feishu.discover(tmp_path / "repo", dsh_dir=tmp_path / "absent", env={})

    assert cfg.mode == "none"
    assert (cfg.app_id, cfg.app_secret, cfg.chat_id, cfg.webhook_url) == ("", "", "", "")

    result = feishu.send(cfg, "摘要")
    assert result["status"] == "blocked"
    assert result["mode"] == "none"
    for name in ("app_id", "app_secret", "chat_id"):
        assert name in result["reason"]
    assert "JUMPER_FEISHU_WEBHOOK" in result["reason"]


# --------------------------------------------------------------------------
# app mode: every failure is reported, not raised
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("connection reset by peer"),
        FakeResponse(200, {"code": 10003, "msg": "invalid app_secret"}),
        FakeResponse(403, {"code": 99991672, "msg": "no permission"}),
    ],
)
def test_app_mode_reports_a_failed_token_request(reply) -> None:
    transport = FakeTransport(posts=[reply])

    result = feishu.send(_app_config(), "摘要", transport=transport)

    assert result["status"] == "failed"
    assert result["mode"] == "app"
    assert result["video"] == "not_attempted"
    assert len(transport.json_calls) == 1
    assert SECRET not in _dump(result)


def test_app_mode_reports_a_failed_text_message() -> None:
    transport = FakeTransport(posts=[
        FakeResponse(200, TOKEN_OK),
        FakeResponse(400, {"code": 230002, "msg": "bot is not in the chat"}),
    ])

    result = feishu.send(_app_config(), "摘要", transport=transport)

    assert result["status"] == "failed"
    assert result["video"] == "not_attempted"
    assert "230002" in result["reason"]
    assert len(transport.json_calls) == 2
    assert SECRET not in _dump(result)


def test_app_mode_sends_the_text_summary() -> None:
    transport = FakeTransport(posts=[
        FakeResponse(200, TOKEN_OK),
        FakeResponse(200, MESSAGE_OK),
    ])

    result = feishu.send(_app_config(), "摘要正文", transport=transport)

    assert result["status"] == "ok"
    assert result["mode"] == "app"
    assert result["video"] == "not_attempted"
    assert result["details"]["token"] == "ok"
    assert result["details"]["text_message"] == "ok"
    sent = transport.json_calls[1]["payload"]
    assert sent["receive_id"] == CHAT_ID
    assert sent["msg_type"] == "text"
    assert json.loads(sent["content"])["text"] == "摘要正文"
    assert SECRET not in _dump(result)


def test_an_exception_carrying_the_secret_is_scrubbed() -> None:
    """A leaky transport error must not become a leaky result."""
    transport = FakeTransport(posts=[RuntimeError(f"auth failed for {SECRET}")])

    result = feishu.send(_app_config(), "摘要", transport=transport)

    assert result["status"] == "failed"
    assert SECRET not in _dump(result)
    assert "<redacted secret>" in result["reason"]


def test_a_requests_like_transport_object_is_accepted() -> None:
    transport = RequestsLikeTransport([
        FakeResponse(200, TOKEN_OK),
        FakeResponse(200, MESSAGE_OK),
    ])

    result = feishu.send(_app_config(), "摘要", transport=transport)

    assert result["status"] == "ok"
    assert transport.calls[0]["json"]["app_id"] == "cli_test"
    assert json.loads(transport.calls[1]["json"]["content"])["text"] == "摘要"


# --------------------------------------------------------------------------
# video: the summary outlives a rejected attachment
# --------------------------------------------------------------------------


def test_a_denied_video_keeps_the_text_and_marks_the_video_failed(tmp_path: Path) -> None:
    """A permission the bot does not have is the common case; retrying is not."""
    video = tmp_path / "out" / "eval.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not a real video")
    text = feishu.build_summary(
        "jumper.flat", "job-1", str(tmp_path / "run" / "model_19.pt"), "pass",
        [("摔倒次数", "0")], "ok",
    )
    transport = FakeTransport(
        posts=[FakeResponse(200, TOKEN_OK), FakeResponse(200, MESSAGE_OK)],
        uploads=[FakeResponse(403, {"code": 99991672, "msg": "no file permission"})],
    )

    result = feishu.send(_app_config(), text, video, transport)

    assert result["status"] == "ok"
    assert result["video"] == "failed"
    assert "not permitted" in result["reason"]
    assert "99991672" in result["reason"]
    assert len(transport.multipart_calls) == 1  # exactly one attempt, no retry
    sent_text = json.loads(transport.json_calls[1]["payload"]["content"])["text"]
    assert "Checkpoint：model_19.pt" in sent_text
    assert str(tmp_path) not in sent_text
    assert SECRET not in _dump(result)
    assert str(video) not in _dump(result)


def test_a_video_accepted_by_feishu_is_sent_as_media(tmp_path: Path) -> None:
    video = tmp_path / "eval.mp4"
    video.write_bytes(b"video bytes")
    transport = FakeTransport(
        posts=[
            FakeResponse(200, TOKEN_OK),
            FakeResponse(200, MESSAGE_OK),
            FakeResponse(200, {"code": 0, "data": {"message_id": "om_2"}}),
        ],
        uploads=[FakeResponse(200, {"code": 0, "data": {"file_key": "file_v2_xyz"}})],
    )

    result = feishu.send(_app_config(), "摘要", video, transport)

    assert result["status"] == "ok"
    assert result["video"] == "sent"
    assert transport.multipart_calls[0]["data"] == {
        "file_type": "mp4", "file_name": "eval.mp4",
    }
    media = transport.json_calls[2]["payload"]
    assert media["msg_type"] == "media"
    assert json.loads(media["content"]) == {"file_key": "file_v2_xyz"}


def test_the_multipart_upload_carries_no_json_content_type(tmp_path: Path) -> None:
    """Feishu answered 234001 to an otherwise valid upload sent as JSON.

    The shared auth headers include `Content-Type: application/json`; on a
    multipart request that header overrides the boundary `requests` would set,
    the server rejects the body, and the clip is dropped after the summary has
    already gone out -- reading as a permission problem. The upload headers must
    carry the token and nothing else.
    """
    video = tmp_path / "eval.mp4"
    video.write_bytes(b"video bytes")
    transport = FakeTransport(
        posts=[
            FakeResponse(200, TOKEN_OK),
            FakeResponse(200, MESSAGE_OK),
            FakeResponse(200, {"code": 0, "data": {"message_id": "om_3"}}),
        ],
        uploads=[FakeResponse(200, {"code": 0, "data": {"file_key": "file_v2_abc"}})],
    )

    result = feishu.send(_app_config(), "摘要", video, transport)

    assert result["video"] == "sent"
    upload_headers = transport.multipart_calls[0]["headers"]
    assert "Content-Type" not in upload_headers
    assert upload_headers.get("Authorization", "").startswith("Bearer ")


# --------------------------------------------------------------------------
# webhook mode: text only, and the URL stays inside the process
# --------------------------------------------------------------------------


def _webhook_config(tmp_path: Path) -> feishu.FeishuConfig:
    cfg = feishu.discover(
        tmp_path / "repo",
        dsh_dir=tmp_path / "absent",
        env={"JUMPER_FEISHU_WEBHOOK": WEBHOOK},
    )
    assert cfg.mode == "webhook"
    return cfg


def test_webhook_mode_sends_text_and_skips_the_video(tmp_path: Path) -> None:
    video = tmp_path / "eval.mp4"
    video.write_bytes(b"video bytes")
    transport = FakeTransport(posts=[FakeResponse(200, {"code": 0, "msg": "success"})])

    result = feishu.send(_webhook_config(tmp_path), "摘要", video, transport)

    assert result["status"] == "ok"
    assert result["video"] == "skipped"
    assert "cannot attach" in result["reason"]
    assert transport.json_calls[0]["url"] == WEBHOOK
    assert transport.json_calls[0]["payload"] == {
        "msg_type": "text", "content": {"text": "摘要"},
    }
    assert transport.multipart_calls == []
    dumped = _dump(result)
    assert WEBHOOK not in dumped
    assert "bot/v2/hook" not in dumped


def test_a_webhook_failure_never_echoes_the_url(tmp_path: Path) -> None:
    transport = FakeTransport(posts=[RuntimeError(f"cannot reach {WEBHOOK}")])

    result = feishu.send(_webhook_config(tmp_path), "摘要", transport=transport)

    assert result["status"] == "failed"
    dumped = _dump(result)
    assert WEBHOOK not in dumped
    assert "bot/v2/hook" not in dumped
    assert "<redacted webhook>" in result["reason"]

    # Any other URL keeps its host and loses its path, which may carry a token.
    proxy = FakeTransport(
        posts=[RuntimeError("proxy refused https://proxy.example.com/admin/token-1")]
    )
    other = feishu.send(_webhook_config(tmp_path), "摘要", transport=proxy)
    assert "token-1" not in _dump(other)
    assert "proxy.example.com/<redacted>" in other["reason"]


# --------------------------------------------------------------------------
# preview queue
# --------------------------------------------------------------------------


def test_preview_writes_a_queue_file_without_any_secret(tmp_path: Path) -> None:
    queued = feishu.preview(_app_config(), "待发送摘要", tmp_path / "pending")

    payload = json.loads(queued.read_text("utf-8"))
    assert payload["text"] == "待发送摘要"
    assert payload["mode"] == "app"
    assert payload["missing"] == []
    assert isinstance(payload["created_at"], str)
    assert SECRET not in queued.read_text("utf-8")

    webhook_path = feishu.preview(
        feishu.FeishuConfig(mode="webhook", webhook_url=WEBHOOK), "x", tmp_path / "pending"
    )
    assert WEBHOOK not in webhook_path.read_text("utf-8")


def test_preview_of_an_unconfigured_target_names_the_missing_keys(tmp_path: Path) -> None:
    queued = feishu.preview(
        feishu.FeishuConfig(mode="none"), "x", tmp_path / "pending"
    )

    assert json.loads(queued.read_text("utf-8"))["missing"] == [
        "app_id", "app_secret", "chat_id",
    ]


# --------------------------------------------------------------------------
# the Chinese summary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "label"),
    [
        ("pass", "达标"),
        ("failed", "未达标"),
        ("unsupported", "不支持"),
        ("error", "程序失败"),
    ],
)
def test_every_verdict_has_a_chinese_label(verdict: str, label: str) -> None:
    text = feishu.build_summary(
        "jumper.flat", "20261124-1", "model_1599.pt", verdict,
        [("摔倒次数", "0"), ("速度误差", "0.12 m/s")], "ok",
    )

    assert f"评估结果：{label}" in text
    assert "任务：jumper.flat" in text
    assert "Job：20261124-1" in text
    assert "Checkpoint：model_1599.pt" in text
    assert "- 摔倒次数：0" in text
    assert "- 速度误差：0.12 m/s" in text
    assert "录像：已录制" in text


def test_the_summary_keeps_local_paths_out_and_reports_the_failure() -> None:
    # Built from parts: the layout suite forbids a quoted absolute home path.
    machine_home = "/" + "Users/lcfc/project/jumper-handoff"
    reason = f"could not write {machine_home}/out/eval.mp4"
    text = feishu.build_summary(
        "jumper.flat", "job-1", "/tmp/work/model_19.pt", "fail", [], "failed",
        reason=reason,
    )

    assert "Checkpoint：model_19.pt" in text
    assert "/tmp/work" not in text
    assert "评估结果：未达标" in text
    assert "关键指标：无" in text
    assert "录像：录制失败" in text
    assert machine_home not in text
    assert "<path>" in text
