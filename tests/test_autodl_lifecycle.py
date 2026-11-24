"""The lifecycle's silent failure: shutting a box down with the only copy on it.

Every test here is about a moment that looks fine from the terminal -- a job that
"finished", a sync that "ran", a shutdown that "was sent" -- and the checks that
make those moments either true or refused. The network is never touched: the two
AutoDL APIs get a fake transport, and SSH is a recorded object.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "autodl_lifecycle.py"


def _load():
    spec = importlib.util.spec_from_file_location("autodl_lifecycle_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lifecycle = _load()


def _config(tmp_path: Path, *, backend: str = "manual", uuid: str = "pro-test",
            shutdown: str = "ssh") -> object:
    api = lifecycle.ApiConfig(
        backend=backend, instance_uuid=uuid, token_env="AUTODL_TEST_TOKEN",
        poll_s=0.01, boot_timeout_s=1.0,
    )
    ssh = SimpleNamespace(hostname="box.example", port=22, workdir="/root/w",
                          results="outputs/remote")
    return lifecycle.LifecycleConfig(
        api=api, ssh=ssh, results_dir=tmp_path / "results", log_dir=tmp_path / "logs",
        shutdown=shutdown,
    )


class Recorder:
    """A transport: records the call, returns what the test queued."""

    def __init__(self, replies=None):
        self.calls: list[dict] = []
        self._replies = list(replies or [])

    def __call__(self, url, body, headers, *, method="POST", timeout=20.0):
        self.calls.append({"url": url, "body": body, "headers": dict(headers), "method": method})
        if not self._replies:
            raise AssertionError(f"unexpected request {method} {url}")
        return self._replies.pop(0)


# --------------------------------------------------------------------------
# the two AutoDL APIs
# --------------------------------------------------------------------------


def test_pro_api_sends_the_documented_power_payloads(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-pro-token")
    config = _config(tmp_path, backend="pro")
    transport = Recorder([
        {"code": "Success", "data": "running"},
        {"code": "Success", "data": None},
        {"code": "Success", "data": None},
    ])
    assert lifecycle.api_status(config.api, transport) == "running"
    lifecycle.api_power_on(config.api, transport)
    lifecycle.api_power_off(config.api, transport)

    status_call, on_call, off_call = transport.calls
    assert "/dev/instance/pro/status?" in status_call["url"]
    assert "instance_uuid=pro-test" in status_call["url"]
    assert status_call["body"] is None
    assert status_call["method"] == "GET"
    assert on_call["url"].endswith("/dev/instance/pro/power_on")
    assert on_call["body"] == {"instance_uuid": "pro-test", "payload": "gpu"}
    assert off_call["url"].endswith("/dev/instance/pro/power_off")
    assert off_call["body"] == {"instance_uuid": "pro-test"}
    assert on_call["headers"]["Authorization"] == "secret-pro-token"


def test_console_api_lists_the_instance_and_uses_the_console_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-console-token")
    config = _config(tmp_path, backend="console")
    transport = Recorder([
        {"code": "Success", "data": {"list": [
            {"uuid": "other", "status": "running"},
            {"uuid": "pro-test", "status": "shutdown"},
        ]}},
        {"code": "Success", "data": None},
    ])
    assert lifecycle.api_status(config.api, transport) == "shutdown"
    lifecycle.api_power_on(config.api, transport)

    assert transport.calls[0]["url"].endswith("/api/v1/instance")
    assert transport.calls[1]["url"].endswith("/api/v1/instance/power_on")
    assert transport.calls[1]["body"] == {"instance_uuid": "pro-test"}
    assert transport.calls[0]["headers"]["AppVersion"]


def test_missing_token_names_the_variable_and_never_invents_one(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTODL_TEST_TOKEN", raising=False)
    config = _config(tmp_path, backend="pro")
    with pytest.raises(lifecycle.LifecycleError, match="AUTODL_TEST_TOKEN"):
        lifecycle.api_status(config.api, Recorder())


def test_an_error_envelope_becomes_a_reason_not_a_traceback(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-pro-token")
    config = _config(tmp_path, backend="pro")
    transport = Recorder([
        {"code": "AuthorizeFailed", "data": None, "msg": "登录超时，请重新登录"},
    ])
    with pytest.raises(lifecycle.LifecycleError, match="AuthorizeFailed"):
        lifecycle.api_status(config.api, transport)


# --------------------------------------------------------------------------
# discovery: which instance is this box?
# --------------------------------------------------------------------------


def test_match_instance_uses_the_ssh_port_then_the_host() -> None:
    rows = [
        {"uuid": "u1", "ssh_port": "24227", "proxy_host": "connect.westb.seetacloud.com"},
        {"uuid": "u2", "ssh_port": "24228", "proxy_host": "connect.westb.seetacloud.com"},
    ]
    ssh = SimpleNamespace(port=24227)
    assert [row["uuid"] for row in lifecycle.match_instance(rows, ssh)] == ["u1"]
    assert [row["uuid"] for row in lifecycle.match_instance(rows, ssh, "connect.westb")] == ["u1"]


def test_console_discovery_finds_the_box_by_ssh_port(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-console-token")
    config = _config(tmp_path, backend="console", uuid="")
    config.ssh.port = 24227
    transport = Recorder([
        {"code": "Success", "data": {"list": [
            {"uuid": "u-other", "ssh_port": "24228"},
            {"uuid": "u-box", "ssh_port": "24227", "instance_name": "jumper"},
        ]}},
    ])
    assert lifecycle.resolve_instance_uuid(config, transport) == "u-box"


def test_discovery_refuses_when_no_instance_answers_the_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-console-token")
    config = _config(tmp_path, backend="console", uuid="")
    transport = Recorder([{"code": "Success", "data": {"list": [{"uuid": "u", "ssh_port": "1"}]}}])
    with pytest.raises(lifecycle.LifecycleError, match="no instance answers ssh port 22"):
        lifecycle.resolve_instance_uuid(config, transport)


def test_up_dry_run_does_not_query_the_api(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, backend="console", uuid="")
    monkeypatch.setattr(lifecycle, "api_status",
                        lambda *a, **k: pytest.fail("dry-run must not call the API"))
    assert lifecycle.up(config, dry_run=True) == 0


# --------------------------------------------------------------------------
# the browser backend
# --------------------------------------------------------------------------


class FakeLocator:
    def __init__(self, text: str) -> None:
        self._text = text

    def inner_text(self) -> str:
        return self._text


class FakePage:
    """Enough of a Playwright page for the login and token paths."""

    def __init__(self, url: str, body: str, requests=(), start_at: str | None = None):
        self.url = url
        self._body = body
        self._requests = list(requests)
        self._start_at = start_at
        self._callbacks = []
        self.waits = 0

    def on(self, event: str, callback) -> None:
        assert event == "request"
        self._callbacks.append(callback)

    def goto(self, url: str, **kwargs) -> None:
        if self._start_at:
            self.url = self._start_at
        for request in self._requests:
            for callback in self._callbacks:
                callback(request)

    def locator(self, selector: str) -> FakeLocator:
        assert selector == "body"
        return FakeLocator(self._body)

    def wait_for_timeout(self, ms: int) -> None:
        self.waits += 1

    def close(self) -> None:
        pass


class FakeRequest:
    def __init__(self, url: str, headers: dict) -> None:
        self.url = url
        self.headers = headers


class FakeContext:
    def __init__(self, page: FakePage) -> None:
        self.pages = [page]
        self.saved: dict = {}

    def new_page(self) -> FakePage:
        return self.pages[0]

    def storage_state(self, path: str) -> None:
        self.saved["path"] = path
        Path(path).write_text("{}", "utf-8")

    def close(self) -> None:
        pass


def _fake_browser(monkeypatch, page: FakePage) -> None:
    from contextlib import contextmanager

    @contextmanager
    def fake_context(config, *, headless):
        yield FakeContext(page)

    monkeypatch.setattr(lifecycle, "_browser_context", fake_context)


def test_looks_logged_in_reads_the_console_not_just_the_url() -> None:
    assert lifecycle._looks_logged_in("https://www.autodl.com/console/instance/",
                                      "实例ID /名称 状态 开机")
    assert not lifecycle._looks_logged_in("https://www.autodl.com/login?url=/console",
                                         "实例ID /名称")
    assert not lifecycle._looks_logged_in("https://www.autodl.com/console/instance/",
                                          "登录注册 微信登录 短信登录")


def test_capture_token_takes_the_consoles_authorization_header(monkeypatch) -> None:
    page = FakePage(
        "https://www.autodl.com/console/instance/",
        "实例ID /名称",
        requests=[FakeRequest("https://www.autodl.com/api/v1/instance",
                              {"Authorization": "Bearer console-jwt"})],
    )
    assert lifecycle._capture_token_from_page(page, 30.0) == "Bearer console-jwt"


def test_capture_token_refuses_a_logged_out_console() -> None:
    page = FakePage("https://www.autodl.com/login?url=/console/instance/", "登录")
    with pytest.raises(lifecycle.LifecycleError, match="browser-auth"):
        lifecycle._capture_token_from_page(page, 30.0)


def test_browser_auth_waits_for_login_and_keeps_the_profile(tmp_path,
                                                            monkeypatch) -> None:
    page = FakePage("https://www.autodl.com/console/instance/", "实例ID /名称 开机")
    _fake_browser(monkeypatch, page)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(lifecycle, "browser_token", lambda config: "Bearer token")
    config = _config(tmp_path, backend="browser")
    config.api.browser_profile = tmp_path / "profile"
    config.api.browser_storage = tmp_path / "storage.json"
    assert lifecycle.browser_auth(config, wait_s=30) == 0
    assert (tmp_path / "storage.json").is_file()


def test_browser_auth_says_so_when_login_never_happens(tmp_path,
                                                      monkeypatch) -> None:
    page = FakePage("https://www.autodl.com/login?url=/console", "登录 微信登录")
    _fake_browser(monkeypatch, page)
    ticks = iter(range(10_000))
    monkeypatch.setattr(lifecycle.time, "time", lambda: next(ticks))
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)
    config = _config(tmp_path, backend="browser")
    config.api.browser_profile = tmp_path / "profile"
    with pytest.raises(lifecycle.LifecycleError, match="login was not completed"):
        lifecycle.browser_auth(config, wait_s=30)


def test_browser_auth_verifies_the_session_before_calling_it_done(tmp_path,
                                                                monkeypatch) -> None:
    page = FakePage("https://www.autodl.com/console/instance/", "实例ID /名称 开机")
    _fake_browser(monkeypatch, page)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)

    def rejected(config) -> str:
        raise lifecycle.LifecycleError("the console session is logged out")

    monkeypatch.setattr(lifecycle, "browser_token", rejected)
    config = _config(tmp_path, backend="browser")
    config.api.browser_profile = tmp_path / "profile"
    with pytest.raises(lifecycle.LifecycleError, match="not usable"):
        lifecycle.browser_auth(config, wait_s=30)


def test_the_browser_launch_removes_the_automation_fingerprint(monkeypatch) -> None:
    """AutoDL's slider is the check that notices it; the two switches matter."""
    source = Path(lifecycle.__file__).read_text("utf-8")
    assert "ignore_default_args" in source and "enable-automation" in source
    assert "AutomationControlled" in source
    assert "navigator, 'webdriver'" in source


def test_browser_token_is_used_as_the_api_credential() -> None:
    calls = {"n": 0}

    def provider() -> str:
        calls["n"] += 1
        return "Bearer captured-jwt"

    api = lifecycle.ApiConfig(backend="browser", token_provider=provider)
    assert lifecycle._api_headers(api)["Authorization"] == "Bearer captured-jwt"
    assert api.token() == "Bearer captured-jwt"
    assert calls["n"] == 1, "the captured token must be cached for the command"
    assert "captured-jwt" not in repr(api)


def test_browser_backend_configuration_round_trips(tmp_path) -> None:
    remote = tmp_path / "remote.toml"
    remote.write_text('hostname = "box.example"\nport = 24227\nworkdir = "/root/w"\n')
    config_file = tmp_path / "autodl.toml"
    config_file.write_text(
        """
[autodl]
api = "browser"
instance_uuid = ""
browser_profile = "/tmp/browser-profile"
browser_headless = true
browser_timeout_s = 45
""",
        "utf-8",
    )
    config = lifecycle.load_config(config_file, remote)
    assert config.api.backend == "browser"
    assert str(config.api.browser_profile) == "/tmp/browser-profile"
    assert config.api.browser_headless is True
    assert config.api.browser_timeout_s == 45.0


# --------------------------------------------------------------------------
# the compute market (read-only planning)
# --------------------------------------------------------------------------


def _machine(gpu: str, idle: int, price: int, machine_id: str = "m1") -> dict:
    return {"gpu_name": gpu, "gpu_idle_num": idle, "payg_price": price,
            "machine_id": machine_id, "machine_alias": "机", "region_name": "内蒙B区"}


def test_pick_machine_prefers_the_configured_type_then_idle_then_price() -> None:
    rows = [
        _machine("RTX 3080", 8, 930, "cheap"),
        _machine("RTX 3080 Ti", 1, 1140, "one"),
        _machine("RTX 3080 Ti", 8, 1140, "many"),
        _machine("RTX 3080 Ti", 4, 1200, "pricey"),
    ]
    best = lifecycle.pick_machine(rows, ("RTX 3080 Ti", "RTX 3080"), 1.5)
    assert best is not None and best["machine_id"] == "many"


def test_pick_machine_respects_the_price_ceiling_and_idle_floor() -> None:
    rows = [_machine("RTX 3080 Ti", 2, 2000, "expensive")]
    assert lifecycle.pick_machine(rows, ("RTX 3080 Ti",), 1.5) is None
    rows = [_machine("RTX 3080 Ti", 0, 1140, "full")]
    assert lifecycle.pick_machine(rows, ("RTX 3080 Ti",), 1.5) is None


def test_market_scan_uses_the_console_machine_list(tmp_path,
                                                   monkeypatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-console-token")
    config = _config(tmp_path, backend="console")
    transport = Recorder([{"code": "Success", "data": {"list": [
        _machine("RTX 3080 Ti", 8, 1140, "m1"),
    ]}}])
    rows = lifecycle.api_machines(config.api, transport=transport)
    assert rows[0]["machine_id"] == "m1"
    assert transport.calls[0]["url"].endswith("/user/machine/list")
    assert transport.calls[0]["body"]["gpu_type_name"]


def test_market_plan_says_what_image_is_missing(tmp_path,
                                                monkeypatch) -> None:
    config = _config(tmp_path, backend="console")
    monkeypatch.setattr(lifecycle, "api_machines",
                        lambda *a, **k: [_machine("RTX 3080 Ti", 8, 1140, "m1")])
    plan = lifecycle.market_plan(config)
    assert plan["best"]["machine_id"] == "m1"
    assert "no image configured" in plan["image_problem"]


def test_up_rent_new_dry_run_prints_the_market_plan(tmp_path, monkeypatch,
                                                    capsys) -> None:
    config = _config(tmp_path, backend="console")
    monkeypatch.setattr(lifecycle, "market_plan", lambda cfg: {
        "allowed": ["RTX 3080 Ti"], "max_price_cny": 1.5,
        "best": _machine("RTX 3080 Ti", 8, 1140, "m1"),
        "image_problem": "no image configured", "instance_name": "jumper-auto",
    })
    assert lifecycle.up(config, dry_run=True, rent_new=True) == 0
    printed = capsys.readouterr().out
    assert "would create instance" in printed
    assert "no image configured" in printed


# --------------------------------------------------------------------------
# the Pro API (unattended create; no 3080 Ti in the catalogue)
# --------------------------------------------------------------------------


def test_console_nocard_power_on_sends_non_gpu_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-console-token")
    config = _config(tmp_path, backend="console")
    transport = Recorder([{"code": "Success", "data": None}])
    lifecycle.api_power_on(config.api, transport, nocard=True)
    assert transport.calls[0]["body"] == {
        "instance_uuid": config.api.instance_uuid, "payload": "non_gpu"}


def test_pro_create_body_matches_the_documented_fields(tmp_path) -> None:
    config = _config(tmp_path, backend="pro")
    body = lifecycle.pro_create_body(config.api)
    assert body["gpu_spec_uuid"] == "v-32g-p"
    assert body["image_uuid"] == "base-image-l2t43iu6uk"
    assert body["cuda_v_from"] == 128
    assert body["expand_system_disk_by_gb"] == 50
    assert body["req_gpu_amount"] == 1
    assert body["start_command"]


def test_pro_create_and_snapshot_use_the_dev_endpoints(tmp_path,
                                                       monkeypatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-dev-token")
    config = _config(tmp_path, backend="pro")
    transport = Recorder([
        {"code": "Success", "data": "pro-12345"},
        {"code": "Success", "data": {
            "ssh_command": "ssh -p 34222 root@connect.example",
            "proxy_host": "connect.example", "ssh_port": 34222,
            "root_password": "pw", "payg_price": 1970,
            "snapshot_gpu_alias_name": "NVIDIA RTX PRO 6000",
        }},
    ])
    uuid = lifecycle.api_pro_create(config.api, transport)
    assert uuid == "pro-12345"
    snapshot = lifecycle.api_pro_snapshot(config.api, uuid, transport)
    assert snapshot["ssh_port"] == 34222
    assert transport.calls[0]["url"].endswith("/dev/instance/pro/create")
    assert "/dev/instance/pro/snapshot?" in transport.calls[1]["url"]
    assert "instance_uuid=pro-12345" in transport.calls[1]["url"]
    assert transport.calls[1]["method"] == "GET"


def test_pro_image_save_and_list_use_the_dev_endpoints(tmp_path,
                                                       monkeypatch) -> None:
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-dev-token")
    config = _config(tmp_path, backend="pro")
    transport = Recorder([
        {"code": "Success", "data": {"image_uuid": "image-abc"}},
        {"code": "Success", "data": {"list": [
            {"image_uuid": "image-abc", "name": "jumper-pro-ready",
             "status": "finished", "image_size": 2445271040},
        ]}},
    ])
    image_uuid = lifecycle.api_pro_image_save(
        config.api, "pro-x", "jumper-pro-ready", transport)
    assert image_uuid == "image-abc"
    assert transport.calls[0]["body"] == {
        "instance_uuid": "pro-x", "image_name": "jumper-pro-ready"}
    images = lifecycle.api_pro_images(config.api, transport)
    assert images[0]["status"] == "finished"
    assert transport.calls[1]["url"].endswith("/dev/instance/pro/image/private/list")


def test_pro_plan_says_what_the_catalogue_cannot_do(tmp_path) -> None:
    config = _config(tmp_path, backend="pro")
    plan = lifecycle.pro_plan(config)
    warnings = " ".join(plan["warnings"])
    assert "no RTX 3080 Ti" in warnings
    assert "torch 2.8" in warnings
    assert "system disk" in warnings


def test_pro_up_retries_create_across_specs_when_out_of_stock(tmp_path,
                                                             monkeypatch) -> None:
    """All Pro specs can answer 当前算力规格暂无库存; the next one is tried."""
    config = _config(tmp_path, backend="pro")
    config.api.instance_uuid = ""  # force the create path
    config.api.pro_gpu_spec = "v-32g-p"
    config.api.pro_gpu_specs = ("4090D",)
    calls: list[str] = []

    def fake_create(api, transport=None) -> str:
        calls.append(api.pro_gpu_spec)
        if len(calls) == 1:
            raise lifecycle.LifecycleError("当前算力规格暂无库存")
        return "pro-1"

    monkeypatch.setattr(lifecycle, "pro_find_instance", lambda *a, **k: "")
    monkeypatch.setattr(lifecycle, "api_pro_create", fake_create)
    monkeypatch.setattr(lifecycle, "_wait_status", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "api_pro_snapshot",
                        lambda *a, **k: {"ssh_command": "ssh -p 1 root@box",
                                         "root_password": "pw", "payg_price": 1680})
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)
    assert lifecycle.up(config, rent_new=True, retry_minutes=1) == 0
    assert calls == ["v-32g-p", "4090D"]


def test_pro_session_config_names_the_new_instance_not_the_old_one(tmp_path) -> None:
    config = _config(tmp_path, backend="pro")
    config.log_dir = tmp_path / "logs"
    config.log_dir.mkdir()
    snapshot = {"proxy_host": "connect.pro.example", "ssh_port": 34567}
    path = lifecycle._pro_session_config(config, snapshot, "/root/miniconda3/bin/python")
    text = path.read_text()
    assert 'hostname = "connect.pro.example"' in text
    assert "port = 34567" in text
    assert 'python = "/root/miniconda3/bin/python"' in text
    assert str(config.results_dir) in text
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_pro_session_dry_run_creates_nothing(tmp_path, monkeypatch, capsys) -> None:
    config = _config(tmp_path, backend="pro")
    monkeypatch.setattr(lifecycle, "pro_ensure_instance",
                        lambda *a, **k: pytest.fail("dry-run must not create"))
    monkeypatch.setattr(lifecycle, "api_pro_create",
                        lambda *a, **k: pytest.fail("dry-run must not create"))
    assert lifecycle.pro_session(
        config, task="jumper.flat", submit_args="", max_hours=0.5,
        post=False, retry_minutes=0, dry_run=True,
    ) == 0
    printed = capsys.readouterr().out
    assert "would create Pro instance" in printed


def test_pro_up_dry_run_prints_the_create_plan(tmp_path, monkeypatch, capsys) -> None:
    config = _config(tmp_path, backend="pro")
    assert lifecycle.up(config, dry_run=True, rent_new=True) == 0
    printed = capsys.readouterr().out
    assert "would create Pro instance" in printed
    assert "v-32g-p" in printed


# --------------------------------------------------------------------------
# the standard-instance fallback (hand-created, then adopted)
# --------------------------------------------------------------------------


def test_adopt_refuses_when_no_standard_instance_is_running(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, backend="browser")
    monkeypatch.setattr(lifecycle, "standard_instances", lambda cfg: [])
    with pytest.raises(lifecycle.LifecycleError, match="no running standard instance"):
        lifecycle.adopt_instance(config, task="jumper.flat", max_hours=0.5,
                                 shutdown=False)


def test_market_check_records_the_signature(tmp_path, monkeypatch, capsys) -> None:
    config = _config(tmp_path, backend="pro")
    config.log_dir = tmp_path / "logs"
    rows = [{"gpu": "RTX 3080",
             "machines": [{"machine_id": "3080a", "alias": "890机", "region": "",
                            "price": "¥0.93"}], "total": 1}]
    monkeypatch.setattr(lifecycle, "standard_available", lambda cfg, gpus: rows)
    assert lifecycle.market_check(config, gpus=("RTX 3080", "RTX 3080 Ti"),
                                  min_idle=1, notify=False) == 0
    state = json.loads((config.log_dir / "market-state.json").read_text())
    assert "3080a" in state["signature"]
    assert "RTX 3080" in capsys.readouterr().out


# --------------------------------------------------------------------------
# the sync gate
# --------------------------------------------------------------------------


def _fake_remote_module(jobs: dict, remote: object) -> object:
    class FakeModule:
        @staticmethod
        def job_ids(_remote, _cfg, limit=100):
            return list(jobs)

        @staticmethod
        def read_job(_remote, _cfg, job_id):
            return jobs[job_id]

        @staticmethod
        def Remote(_cfg):
            return remote

    return FakeModule()


def _terminal_job(job_id: str, status: str = "succeeded") -> dict:
    return {"job_id": job_id, "status": status,
            "checkpoints": [{"path": f"/root/autodl-tmp/jumper/runs/{job_id}/model_299.pt"}]}


def test_sync_gate_refuses_a_running_job_and_a_terminal_run_with_no_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = {"running1": _terminal_job("running1", "running"),
            "done1": _terminal_job("done1")}
    remote = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(lifecycle, "_remote_train", lambda: _fake_remote_module(jobs, remote))
    config = _config(tmp_path)

    assert lifecycle.running_jobs(config, remote) == ["running1"]
    problems = lifecycle.sync_problems(config, remote, verify_hash=False)
    assert len(problems) == 1 and "never fetched" in problems[0]


def test_sync_gate_passes_when_the_terminal_run_is_fetched_and_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = {"done1": _terminal_job("done1")}
    remote = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(lifecycle, "_remote_train", lambda: _fake_remote_module(jobs, remote))
    local = tmp_path / "results" / "done1"
    (local / "tree").mkdir(parents=True)
    model = local / "tree" / "model_299.pt"
    model.write_bytes(b"a checkpoint")
    (local / "FETCH.json").write_text(json.dumps({"verified": 8, "missing": []}), "utf-8")
    monkeypatch.setattr(lifecycle, "_remote_sha256", lambda *a, **k: lifecycle._sha256_file(model))

    assert lifecycle.sync_problems(_config(tmp_path), remote) == []


def test_sync_gate_catches_a_checkpoint_that_changed_after_the_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = {"done1": _terminal_job("done1")}
    remote = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(lifecycle, "_remote_train", lambda: _fake_remote_module(jobs, remote))
    local = tmp_path / "results" / "done1"
    (local / "tree").mkdir(parents=True)
    model = local / "tree" / "model_299.pt"
    model.write_bytes(b"a checkpoint")
    (local / "FETCH.json").write_text(json.dumps({"verified": 8, "missing": []}), "utf-8")
    monkeypatch.setattr(lifecycle, "_remote_sha256", lambda *a, **k: "0" * 64)

    problems = lifecycle.sync_problems(_config(tmp_path), remote)
    assert len(problems) == 1 and "differs from the box copy" in problems[0]


def test_a_failed_fetch_is_not_a_synced_run(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = {"done1": _terminal_job("done1")}
    remote = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(lifecycle, "_remote_train", lambda: _fake_remote_module(jobs, remote))
    local = tmp_path / "results" / "done1"
    local.mkdir(parents=True)
    (local / "FETCH.json").write_text(json.dumps({"verified": 0, "missing": ["x"]}), "utf-8")

    problems = lifecycle.sync_problems(_config(tmp_path), remote, verify_hash=False)
    assert problems and "did not verify" in problems[0]


# --------------------------------------------------------------------------
# down: refuse, or shut down over SSH
# --------------------------------------------------------------------------


def test_down_refuses_while_a_job_is_running(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(lifecycle, "_ssh_ready", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "running_jobs", lambda *a, **k: ["job1"])
    monkeypatch.setattr(lifecycle, "sync_problems", lambda *a, **k: [])
    monkeypatch.setattr(lifecycle, "_remote_train",
                        lambda: SimpleNamespace(Remote=lambda cfg: SimpleNamespace(run=None)))
    with pytest.raises(lifecycle.LifecycleError, match="still running"):
        lifecycle.down(config)


def test_down_dry_run_never_sends_a_shutdown(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    remote = SimpleNamespace(run=lambda script, **k: calls.append(script))
    monkeypatch.setattr(lifecycle, "_ssh_ready", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "running_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lifecycle, "sync_problems", lambda *a, **k: [])
    monkeypatch.setattr(lifecycle, "_remote_train",
                        lambda: SimpleNamespace(Remote=lambda cfg: remote))
    assert lifecycle.down(config, dry_run=True) == 0
    assert calls == []


def test_down_shuts_down_over_ssh_only_after_the_gate(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    remote = SimpleNamespace(run=lambda script, **k: calls.append(script))
    ready = iter([True, True, False, False])  # reachable, still up, then gone

    def fake_ready(*args, **kwargs):
        return next(ready, False)

    monkeypatch.setattr(lifecycle, "_ssh_ready", fake_ready)
    monkeypatch.setattr(lifecycle, "running_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lifecycle, "sync_problems", lambda *a, **k: [])
    monkeypatch.setattr(lifecycle, "_remote_train",
                        lambda: SimpleNamespace(Remote=lambda cfg: remote))
    assert lifecycle.down(config) == 0
    assert calls and "/usr/bin/shutdown" in calls[0]
    assert "sync" in calls[0]  # sync first: a dirty page could otherwise be lost


def test_up_retries_a_capacity_refusal_within_the_window(tmp_path,
                                                        monkeypatch) -> None:
    """AutoDL answers "机器空闲GPU不足"; retrying is the only action available."""
    monkeypatch.setenv("AUTODL_TEST_TOKEN", "secret-console-token")
    config = _config(tmp_path, backend="console")
    monkeypatch.setattr(lifecycle, "api_status", lambda cfg: "shutdown")
    calls = {"n": 0}

    def power_on(cfg, **kwargs) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise lifecycle.LifecycleError("BadRequest: 机器空闲GPU不足")

    monkeypatch.setattr(lifecycle, "api_power_on", power_on)
    monkeypatch.setattr(lifecycle, "_wait_status", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "_wait_ssh", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(lifecycle, "_log", lambda *a, **k: None)
    assert lifecycle.up(config, retry_minutes=1) == 0
    assert calls["n"] == 2


def test_manual_up_waits_for_ssh_and_calls_no_api(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(lifecycle, "_ssh_ready", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "api_status",
                        lambda *a, **k: pytest.fail("manual mode must not call the API"))
    assert lifecycle.up(config, manual=True) == 0


# --------------------------------------------------------------------------
# session planning
# --------------------------------------------------------------------------


def test_release_from_sync_reads_the_driver_line() -> None:
    text = "release 20261124-010203-96a28aa: 1253 files, 79M on the box\n"
    assert lifecycle._release_from_sync(text) == "20261124-010203-96a28aa"
    assert lifecycle._release_from_sync("nothing here") == ""


def test_session_dry_run_prints_the_chain_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(lifecycle, "_remote_call",
                        lambda *a, **k: pytest.fail("dry-run must not run the driver"))
    assert lifecycle.session(
        config, task="jumper.flat", release=None, submit_args="", max_hours=1.0,
        sync=True, post=False, dry_run=True,
    ) == 0
    printed = capsys.readouterr().out
    assert "sync" in printed and "submit" in printed
    assert "leave the instance running" in printed
    assert lifecycle.session(
        config, task="jumper.flat", release=None, submit_args="", max_hours=1.0,
        sync=True, post=False, dry_run=True, auto_shutdown=True,
    ) == 0
    assert "shut down" in capsys.readouterr().out
