"""Start the AutoDL instance when training needs it, shut it down after the results are safe.

The lifecycle this tool owns is:

    up    -> the instance is running and reachable
    train -> submit a job through tools/remote_train.py (optional: sync first)
    wait  -> the job reaches a terminal status, not "looks finished"
    sync  -> every run on the box is fetched locally and its SHA-256 verified
    down  -> the instance is powered off

Each step is printed and appended to an audit log; `--dry-run` prints the plan
without touching anything. The shutdown step refuses unless the box has no
running job **and** every terminal run's evidence is on this Mac, because a
machine that powers off between the training and the download is how a run's
only copy is lost.

Three ways to start an instance, because AutoDL offers three:

- `api = "pro"`: the official developer API (`https://api.autodl.com`), which is
  documented for **container instances Pro** only and needs a Developer Token
  (console -> account -> settings -> developer token). This is the stable path.
- `api = "console"`: the web console's own JSON API, which also works for a
  standard container instance but authenticates with a browser JWT that expires.
  It is not an official interface; a console redesign can break it.
- `api = "manual"`: `up` waits for a reachable box and tells the operator to
  start it. Everything after that (training, sync gate, shutdown) still runs.

Shutdown itself is the AutoDL-documented in-container `shutdown` command
(https://www.autodl.com/docs/save_money/, "场景三"): after training finishes and
the evidence is fetched, this tool SSHes in and runs `/usr/bin/shutdown`. No API
call is needed to stop paying for the instance, which is why a standard instance
still gets automatic shutdown even though it cannot be started by API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTE_TRAIN = REPO / "tools" / "remote_train.py"
DEFAULT_REMOTE_CONFIG = REPO / ".remote.toml"
DEFAULT_CONFIG = Path.home() / ".config" / "jumper" / "autodl.toml"

PRO_BASE = "https://api.autodl.com"
CONSOLE_BASE = "https://www.autodl.com"
DEFAULT_APP_VERSION = "v6.32.0"
CONSOLE_URL = "https://www.autodl.com/console/instance/"
#: A console URL or body that still says login; and the labels that only the
#: instance list shows. Both are the same markers the console uses in Chinese.
CONSOLE_LOGIN_MARKERS = ("/login", "/register")
CONSOLE_PAGE_MARKERS = ("实例ID /名称", "查看详情", "开机", "关机")
DEFAULT_BROWSER_PROFILE = Path.home() / ".config" / "jumper" / "autodl-browser-profile"
DEFAULT_BROWSER_STORAGE = Path.home() / ".config" / "jumper" / "autodl-storage.json"


class LifecycleError(RuntimeError):
    """A step could not be completed; the message is what a human needs."""


def _clean_token(value: str) -> str:
    """Trim whitespace, a leading `%s `, and wrapping quotes from a pasted token."""
    text = value.strip()
    if text.startswith("%s"):
        text = text[2:].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ApiConfig:
    """Where the power commands go, and where the credential comes from.

    The token itself is not stored here: `token_env` names an environment
    variable and `token_file` names an ignored file, so this object can be
    logged or printed without leaking the credential.
    """

    backend: str = "manual"  # pro | console | manual
    instance_uuid: str = ""
    token_env: str = "AUTODL_TOKEN"
    token_file: Path | None = None
    app_version: str = DEFAULT_APP_VERSION
    poll_s: float = 10.0
    boot_timeout_s: float = 900.0
    browser_profile: Path | None = None
    browser_storage: Path | None = None
    browser_channel: str = "chrome"
    browser_headless: bool = True
    browser_timeout_s: float = 120.0
    #: What a replacement instance may be, when waiting for the old host is not
    #: the plan. Used by `market` and the `--rent-new` dry-run only.
    gpu_types: tuple[str, ...] = ("RTX 3080 Ti",)
    max_price_cny: float = 1.5
    #: Container-instance-Pro settings. The official API has no 3080 Ti; the
    #: available specs are listed in the docs' appendix (3090-48G, 4080S-32G,
    #: 4090-48G, 4090D, PRO6000-96G, 5090-32G, H800).
    pro_gpu_spec: str = "v-32g-p"
    #: Specs tried in order when the first has no stock ("当前算力规格暂无库存").
    pro_gpu_specs: tuple[str, ...] = ()
    pro_cuda_from: int = 128
    pro_image_uuid: str = "base-image-l2t43iu6uk"
    pro_data_center: tuple[str, ...] = ()
    pro_instance_name: str = "jumper-pro"
    pro_system_disk_gb: int = 50
    image_type: str = ""
    image_url: str = ""
    community_uuid: str = ""
    private_image_uuid: str = ""
    private_image_url: str = ""
    instance_name: str = "jumper-auto"
    #: Set by the browser backend: returns a JWT captured from the console's own
    #: authenticated XHRs. Kept out of `repr` because it is a credential source.
    token_provider: Callable[[], str] | None = field(default=None, repr=False, compare=False)
    _cached_token: str = field(default="", repr=False, compare=False)

    def token(self) -> str:
        """The API credential, in documented order; browser tokens are cached.

        Environment first, then the token file, then -- for the `browser`
        backend -- a token captured from a logged-in console. The cache means one
        command does not open two browsers. A pasted value sometimes carries the
        `%s ` of the shell instruction that wrote it; that is stripped rather
        than reported as an authentication failure.
        """
        value = os.environ.get(self.token_env, "").strip()
        if value:
            return _clean_token(value)
        if self.token_file and self.token_file.is_file():
            return _clean_token(self.token_file.read_text("utf-8"))
        if self._cached_token:
            return self._cached_token
        if self.token_provider is not None:
            self._cached_token = _clean_token(self.token_provider() or "")
            return self._cached_token
        return ""


@dataclass
class LifecycleConfig:
    api: ApiConfig
    ssh: object  # remote_train.Config, loaded by path to avoid a package
    results_dir: Path
    log_dir: Path
    shutdown: str = "ssh"  # ssh | api
    #: Low-end GPUs are hard to rent again, so a finished session leaves the
    #: instance running by default; only `down` (or --auto-shutdown) stops it.
    auto_shutdown: bool = False
    fetch_all: bool = False
    post_fetch: bool = False
    source: Path | None = None


def _remote_train():
    """Load `tools/remote_train.py` by path; the driver stays stdlib-only."""
    name = "autodl_lifecycle_remote_train"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REMOTE_TRAIN)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise LifecycleError(f"cannot load {REMOTE_TRAIN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_config(config_path: str | Path | None = None,
                remote_config: str | Path | None = None) -> LifecycleConfig:
    """Read the lifecycle file; SSH settings stay in the training config.

    A missing lifecycle file is not an error: the default is `manual` start and
    shutdown by SSH, which needs no token at all. That is deliberate -- the
    safety-relevant half (sync gate + shutdown) works before any credential is
    configured.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.9/3.10: the driver's own fallback
        import tomli as tomllib  # type: ignore[no-redef]

    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    env_path = os.environ.get("JUMPER_AUTODL_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates += [REPO / ".autodl.toml", DEFAULT_CONFIG]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None and config_path:
        raise FileNotFoundError(f"lifecycle config not found: {config_path}")
    document: dict = {}
    if source is not None:
        with source.open("rb") as handle:
            document = tomllib.load(handle)

    autodl = document.get("autodl", {})
    if not isinstance(autodl, dict):
        raise TypeError("[autodl] must be a table")
    local = document.get("local", {})
    if not isinstance(local, dict):
        raise TypeError("[local] must be a table")

    remote_path = Path(remote_config).expanduser() if remote_config else DEFAULT_REMOTE_CONFIG
    remote_module = _remote_train()
    ssh = remote_module.load_config(str(remote_path))

    backend = str(autodl.get("api", "manual")).lower()
    if backend not in {"pro", "console", "browser", "manual"}:
        raise LifecycleError(
            f"autodl.api must be one of pro/console/browser/manual, not {backend!r}"
        )
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (REPO / path).resolve()

    token_file = autodl.get("token_file")
    token_path = _resolve(str(token_file)) if token_file else None
    if token_path and token_path.is_file():
        mode = token_path.stat().st_mode & 0o777
        if mode & 0o077:
            print(
                f"warning: {token_path} is readable by others (mode {mode:o}); "
                "chmod 600 it",
                file=sys.stderr,
            )
    api = ApiConfig(
        backend=backend,
        instance_uuid=str(autodl.get("instance_uuid", "")).strip(),
        token_env=str(autodl.get("token_env", "AUTODL_TOKEN")),
        token_file=token_path,
        app_version=str(autodl.get("app_version", DEFAULT_APP_VERSION)),
        poll_s=float(autodl.get("poll_seconds", 10.0)),
        boot_timeout_s=float(autodl.get("boot_timeout_s", 900.0)),
        browser_profile=_resolve(str(autodl.get("browser_profile", DEFAULT_BROWSER_PROFILE))),
        browser_storage=_resolve(str(autodl.get("browser_storage", DEFAULT_BROWSER_STORAGE))),
        browser_channel=str(autodl.get("browser_channel", "chrome")),
        browser_headless=bool(autodl.get("browser_headless", True)),
        browser_timeout_s=float(autodl.get("browser_timeout_s", 120.0)),
        gpu_types=tuple(str(value) for value in autodl.get("gpu_types", ["RTX 3080 Ti"])),
        max_price_cny=float(autodl.get("max_price_cny", 1.5)),
        pro_gpu_spec=str(autodl.get("pro_gpu_spec", "v-32g-p")),
        pro_gpu_specs=tuple(str(v) for v in autodl.get("pro_gpu_specs", [])),
        pro_cuda_from=int(autodl.get("pro_cuda_from", 128)),
        pro_image_uuid=str(autodl.get("pro_image_uuid", "base-image-l2t43iu6uk")),
        pro_data_center=tuple(str(v) for v in autodl.get("pro_data_center", [])),
        pro_instance_name=str(autodl.get("pro_instance_name", "jumper-pro")),
        pro_system_disk_gb=int(autodl.get("pro_system_disk_gb", 50)),
        image_type=str(autodl.get("image_type", "")),
        image_url=str(autodl.get("image_url", "")),
        community_uuid=str(autodl.get("community_uuid", "")),
        private_image_uuid=str(autodl.get("private_image_uuid", "")),
        private_image_url=str(autodl.get("private_image_url", "")),
        instance_name=str(autodl.get("instance_name", "jumper-auto")),
    )
    shutdown = str(autodl.get("shutdown", "ssh")).lower()
    if shutdown not in {"ssh", "api"}:
        raise LifecycleError("autodl.shutdown must be 'ssh' or 'api'")

    results = Path(local.get("results", ssh.results))
    log_dir = Path(local.get("log", "outputs/autodl"))
    if not log_dir.is_absolute():
        log_dir = (REPO / log_dir).resolve()
    if not results.is_absolute():
        results = (REPO / results).resolve()

    return LifecycleConfig(
        api=api,
        ssh=ssh,
        results_dir=results,
        log_dir=log_dir,
        shutdown=shutdown,
        auto_shutdown=bool(autodl.get("auto_shutdown", False)),
        fetch_all=bool(autodl.get("fetch_all", False)),
        post_fetch=bool(autodl.get("post_fetch", False)),
        source=source,
    )


# ──────────────────────────────────────────────────────────────────────────────
# HTTP transport and the two APIs
# ──────────────────────────────────────────────────────────────────────────────


def _request(url: str, body: dict | None, headers: dict, *, method: str = "POST",
             timeout: float = 20.0) -> dict:
    """One JSON request; returns the parsed body. Secrets never enter the error."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise LifecycleError(f"{method} {url} answered HTTP {exc.code}") from None
    except OSError as exc:
        raise LifecycleError(f"cannot reach {url}: {type(exc).__name__}") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise LifecycleError(f"{method} {url} returned non-JSON") from None
    _expect_success(payload, f"{method} {url}")
    return payload


def _expect_success(payload: dict, where: str) -> dict:
    """`code == "Success"` or a readable error; both APIs use this envelope."""
    if payload.get("code") != "Success":
        raise LifecycleError(
            f"{where} returned {payload.get('code')!r}: {str(payload.get('msg', ''))[:200]}"
        )
    return payload


def _api_headers(config: ApiConfig) -> dict:
    token = config.token()
    if not token:
        raise LifecycleError(
            f"no AutoDL credential: set ${config.token_env}, autodl.token_file, or "
            "-- for autodl.api = \"browser\" -- run `browser-auth` once to log in"
        )
    headers = {"Authorization": token, "Content-Type": "application/json;charset=utf-8"}
    if config.backend in {"console", "browser"}:
        headers["AppVersion"] = config.app_version
    return headers


def _pro_get_url(path: str, **params: object) -> str:
    """The Pro API's GET endpoints take query parameters, not a JSON body."""
    return _pro_url(path) + "?" + urllib.parse.urlencode({k: str(v) for k, v in params.items()})


def _pro_url(path: str) -> str:
    return f"{PRO_BASE}/api/v1/dev/instance/pro/{path}"


def _console_url(path: str) -> str:
    return f"{CONSOLE_BASE}/api/v1/{path}"


def api_status(config: ApiConfig, transport=_request) -> str:
    """The instance's status: running / shutdown / creating / abnormal / unknown."""
    if config.backend == "pro":
        payload = _expect_success(transport(
            _pro_get_url("status", instance_uuid=config.instance_uuid), None,
            _api_headers(config), method="GET",
        ), "instance status")
        return str(payload.get("data") or "unknown").lower()
    if config.backend in {"console", "browser"}:
        payload = _expect_success(transport(
            _console_url("instance"),
            {
                "page_index": 1, "page_size": 100, "status": [], "charge_type": [],
                "sub_name": "", "unbind_sub_user": False,
            },
            _api_headers(config),
        ), "instance list")
        for item in payload.get("data", {}).get("list", []):
            if item.get("uuid") == config.instance_uuid:
                return str(item.get("status") or "unknown").lower()
        return "unknown"
    raise LifecycleError("status is not available in manual mode")


def api_power_on(config: ApiConfig, transport=_request, nocard: bool = False) -> None:
    if config.backend == "pro":
        if nocard:
            raise LifecycleError("the Pro API does not support no-card boot")
        _expect_success(transport(
            _pro_url("power_on"),
            {"instance_uuid": config.instance_uuid, "payload": "gpu"},
            _api_headers(config),
        ), "power on")
        return
    if config.backend in {"console", "browser"}:
        body = {"instance_uuid": config.instance_uuid}
        if nocard:
            # the console's 无卡模式开机: 0.5 CPU / 2GB / no GPU, ￥0.10/h
            body["payload"] = "non_gpu"
        _expect_success(transport(_console_url("instance/power_on"), body,
                                  _api_headers(config)), "power on")
        return
    raise LifecycleError("manual mode cannot power the instance on")


def api_power_off(config: ApiConfig, transport=_request) -> None:
    if config.backend == "pro":
        _expect_success(transport(_pro_url("power_off"),
                                  {"instance_uuid": config.instance_uuid},
                                  _api_headers(config)), "power off")
        return
    if config.backend in {"console", "browser"}:
        _expect_success(transport(_console_url("instance/power_off"),
                                  {"instance_uuid": config.instance_uuid},
                                  _api_headers(config)), "power off")
        return
    raise LifecycleError("manual mode cannot power the instance off")


def pro_create_body(config: ApiConfig) -> dict:
    """The documented create body; every field is required except where noted."""
    return {
        "data_center_list": list(config.pro_data_center),
        "req_gpu_amount": 1,
        "expand_system_disk_by_gb": int(config.pro_system_disk_gb),
        "gpu_spec_uuid": config.pro_gpu_spec,
        "image_uuid": config.pro_image_uuid,
        "cuda_v_from": int(config.pro_cuda_from),
        "instance_name": config.pro_instance_name,
        "start_command": "sleep infinity",
    }


def api_pro_create(config: ApiConfig, transport=_request) -> str:
    payload = _expect_success(
        transport(_pro_url("create"), pro_create_body(config), _api_headers(config)),
        "instance create",
    )
    uuid = str(payload.get("data") or "")
    if not uuid:
        raise LifecycleError("instance create returned no instance id")
    return uuid


def api_pro_list(config: ApiConfig, transport=_request) -> list[dict]:
    payload = _expect_success(
        transport(_pro_url("list"), {"page_index": 1, "page_size": 100},
                  _api_headers(config)),
        "instance list",
    )
    data = payload.get("data", {})
    return list(data.get("list", [])) if isinstance(data, dict) else []


def api_pro_snapshot(config: ApiConfig, instance_uuid: str, transport=_request) -> dict:
    payload = _expect_success(
        transport(_pro_get_url("snapshot", instance_uuid=instance_uuid), None,
                  _api_headers(config), method="GET"),
        "instance snapshot",
    )
    data = payload.get("data", {})
    return data if isinstance(data, dict) else {}


def api_pro_image_save(config: ApiConfig, instance_uuid: str, image_name: str,
                       transport=_request) -> str:
    """Snapshot the instance's system disk; returns the new private image uuid."""
    payload = _expect_success(
        transport(_pro_url("image/save"),
                  {"instance_uuid": instance_uuid, "image_name": image_name},
                  _api_headers(config)),
        "image save",
    )
    data = payload.get("data", {})
    image_uuid = str(data.get("image_uuid") or "") if isinstance(data, dict) else ""
    if not image_uuid:
        raise LifecycleError("image save returned no image uuid")
    return image_uuid


def api_pro_images(config: ApiConfig, transport=_request) -> list[dict]:
    """The account's private Pro images, with their save status."""
    payload = _expect_success(
        transport(_pro_url("image/private/list"),
                  {"page_index": 1, "page_size": 100}, _api_headers(config)),
        "image list",
    )
    data = payload.get("data", {})
    return list(data.get("list", [])) if isinstance(data, dict) else []


def api_pro_release(config: ApiConfig, instance_uuid: str, transport=_request) -> None:
    _expect_success(
        transport(_pro_url("release"), {"instance_uuid": instance_uuid},
                  _api_headers(config)),
        "instance release",
    )


def pro_plan(config: LifecycleConfig) -> dict:
    """What the Pro API would create; read-only, no request is sent."""
    return {
        "body": pro_create_body(config.api),
        "spec": config.api.pro_gpu_spec,
        "image": config.api.pro_image_uuid,
        "data_center": list(config.api.pro_data_center) or ["(auto)"],
        "disk_gb": config.api.pro_system_disk_gb,
        "warnings": [
            ("the Pro API has no RTX 3080 Ti; specs are 3090-48G/4080S-32G/"
            "4090-48G/4090D/PRO6000-96G/5090-32G/H800"),
            ("the Pro docs list public images only up to PyTorch 2.0/CUDA 11.8; this "
            "config uses the standard catalogue's PyTorch 2.8/CUDA 12.8 base image "
            "and the first bootstrap installs torch 2.8 if the image lacks it"),
            ("Pro instances have a system disk but no separate data disk; the "
            "workdir must live under /root and the system disk is expanded here"),
        ],
    }


def format_pro_plan(plan: dict) -> str:
    body = plan["body"]
    return (
        f"would create Pro instance {body['instance_name']!r} "
        f"spec={plan['spec']} image={plan['image']} "
        f"regions={plan['data_center']} disk={plan['disk_gb']}GB "
        f"cuda_from={body['cuda_v_from']}. Warnings: " + " | ".join(plan["warnings"])
    )


def api_list(config: ApiConfig, transport=_request) -> list[dict]:
    """Every instance this account owns, for discovery and status."""
    if config.backend == "pro":
        payload = _expect_success(transport(
            _pro_url("list"), {"page_index": 1, "page_size": 100},
            _api_headers(config),
        ), "instance list")
        return list(payload.get("data", {}).get("list", []))
    if config.backend in {"console", "browser"}:
        payload = _expect_success(transport(
            _console_url("instance"),
            {
                "page_index": 1, "page_size": 100, "status": [], "charge_type": [],
                "sub_name": "", "unbind_sub_user": False,
            },
            _api_headers(config),
        ), "instance list")
        return list(payload.get("data", {}).get("list", []))
    raise LifecycleError("instance discovery needs an API; manual mode has none")


def price_cny(row: dict) -> float:
    """`payg_price` is in 毫元/hour; this is the number a person compares."""
    try:
        return float(row.get("payg_price", 0)) / 1000.0
    except (TypeError, ValueError):
        return float("inf")


def api_machines(config: ApiConfig, gpu_types: tuple[str, ...] = (),
                 transport=_request) -> list[dict]:
    """The console's compute-market list: machines and their idle GPUs."""
    body = {
        "charge_type": "payg", "region_sign": "",
        "gpu_type_name": list(gpu_types or config.gpu_types),
        "machine_tag_name": [], "gpu_idle_num": 1, "mount_net_disk": False,
        "instance_disk_size_order": "", "date_range": "", "date_from": "",
        "date_to": "", "page_index": 1, "page_size": 100, "pay_price_order": "",
        "gpu_idle_type": "", "default_order": True, "region_sign_list": [],
        "cpu_arch": [], "chip_corp": [], "machine_id": "",
    }
    payload = _expect_success(
        transport(_console_url("user/machine/list"), body, _api_headers(config)),
        "machine list",
    )
    data = payload.get("data", {})
    return list(data.get("list", [])) if isinstance(data, dict) else []


def pick_machine(rows: list[dict], allowed: tuple[str, ...], max_price_cny: float,
                 min_idle: int = 1) -> dict | None:
    """The machine a replacement instance should be created on.

    Order: the allowed GPU type in the configured order, then most idle cards,
    then cheapest, then a stable name. "Most idle" first keeps a fleet-ish pool
    from all landing on the same single card.
    """
    priority = {name: index for index, name in enumerate(allowed)}
    candidates = [
        row for row in rows
        if row.get("gpu_name") in priority
        and int(row.get("gpu_idle_num") or 0) >= min_idle
        and price_cny(row) <= max_price_cny
    ]
    candidates.sort(key=lambda row: (
        priority.get(str(row.get("gpu_name")), len(priority)),
        -int(row.get("gpu_idle_num") or 0),
        price_cny(row),
        str(row.get("region_name", "")),
        str(row.get("machine_alias", "")),
    ))
    return candidates[0] if candidates else None


def image_problem(config: LifecycleConfig) -> str:
    """Why a replacement instance cannot be created yet; empty when it can."""
    image_type = config.api.image_type
    if image_type == "base" and not config.api.image_url:
        return "autodl.image_type = \"base\" needs autodl.image_url"
    if image_type == "community" and not config.api.community_uuid:
        return "autodl.image_type = \"community\" needs autodl.community_uuid"
    if image_type == "private" and not config.api.private_image_uuid:
        return "autodl.image_type = \"private\" needs autodl.private_image_uuid"
    if image_type not in {"base", "community", "private"}:
        return (
            "no image configured: set autodl.image_type to base/community/private "
            "plus the matching URL or UUID"
        )
    return ""


def market_plan(config: LifecycleConfig, *, gpu_types: tuple[str, ...] = (),
                max_price_cny: float | None = None, min_idle: int = 1) -> dict:
    """Scan the market and choose a machine, without creating anything."""
    allowed = gpu_types or config.api.gpu_types
    cap = config.api.max_price_cny if max_price_cny is None else max_price_cny
    rows = api_machines(config.api, allowed)
    return {
        "allowed": list(allowed),
        "max_price_cny": cap,
        "machines": rows,
        "best": pick_machine(rows, allowed, cap, min_idle),
        "image_problem": image_problem(config),
        "instance_name": config.api.instance_name,
    }


def format_market(rows: list[dict], limit: int = 20) -> str:
    if not rows:
        return "no machines returned for those GPU types"
    lines = [f"{'gpu':<16}{'region':<12}{'idle':>5}{'¥/h':>7}  machine    alias"]
    for row in rows[:limit]:
        lines.append(
            f"{row.get('gpu_name', '')!s:<16}{row.get('region_name', '')!s:<12}"
            f"{int(row.get('gpu_idle_num') or 0):>5}{price_cny(row):>7.2f}  "
            f"{row.get('machine_id', '')!s:<10} {row.get('machine_alias', '')}"
        )
    return "\n".join(lines)


def format_market_plan(plan: dict) -> str:
    best = plan.get("best")
    if best is None:
        return (
            f"no free card for {plan['allowed']} under ¥{plan['max_price_cny']:.2f}/h right now"
        )
    image = plan.get("image_problem") or "configured"
    return (
        f"would create instance {plan['instance_name']!r} on {best.get('gpu_name')} "
        f"{best.get('region_name')} {best.get('machine_alias')} "
        f"(machine {best.get('machine_id')}, {best.get('gpu_idle_num')} idle, "
        f"¥{price_cny(best):.2f}/h); image: {image}"
    )


def pro_find_instance(config: LifecycleConfig, transport=_request) -> str:
    """The configured Pro uuid, or the one named like this project."""
    if config.api.instance_uuid:
        return config.api.instance_uuid
    rows = api_pro_list(config.api, transport)
    named = [row for row in rows if row.get("name") == config.api.pro_instance_name]
    if len(named) == 1:
        return str(named[0].get("uuid", ""))
    if len(rows) == 1:
        return str(rows[0].get("uuid", ""))
    return ""


def match_instance(rows: list[dict], ssh, hostname: str = "") -> list[dict]:
    """Rows that look like the box in the training config.

    The SSH port is the reliable key: AutoDL gives every instance its own port,
    and the proxy hostname repeats across instances in a region. The hostname is
    a tie-breaker, not the primary match.
    """
    port = str(getattr(ssh, "port", ""))
    by_port = [row for row in rows if str(row.get("ssh_port", "")) == port]
    if hostname:
        by_host = [row for row in by_port
                   if hostname in str(row.get("proxy_host", ""))]
        if by_host:
            return by_host
    return by_port


def resolve_instance_uuid(config: LifecycleConfig, transport=_request) -> str:
    """The configured uuid, or the one instance whose SSH port matches the box."""
    if config.api.instance_uuid:
        return config.api.instance_uuid
    if config.api.backend == "manual":
        raise LifecycleError("manual mode has no instance uuid; set autodl.instance_uuid")
    rows = api_list(config.api, transport)
    matches = match_instance(rows, config.ssh, getattr(config.ssh, "hostname", ""))
    if len(matches) == 1:
        uuid = str(matches[0].get("uuid", ""))
        print(f"matched instance {uuid} ({matches[0].get('instance_name', '')}) by ssh port "
              f"{config.ssh.port}")
        return uuid
    if not matches:
        ports = sorted({str(row.get("ssh_port", "?")) for row in rows})
        raise LifecycleError(
            f"no instance answers ssh port {config.ssh.port} (account ports: {ports}); "
            "run the `discover` command or set autodl.instance_uuid"
        )
    listing = ", ".join(f"{row.get('uuid')}:{row.get('ssh_port')}" for row in matches)
    raise LifecycleError(f"several instances match port {config.ssh.port}: {listing}")


# ──────────────────────────────────────────────────────────────────────────────
# browser backend: a logged-in console, used as the credential source
# ──────────────────────────────────────────────────────────────────────────────


def _looks_logged_in(url: str, body: str) -> bool:
    """The console's own definition of being in: not the login page, list labels present."""
    if any(marker in url for marker in CONSOLE_LOGIN_MARKERS):
        return False
    return any(marker in body for marker in CONSOLE_PAGE_MARKERS)


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise LifecycleError(
            "Playwright is not installed for this environment: run "
            "`pip install -e '.[browser]'` (or `pip install playwright`); if this "
            "machine has no Chrome, also run `playwright install chromium`"
        ) from exc
    return sync_playwright


@contextmanager
def _browser_context(config: LifecycleConfig, *, headless: bool):
    """A persistent Chrome/Chromium profile, closed in the right order.

    The profile is what keeps the console session alive between commands. It is a
    **full account session**, so it lives outside the repository and is created
    owner-readable only.
    """
    sync_playwright = _require_playwright()
    profile = config.api.browser_profile or DEFAULT_BROWSER_PROFILE
    profile.mkdir(parents=True, exist_ok=True)
    try:
        profile.chmod(0o700)
    except OSError:
        pass
    playwright = sync_playwright().start()
    context = None
    try:
        kwargs = {
            "headless": headless,
            "viewport": {"width": 1280, "height": 900},
            # The login slider is the one check that notices an automation
            # fingerprint; these are the standard two switches that remove it.
            "ignore_default_args": ["--enable-automation"],
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            context = playwright.chromium.launch_persistent_context(
                str(profile), channel=config.api.browser_channel, **kwargs
            )
        except Exception:  # noqa: BLE001 - no Chrome channel; try bundled Chromium
            context = playwright.chromium.launch_persistent_context(str(profile), **kwargs)
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        yield context
    finally:
        if context is not None:
            context.close()
        playwright.stop()


def _body_text(page) -> str:
    try:
        return page.locator("body").inner_text()
    except Exception:  # noqa: BLE001 - a page mid-navigation has no body yet
        return ""


def pro_ensure_instance(config: LifecycleConfig, retry_minutes: float) -> tuple[str, bool]:
    """Find or create a Pro instance; returns (uuid, created).

    All specs can answer "当前算力规格暂无库存"; when retry_minutes > 0 the
    configured specs are cycled with a pause until one succeeds.
    """
    uuid = pro_find_instance(config)
    if uuid:
        return uuid, False
    specs = [config.api.pro_gpu_spec, *config.api.pro_gpu_specs]
    specs = list(dict.fromkeys(spec for spec in specs if spec))
    deadline = time.time() + max(0.0, retry_minutes) * 60.0
    attempt = 0
    while True:
        config.api.pro_gpu_spec = specs[attempt % len(specs)]
        try:
            return api_pro_create(config.api), True
        except LifecycleError as exc:
            stock_out = "暂无库存" in str(exc) or "stock" in str(exc).lower()
            if not stock_out or retry_minutes <= 0 or time.time() >= deadline:
                raise
            nxt = specs[(attempt + 1) % len(specs)]
            delay = 60.0
            print(f"no stock for {config.api.pro_gpu_spec} ({exc}); trying {nxt} in "
                  f"{delay:g}s")
            time.sleep(delay)
            attempt += 1


def browser_auth(config: LifecycleConfig, wait_s: float = 180.0) -> int:
    """Open a browser and wait for a human AutoDL login, then keep the profile.

    AutoDL's login page offers WeChat / SMS / sub-account login plus a slider
    check. The browser is launched with the two switches that remove Playwright's
    automation fingerprint, because the slider is otherwise the failure point --
    and it is the *only* interactive step; everything afterwards runs headless.
    """
    with _browser_context(config, headless=False) as context:
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(CONSOLE_URL, wait_until="domcontentloaded", timeout=60_000)
        deadline = time.time() + max(30.0, wait_s)
        streak = 0
        logged_in = False
        while time.time() < deadline:
            if _looks_logged_in(page.url, _body_text(page)):
                streak += 1
                if streak >= 2:
                    time.sleep(3.0)
                    logged_in = True
                    break
            else:
                streak = 0
            page.wait_for_timeout(2_000)
        if not logged_in:
            shot = config.log_dir / "browser-auth-timeout.png"
            shot.parent.mkdir(parents=True, exist_ok=True)
            try:
                page.screenshot(path=str(shot))
            except Exception:  # noqa: BLE001 - a diagnostic screenshot is best effort
                shot = Path("")
            where = f" (screenshot: {shot})" if shot else ""
            raise LifecycleError(
                f"login was not completed within {int(wait_s)} s{where}; run browser-auth "
                "again and finish the slider + WeChat QR / SMS code in the opened window"
            )
        storage = config.api.browser_storage
        if storage is not None:
            storage.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(storage))
            try:
                storage.chmod(0o600)
            except OSError:
                pass
        profile = config.api.browser_profile or DEFAULT_BROWSER_PROFILE
        print(f"logged in; browser profile kept at {profile}")
    # Closing the window is not proof: ask the console for a token the same way
    # a later `up` would, and fail loudly if the session did not stick.
    try:
        browser_token(config)
    except LifecycleError as exc:
        raise LifecycleError(
            f"the login window closed but the session is not usable: {exc}"
        ) from None
    return 0


def _capture_token_from_page(page, timeout_s: float) -> str:
    """The Authorization header of the console's own authenticated XHR.

    Capturing the header the console itself sends is more stable than guessing a
    localStorage key, and it is exactly the header the HTTP API accepts.
    """
    captured: dict[str, str] = {}

    def on_request(request) -> None:
        if captured:
            return
        if "/api/v1/" not in request.url:
            return
        headers = getattr(request, "headers", {}) or {}
        value = headers.get("authorization") or headers.get("Authorization")
        if value:
            captured["token"] = value

    page.on("request", on_request)
    page.goto(CONSOLE_URL, wait_until="domcontentloaded", timeout=60_000)
    deadline = time.time() + max(10.0, timeout_s)
    not_logged_in = (
        "the console session is not logged in; run "
        "`python tools/autodl_lifecycle.py browser-auth` once"
    )
    while time.time() < deadline:
        body = _body_text(page)
        if any(marker in page.url for marker in CONSOLE_LOGIN_MARKERS):
            raise LifecycleError(not_logged_in)
        if captured and _looks_logged_in(page.url, body):
            return captured["token"]
        page.wait_for_timeout(500)
    if captured:
        raise LifecycleError(
            "a token was captured but the page never looked like the instance console; "
            "log in again with browser-auth"
        )
    raise LifecycleError(
        "no Authorization header was captured from the console; log in again with "
        "browser-auth, or use autodl.api = \"console\" with a token file"
    )


def _token_is_accepted(config: LifecycleConfig, token: str) -> bool:
    """Ask the console API whether the captured header is a live session.

    A login page can issue XHRs of its own; capturing one of those headers and
    then reporting "not logged in" only at the first power command would hide
    the real problem behind a confusing API error.
    """
    try:
        _request(
            _console_url("instance"),
            {
                "page_index": 1, "page_size": 1, "status": [], "charge_type": [],
                "sub_name": "", "unbind_sub_user": False,
            },
            {
                "Authorization": token,
                "Content-Type": "application/json;charset=utf-8",
                "AppVersion": config.api.app_version,
            },
        )
    except LifecycleError:
        return False
    return True


def browser_token(config: LifecycleConfig, timeout_s: float | None = None) -> str:
    """Capture a fresh JWT from the logged-in console, headless, and check it."""
    with _browser_context(config, headless=True) as context:
        page = context.new_page()
        try:
            token = _capture_token_from_page(page, timeout_s or config.api.browser_timeout_s)
        finally:
            page.close()
    if not _token_is_accepted(config, token):
        raise LifecycleError(
            "the console session is logged out or the captured token was rejected "
            "(AuthorizeFailed); run `python tools/autodl_lifecycle.py browser-auth`"
        )
    return token


# ──────────────────────────────────────────────────────────────────────────────
# SSH and the sync gate
# ──────────────────────────────────────────────────────────────────────────────


def _ssh_ready(remote, *, attempts: int = 1, delay_s: float = 0.0,
               timeout: float = 10.0) -> bool:
    for index in range(max(1, attempts)):
        try:
            result = remote.run("true", timeout=timeout, check=False)
            if result.returncode == 0:
                return True
        except Exception:  # noqa: BLE001, S110 - unreachable is the answer, not a crash
            pass
        if index + 1 < attempts:
            time.sleep(delay_s)
    return False


def _wait_ssh(remote, config: LifecycleConfig) -> bool:
    deadline = time.time() + config.api.boot_timeout_s
    while time.time() < deadline:
        if _ssh_ready(remote, timeout=10.0):
            return True
        time.sleep(config.api.poll_s)
    return False


def _wait_status(config: LifecycleConfig, wanted: str) -> bool:
    deadline = time.time() + config.api.boot_timeout_s
    while time.time() < deadline:
        try:
            if api_status(config.api) == wanted:
                return True
        except LifecycleError:
            pass
        time.sleep(config.api.poll_s)
    return False


def sync_problems(config: LifecycleConfig, remote, *, verify_hash: bool = True) -> list[str]:
    """Why shutting down now would lose data; an empty list means it is safe.

    Every remote run that has reached a terminal status must have a local
    `FETCH.json` from `remote_train.py fetch` with `verified > 0` and no
    `missing` entries, and its final checkpoint must still hash to what the box
    holds. `unknown` counts as terminal: a reboot is a result whose evidence is
    still owed. A running or queued job is never a sync problem -- it is a
    reason not to shut down at all, returned separately by `running_jobs`.
    """
    module = _remote_train()
    problems: list[str] = []
    for job_id in module.job_ids(remote, config.ssh, limit=100):
        try:
            job = module.read_job(remote, config.ssh, job_id)
        except Exception as exc:  # noqa: BLE001 - unreadable job is a problem
            problems.append(f"{job_id}: cannot read the job record ({type(exc).__name__})")
            continue
        status = job.get("status", "unknown")
        if status in {"running", "queued"}:
            continue  # `down` refuses on running jobs before it ever asks this
        local = config.results_dir / job_id
        fetch_path = local / "FETCH.json"
        if not fetch_path.is_file():
            problems.append(f"{job_id}: terminal ({status}) but never fetched")
            continue
        try:
            fetch = json.loads(fetch_path.read_text("utf-8"))
        except json.JSONDecodeError:
            problems.append(f"{job_id}: FETCH.json is not JSON")
            continue
        if not fetch.get("verified") or fetch.get("missing"):
            problems.append(
                f"{job_id}: fetch did not verify ({fetch.get('verified', 0)} files, "
                f"missing {fetch.get('missing', [])})"
            )
            continue
        if verify_hash and job.get("checkpoints"):
            newest = job["checkpoints"][-1]
            remote_hash = _remote_sha256(remote, newest["path"])
            local_model = _newest_local_model(local)
            if remote_hash and local_model is not None:
                local_hash = _sha256_file(local_model)
                if local_hash != remote_hash:
                    problems.append(
                        f"{job_id}: local {local_model.name} differs from the box copy"
                    )
    return problems


def running_jobs(config: LifecycleConfig, remote) -> list[str]:
    """Job ids whose process is alive or queued; `down` requires these empty."""
    module = _remote_train()
    alive: list[str] = []
    for job_id in module.job_ids(remote, config.ssh, limit=100):
        try:
            job = module.read_job(remote, config.ssh, job_id)
        except Exception:  # noqa: BLE001, S112 - an unreadable job is not evidence
            continue
        if job.get("status") in {"running", "queued"}:
            alive.append(job_id)
    return alive


def _remote_sha256(remote, path: str) -> str:
    result = remote.run(f"sha256sum {shlex.quote(path)} 2>/dev/null | cut -d' ' -f1",
                        timeout=120, check=False)
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def _newest_local_model(local: Path) -> Path | None:
    models = list(local.rglob("model_*.pt"))
    if not models:
        return None

    def iteration(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(models, key=iteration)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_submit(config: LifecycleConfig, argv: list[str]) -> int:
    return subprocess.run([sys.executable, str(REMOTE_TRAIN), *argv], cwd=REPO,
                          check=False).returncode


def _release_from_sync(stdout: str) -> str:
    """The id `remote_train.py sync` prints: `release <id>: <n> files, ...`."""
    for line in stdout.splitlines():
        if line.startswith("release "):
            return line.split()[1].rstrip(":")
    return ""


def _remote_call(config: LifecycleConfig, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REMOTE_TRAIN), *argv], cwd=REPO,
                          capture_output=True, text=True, check=False)


# ──────────────────────────────────────────────────────────────────────────────
# audit log
# ──────────────────────────────────────────────────────────────────────────────


def _log(config: LifecycleConfig, action: str, **fields: object) -> None:
    config.log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "action": action,
        **fields,
    }
    with (config.log_dir / "lifecycle.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# the steps
# ──────────────────────────────────────────────────────────────────────────────


def up(config: LifecycleConfig, *, manual: bool = False, dry_run: bool = False,
       retry_minutes: float = 0.0, rent_new: bool = False,
       nocard: bool = False) -> int:
    """Make the instance reachable; returns 0 once SSH answers."""
    module = _remote_train()
    remote = module.Remote(config.ssh)
    backend = "manual" if manual else config.api.backend
    if backend == "pro":
        if dry_run:
            if rent_new:
                print(format_pro_plan(pro_plan(config)))
            else:
                print("[dry-run] would find or create the configured Pro instance")
            return 0
        config.api.instance_uuid = pro_find_instance(config)
        if not config.api.instance_uuid:
            if not rent_new:
                raise LifecycleError(
                    "no Pro instance found; pass --rent-new to create one with the "
                    "official API (this needs the developer token and costs money)"
                )
            config.api.instance_uuid, created = pro_ensure_instance(config, retry_minutes)
            if created:
                print(f"created Pro instance {config.api.instance_uuid} "
                      f"(spec {config.api.pro_gpu_spec})")
                _log(config, "pro_create", instance=config.api.instance_uuid,
                     spec=config.api.pro_gpu_spec)
        if not _wait_status(config, "running"):
            raise LifecycleError("the Pro instance did not reach running before the timeout")
        snapshot = api_pro_snapshot(config.api, config.api.instance_uuid)
        print(f"ssh: {snapshot.get('ssh_command', '(no ssh command yet)')}")
        print(f"password: {'set' if snapshot.get('root_password') else 'missing'}; "
              f"price: {snapshot.get('payg_price', '?')} 毫元/h")
        return 0
    if backend != "manual":
        if dry_run:
            print(f"[dry-run] would resolve the instance by ssh port {config.ssh.port} "
                  f"and power it on if it is not running"
                  + (" (no-card mode)" if nocard else ""))
            if rent_new:
                print(format_market_plan(market_plan(config)))
            else:
                print("[dry-run] would wait for SSH")
            return 0
        config.api.instance_uuid = resolve_instance_uuid(config)
        status = api_status(config.api)
        print(f"instance {config.api.instance_uuid}: {status}")
        if status not in {"running"}:
            deadline = time.time() + max(0.0, retry_minutes) * 60.0
            delay = max(5.0, config.api.poll_s)
            while True:
                try:
                    api_power_on(config.api, nocard=nocard)
                    break
                except LifecycleError as exc:
                    # "机器空闲GPU不足": the host has no free card right now.
                    # Retrying is the only action available; the message is kept
                    # verbatim because it is the platform's answer. The delay
                    # grows so a two-hour wait is not 700 API calls.
                    if retry_minutes <= 0 or time.time() >= deadline:
                        _log(config, "power_on_refused",
                             instance=config.api.instance_uuid, reason=str(exc))
                        if rent_new:
                            print(format_market_plan(market_plan(config)))
                            raise LifecycleError(
                                "the old host has no free card; the market plan is above. "
                                "Creating a replacement instance is not enabled in this "
                                "build -- review the plan first (`up --rent-new --dry-run`)"
                            ) from None
                        raise
                    print(f"power on refused ({exc}); retrying in {delay:g}s")
                    time.sleep(delay)
                    delay = min(120.0, delay * 1.5)
            _log(config, "power_on", instance=config.api.instance_uuid, backend=backend)
            if not _wait_status(config, "running"):
                raise LifecycleError("the instance did not reach running before the timeout")
    else:
        print("manual start: start the instance in the AutoDL console if it is not running")
    if dry_run:
        print("[dry-run] would wait for SSH")
        return 0
    if not _wait_ssh(remote, config):
        raise LifecycleError("SSH never answered; the instance may still be starting")
    print(f"ssh reachable: {config.ssh.hostname}:{config.ssh.port}")
    _log(config, "up", host=config.ssh.hostname, port=config.ssh.port)
    return 0


def down(config: LifecycleConfig, *, force: bool = False, fetch_missing: bool = False,
         dry_run: bool = False, keep_running: bool = False) -> int:
    """Shut the instance down only when no work and no evidence is left on it."""
    module = _remote_train()
    remote = module.Remote(config.ssh)

    if not _ssh_ready(remote, attempts=2, delay_s=2.0):
        print("SSH is not reachable; nothing to shut down")
        return 0

    alive = running_jobs(config, remote)
    if alive and not force:
        raise LifecycleError(
            f"{len(alive)} job(s) still running: {', '.join(alive[:5])}; stop them or pass --force"
        )

    problems = sync_problems(config, remote)
    if problems and fetch_missing and not dry_run:
        module = _remote_train()
        for job_id in module.job_ids(remote, config.ssh, limit=100):
            if not any(item.startswith(job_id + ":") for item in problems):
                continue
            try:
                job = module.read_job(remote, config.ssh, job_id)
            except Exception as exc:  # noqa: BLE001 - reported, not fetched blind
                print(f"cannot fetch {job_id}: {type(exc).__name__}")
                continue
            print(f"fetching {job_id} before shutdown")
            # A failed or stopped run has no checkpoint, and a selective fetch
            # refuses one: `--all` is the only way to take its logs off the box.
            want_all = config.fetch_all or not job.get("checkpoints")
            code = _remote_submit(config, ["fetch", job_id] + (["--all"] if want_all else []))
            if code != 0:
                raise LifecycleError(f"fetch {job_id} failed with exit code {code}")
        problems = sync_problems(config, remote)
    if problems and not force:
        raise LifecycleError(
            "refusing to shut down, the Mac does not hold a verified copy of:\n  "
            + "\n  ".join(problems)
        )
    if problems:
        print("warning: shutting down despite unsynced evidence (--force):")
        for problem in problems:
            print(f"  {problem}")

    if keep_running:
        print("[dry-run] would verify the sync gate and stop here; not shutting down")
        return 0
    if dry_run:
        print("[dry-run] would shut down the instance"
              + (" (after killing/stopping jobs)" if alive and force else ""))
        return 0

    if config.shutdown == "api":
        if config.api.backend == "manual" or not config.api.instance_uuid:
            raise LifecycleError("shutdown='api' needs a configured AutoDL API and instance_uuid")
        api_power_off(config.api)
        _log(config, "power_off", instance=config.api.instance_uuid, backend=config.api.backend)
        _wait_status(config, "shutdown")
    else:
        # The AutoDL-documented way to stop billing from inside the container:
        # https://www.autodl.com/docs/save_money/ (场景三). The SSH session dies
        # with the box, so a dropped connection is the expected answer.
        try:
            remote.run("sync; /usr/bin/shutdown -h now", timeout=30, check=False)
        except Exception:  # noqa: BLE001, S110 - the connection dropping is success
            pass
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if not _ssh_ready(remote, timeout=5.0):
                print("shutdown sent; SSH no longer answers (expected)")
                _log(config, "power_off", instance=config.api.instance_uuid or config.ssh.hostname,
                     backend="ssh")
                return 0
            time.sleep(2.0)
        raise LifecycleError("shutdown was sent but SSH still answers after 120 s")
    print("instance is shut down")
    return 0


def verify(config: LifecycleConfig) -> int:
    """Print the sync gate without shutting anything down."""
    module = _remote_train()
    remote = module.Remote(config.ssh)
    if not _ssh_ready(remote, attempts=2, delay_s=2.0):
        raise LifecycleError("SSH is not reachable")
    alive = running_jobs(config, remote)
    problems = sync_problems(config, remote)
    if alive:
        print(f"running jobs: {', '.join(alive)}")
    if problems:
        print("unsynchronised evidence:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("sync gate: every terminal run is fetched and verified")
    return 0


def session(config: LifecycleConfig, *, task: str, release: str | None,
            submit_args: str, max_hours: float | None, sync: bool, post: bool,
            dry_run: bool, retry_minutes: float = 0.0, rent_new: bool = False,
            auto_shutdown: bool = False) -> int:
    """up -> (sync) -> submit -> wait -> fetch -> verify -> down.

    The shutdown happens only after the job reached a terminal status and the
    sync gate passed. If a step before that fails, the box is left running --
    shutting down a machine that is in an unknown state is worse than leaving
    it for a human.
    """
    config.auto_shutdown = config.auto_shutdown or auto_shutdown
    if config.api.backend == "pro":
        return pro_session(
            config, task=task, submit_args=submit_args, max_hours=max_hours,
            post=post, retry_minutes=retry_minutes, dry_run=dry_run,
        )
    module = _remote_train()
    if up(config, dry_run=dry_run, retry_minutes=retry_minutes,
          rent_new=rent_new) != 0:
        raise LifecycleError("the instance never became reachable")
    remote = module.Remote(config.ssh)
    if not dry_run:
        check = _remote_call(config, ["check"])
        if check.returncode != 0:
            print(check.stdout.strip())
            raise LifecycleError("remote_train.py check failed; leaving the instance running")
    if sync:
        print("[session] sync the checkout before submitting")
        if not dry_run:
            result = _remote_call(config, ["sync"])
            print(result.stdout.strip())
            if result.returncode != 0:
                raise LifecycleError("sync failed; leaving the instance alone")
            release = _release_from_sync(result.stdout) or release
    if not release and not sync:
        raise LifecycleError(
            "--no-sync needs an explicit --release; the box only accepts a release id"
        )
    if not release:
        print("[dry-run] release id would come from sync")
        release = "<release-from-sync>"
    submit = [
        "submit", "--task", task, "--release", release,
        "--max-hours", str(max_hours if max_hours is not None else 2.0),
    ]
    if submit_args:

        submit += ["--extra", submit_args]
    if dry_run:
        print(f"[dry-run] would run: remote_train.py {' '.join(submit)}")
        ending = "shut down" if config.auto_shutdown else "leave the instance running"
        print(f"[dry-run] then wait, fetch, verify the sync gate and {ending}")
        return 0

    result = _remote_call(config, submit)
    print(result.stdout.strip())
    if result.returncode != 0:
        raise LifecycleError("submit failed; leaving the instance running")
    job_id = ""
    for line in result.stdout.splitlines():
        if line.startswith("submitted "):
            job_id = line.split()[1]
            break
    if not job_id:
        raise LifecycleError("submit printed no job id; leaving the instance running")

    print(f"[session] waiting for {job_id}")
    while True:
        job = module.read_job(remote, config.ssh, job_id)
        status = job.get("status", "unknown")
        if status in {"succeeded", "failed", "stopped", "unknown"}:
            print(f"[session] {job_id}: {status}")
            break
        time.sleep(config.api.poll_s)

    fetch = ["fetch", job_id] + (["--all"] if config.fetch_all else [])
    if post:
        fetch.append("--post")
    code = _remote_submit(config, fetch)
    if code != 0:
        raise LifecycleError(f"fetch failed with exit code {code}; leaving the instance running")

    problems = sync_problems(config, remote)
    if problems:
        raise LifecycleError(
            "sync gate failed after the fetch; leaving the instance running:\n  "
            + "\n  ".join(problems)
        )
    print("[session] sync gate passed")
    if not config.auto_shutdown:
        print("[session] auto_shutdown is off; the instance is left running "
              "(use `down` when you want to stop it)")
        return 0
    return down(config, fetch_missing=False, dry_run=False)


# ──────────────────────────────────────────────────────────────────────────────
# standard instance: adopt one created by hand, and watch its market
# ──────────────────────────────────────────────────────────────────────────────

#: "3080 or lower" for the fallback path.
STANDARD_GPUS = ("RTX 3080 Ti", "RTX 3080", "RTX 3070", "RTX 3060", "RTX 2080 Ti")


def _console_headers(config: LifecycleConfig) -> dict:
    token = browser_token(config)
    return {"Authorization": token, "Content-Type": "application/json;charset=utf-8",
            "AppVersion": config.api.app_version}


def standard_instances(config: LifecycleConfig) -> list[dict]:
    """Every standard container instance on the account (console API, JWT)."""
    payload = _request(
        _console_url("instance"),
        {"page_index": 1, "page_size": 100, "status": [], "charge_type": [],
         "sub_name": "", "unbind_sub_user": False},
        _console_headers(config),
    )
    data = payload.get("data", {})
    return list(data.get("list", [])) if isinstance(data, dict) else []


def adopt_instance(config: LifecycleConfig, *, task: str, max_hours: float | None,
                   shutdown: bool, instance_uuid: str = "", num_envs: int = 64,
                   max_iterations: int = 20) -> int:
    """Take over a standard instance created by hand (the create step needs a captcha).

    Everything after the hand-created instance exists is automated: SSH key,
    a session-only remote config, sync, bootstrap, a 20-iteration smoke run,
    fetch, verification, and (optionally) shutdown.
    """
    rows = standard_instances(config)
    running = [row for row in rows if row.get("status") == "running"]
    if instance_uuid:
        instance = next((row for row in rows if row.get("uuid") == instance_uuid), None)
        if instance is None or instance.get("status") != "running":
            raise LifecycleError(f"instance {instance_uuid} is not running")
    elif running:
        instance = running[0]
    else:
        raise LifecycleError(
            "no running standard instance; create one in the AutoDL console first "
            "(the create page needs the human captcha)"
        )
    uuid = str(instance.get("uuid", ""))
    host = str(instance.get("proxy_host") or "")
    port = int(instance.get("ssh_port") or 0)
    password = str(instance.get("root_password") or "")
    print(f"adopting {uuid} ({instance.get('snapshot_gpu_alias_name', '')}) "
          f"{host}:{port}")
    if not host or not port:
        raise LifecycleError("the instance has no SSH host/port yet")
    if password:
        _install_pubkey(host, port, password)
        print("installed this Mac's SSH public key")
    snapshot = {"proxy_host": host, "ssh_port": port}
    session_config = _pro_session_config(
        config, snapshot, str(getattr(config.ssh, "python", "python3")),
        workdir=str(getattr(config.ssh, "workdir", "/root/jumper")))
    python = _detect_python(config, session_config)
    session_config = _pro_session_config(config, snapshot, python,
                                         workdir=str(getattr(config.ssh, "workdir", "/root/jumper")))
    print(f"python on the box: {python}")
    _ensure_tmux(session_config)
    sync = _remote_train_cmd(session_config, "sync")
    print(sync.stdout.strip()[-600:])
    if sync.returncode != 0:
        raise LifecycleError("sync failed; the instance is left running")
    release = ""
    for line in sync.stdout.splitlines():
        if line.startswith("release "):
            release = line.split()[1].rstrip(":")
    if not release:
        raise LifecycleError("sync printed no release id")
    boot = _remote_train_cmd(session_config, "bootstrap", "--release", release)
    print(boot.stdout.strip()[-1200:])
    if boot.returncode != 0:
        raise LifecycleError("bootstrap failed; the instance is left running")
    check = _remote_train_cmd(session_config, "check")
    print(check.stdout.strip()[-1500:])
    if check.returncode != 0:
        raise LifecycleError("remote_train.py check failed; the instance is left running")
    submit = ["submit", "--task", task, "--release", release,
              "--num-envs", str(num_envs), "--max-iterations", str(max_iterations),
              "--max-hours", str(max_hours if max_hours is not None else 0.5)]
    result = _remote_train_cmd(session_config, *submit)
    print(result.stdout.strip()[-800:])
    if result.returncode != 0:
        raise LifecycleError("submit failed; the instance is left running")
    job_id = ""
    for line in result.stdout.splitlines():
        if line.startswith("submitted "):
            job_id = line.split()[1]
            break
    if not job_id:
        raise LifecycleError("submit printed no job id")
    status = _wait_pro_job(session_config, job_id, timeout_s=30 * 60.0)
    print(f"job {job_id}: {status}")
    fetch = _remote_train_cmd(session_config, "fetch", job_id)
    print(fetch.stdout.strip()[-500:])
    if fetch.returncode != 0:
        raise LifecycleError("fetch failed; the instance is left running")
    local = config.results_dir / job_id
    fetch_file = local / "FETCH.json"
    if not fetch_file.is_file() or not json.loads(fetch_file.read_text())["verified"]:
        raise LifecycleError("the fetched run did not verify; the instance is left running")
    if shutdown:
        module = _remote_train()
        remote = module.Remote(module.load_config(str(session_config)))
        remote.run("sync; /usr/bin/shutdown -h now", timeout=30, check=False)
        print("shutdown sent; the instance keeps its disk")
    else:
        print("instance left running")
    return 0


_REGION_SIGNS = frozenset({
    "westDC3", "beijingDC2", "chongqingDC1", "neimengDC3", "beijingDC1",
    "foshanDC1", "beijingDC3", "neimengDC2", "shanghaiDC1",
    "payg", "daily", "weekly", "monthly", "basicImage", "CodeWithGPU",
    "customImage", "1", "2", "3", "4", "5", "6", "7", "8", "10", "12",
})


def standard_available(config: LifecycleConfig,
                       gpus: tuple[str, ...]) -> list[dict]:
    """What the create page would actually let this account rent.

    The machine-list API's `gpu_idle_num` and the create page disagree; only the
    page's enabled machine radios predict whether "创建并开机" can work. The page
    re-renders asynchronously and keeps the previous GPU's rows, so every GPU is
    checked on a **fresh page load** and a machine counts only when its own row
    contains the selected GPU's name.
    """
    import re

    def machine_ids() -> list[str]:
        found_ids = []
        for radio in page.locator("input[type=radio]").all():
            value = radio.get_attribute("value") or ""
            if value in _REGION_SIGNS or value.isdigit():
                continue
            if not re.fullmatch(r"[a-z0-9]{10}", value):
                continue
            if not radio.is_disabled():
                found_ids.append(value)
        return found_ids

    found: list[dict] = []
    with _browser_context(config, headless=True) as context:
        page = context.pages[0] if context.pages else context.new_page()
        for gpu in gpus:
            page.goto("https://www.autodl.com/create", wait_until="domcontentloaded",
                      timeout=60000)
            page.wait_for_timeout(3500)
            locator = page.get_by_text(gpu, exact=True)
            if not locator.count():
                continue
            try:
                locator.first.click(timeout=5000)
            except Exception as exc:  # noqa: BLE001 - one GPU must not stop the rest
                print(f"{gpu}: could not select it on the create page "
                      f"({type(exc).__name__})")
                continue
            page.wait_for_timeout(3000)
            text = page.locator("body").inner_text()
            machines = []
            for machine_id in machine_ids():
                position = text.find(machine_id)
                if position < 0:
                    continue
                # the row ends at the next machine id; a stale row from another
                # GPU therefore cannot pass the name check below
                rest = text[position + len(machine_id):]
                next_ids = [match.start() for match in
                            re.finditer(r"[a-z0-9]{10}", rest)]
                row = rest[:next_ids[0]] if next_ids else rest[:500]
                if gpu not in row:
                    continue
                alias = re.search(r"([A-Za-z0-9]+机)", row)
                price = re.search(r"￥([0-9.]+)/时", row)
                machines.append({
                    "machine_id": machine_id,
                    "alias": alias.group(1) if alias else "",
                    "region": "",
                    "price": f"￥{price.group(1)}/时" if price else "",
                })
            found.append({"gpu": gpu, "machines": machines,
                          "total": len(machines)})
    return found


def format_available(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        if not row["machines"]:
            continue
        for machine in row["machines"]:
            lines.append(f"{row['gpu']:<16}{machine.get('alias', ''):<8}"
                         f"{machine.get('price', ''):<10}{machine['machine_id']}")
    return "\n".join(lines)


def market_check(config: LifecycleConfig, *, gpus: tuple[str, ...],
                 min_idle: int, notify: bool, force: bool = False) -> int:
    """Read-only check of what the create page would actually rent.

    With no queued task this returns immediately: standard monitoring is also
    task-driven, so an idle account is not woken up every hour. `--force` is for
    a manual look.
    """
    if read_pending_task() is None and not force:
        print("no pending training task; standard monitoring skipped")
        return 0
    allowed = gpus or STANDARD_GPUS
    found = standard_available(config, allowed)
    available = [row for row in found if row["machines"]]
    signature = "|".join(
        f"{row['gpu']}:{','.join(m['machine_id'] for m in row['machines'])}"
        for row in available
    )
    state_path = config.log_dir / "market-state.json"
    previous = ""
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text()).get("signature", "")
        except json.JSONDecodeError:
            previous = ""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "allowed": list(allowed),
        "available": available,
        "signature": signature,
    }, ensure_ascii=False, indent=2) + "\n", "utf-8")
    if not available:
        print(f"no allocatable card for {'/'.join(allowed)} right now "
              f"(the create page disables every machine radio)")
        return 0
    print(format_available(available))
    if notify and signature != previous:
        text = ("【AutoDL 标准卡可租】\n" + format_available(available)
                + "\n如需保底训练：在控制台手动创建（一次验证码），建好后回复我，"
                  "我用 adopt 接管。")
        try:
            import importlib.util as _ilu

            spec = _ilu.spec_from_file_location(
                "postfetch_feishu", REPO / "tools" / "postfetch" / "feishu.py")
            module = _ilu.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            chat = module.discover(REPO)
            result = module.send(chat, text)
            print(f"feishu: {result.get('status')}")
        except Exception as exc:  # noqa: BLE001 - notification is best effort
            print(f"feishu notification failed: {type(exc).__name__}: {exc}")
    return 0


def _pro_session_config(config: LifecycleConfig, snapshot: dict, python: str,
                        workdir: str = "/root/jumper") -> Path:
    """Write the remote_train config this Pro instance needs (never .remote.toml)."""
    host = str(snapshot.get("proxy_host") or "")
    port = int(snapshot.get("ssh_port") or 0)
    if not host or not port:
        raise LifecycleError("the Pro snapshot has no SSH host/port yet")
    known = (config.log_dir / "known_hosts-pro").resolve()
    path = config.log_dir / "remote-pro.toml"
    identity = str(getattr(config.ssh, "identity", "") or "")
    text = f"""# generated by tools/autodl_lifecycle.py for Pro instance {config.api.instance_uuid}
hostname = "{host}"
port = {port}
user = "root"
identity = "{identity}"
known_hosts = "{known}"
workdir = "{workdir}"
python = "{python}"
results = "{config.results_dir}"
max_hours = 1.0
backend = "warp"
device = "cuda:0"
num_envs = 64
tensorboard_port = 6006
pip_index = "https://pypi.tuna.tsinghua.edu.cn/simple"
mujoco_gl = "disabled"
"""
    path.write_text(text, "utf-8")
    path.chmod(0o600)
    return path


def _ensure_tmux(session_config: Path) -> None:
    """A fresh Pro image has no tmux; `submit` needs it for the background job."""
    module = _remote_train()
    remote = module.Remote(module.load_config(str(session_config)))
    if remote.run("command -v tmux", check=False).stdout.strip():
        return
    print("installing tmux on the box (apt-get)")
    proc = remote.run("apt-get update -y >/dev/null 2>&1 && apt-get install -y tmux",
                      timeout=600, check=False)
    print(proc.stdout[-400:])
    if proc.returncode != 0:
        raise LifecycleError("could not install tmux on the box")


def _install_pubkey(host: str, port: int, password: str) -> None:
    """Put this Mac's key into the new box; Pro instances start password-only."""
    import paramiko

    key_path = Path("~/.ssh/id_ed25519_github_drafly.pub").expanduser()
    if not key_path.is_file():
        raise LifecycleError(f"no public key at {key_path}")
    pub = key_path.read_text("utf-8").strip()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, port=port, username="root", password=password,
                       timeout=30, banner_timeout=30, auth_timeout=30)
        command = (
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
            "touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && "
            f"grep -qF {shlex.quote(pub)} /root/.ssh/authorized_keys || "
            f"echo {shlex.quote(pub)} >> /root/.ssh/authorized_keys"
        )
        _stdin, stdout, stderr = client.exec_command(command, timeout=30)
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise LifecycleError(f"installing the SSH key failed: {stderr.read()[:200]}")
    finally:
        client.close()


def _detect_python(config: LifecycleConfig, session_config: Path) -> str:
    module = _remote_train()
    remote = module.Remote(module.load_config(str(session_config)))
    script = (
        "for p in /root/miniconda3/bin/python /usr/bin/python3 /usr/local/bin/python3; do "
        "[ -x $p ] && $p -c 'import sys; print(sys.executable, sys.version_info[0], sys.version_info[1])'; "
        "done"
    )
    out = remote.run(script, check=False).stdout
    candidates = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            candidates.append((int(parts[1]), int(parts[2]), parts[0]))
    for major, minor, path in sorted(candidates, reverse=True):
        if (major, minor) >= (3, 10):
            return path
    raise LifecycleError(f"no Python >=3.10 on the Pro instance; found {candidates}")


def _remote_train_cmd(session_config: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REMOTE_TRAIN), "-c", str(session_config), *args],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


def _wait_pro_job(session_config: Path, job_id: str, timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = _remote_train_cmd(session_config, "status", job_id, "--json")
        try:
            rows = json.loads(proc.stdout or "[]")
            status = rows[0].get("status", "") if rows else ""
        except json.JSONDecodeError:
            status = ""
        if status in {"succeeded", "failed", "stopped", "unknown"}:
            return status
        time.sleep(15.0)
    return "timeout"


def pro_session(config: LifecycleConfig, *, task: str, submit_args: str,
                max_hours: float | None, post: bool, retry_minutes: float,
                dry_run: bool, num_envs: int = 64,
                max_iterations: int = 20) -> int:
    if dry_run:
        print(format_pro_plan(pro_plan(config)))
        print("[dry-run] would create/find the instance, install the key, sync, "
              "bootstrap, submit a 20-iteration smoke run, fetch and shut down")
        return 0
    config.api.instance_uuid, created = pro_ensure_instance(config, retry_minutes)
    print(f"Pro instance: {config.api.instance_uuid}"
          + (f" (created, spec {config.api.pro_gpu_spec})" if created else ""))
    if created:
        _log(config, "pro_create", instance=config.api.instance_uuid,
             spec=config.api.pro_gpu_spec)
    if not _wait_status(config, "running"):
        raise LifecycleError("the Pro instance did not reach running")
    deadline = time.time() + 300.0
    snapshot: dict = {}
    while time.time() < deadline:
        snapshot = api_pro_snapshot(config.api, config.api.instance_uuid)
        if snapshot.get("ssh_port") and snapshot.get("proxy_host"):
            break
        time.sleep(10.0)
    host = str(snapshot.get("proxy_host") or "")
    port = int(snapshot.get("ssh_port") or 0)
    print(f"ssh {host}:{port}; price {snapshot.get('payg_price', '?')} 毫元/h")
    password = str(snapshot.get("root_password") or "")
    if password:
        _install_pubkey(host, port, password)
        print("installed this Mac's SSH public key")
    session_config = _pro_session_config(
        config, snapshot, str(getattr(config.ssh, "python", "python3")),
        workdir=str(getattr(config.ssh, "workdir", "/root/jumper")))
    python = _detect_python(config, session_config)
    session_config = _pro_session_config(config, snapshot, python,
                                         workdir=str(getattr(config.ssh, "workdir", "/root/jumper")))
    print(f"python on the box: {python}")
    _ensure_tmux(session_config)
    sync = _remote_train_cmd(session_config, "sync")
    print(sync.stdout.strip()[-600:])
    if sync.returncode != 0:
        raise LifecycleError("sync failed; leaving the instance running")
    release = ""
    for line in sync.stdout.splitlines():
        if line.startswith("release "):
            release = line.split()[1].rstrip(":")
    if not release:
        raise LifecycleError("sync printed no release id")
    boot = _remote_train_cmd(session_config, "bootstrap", "--release", release)
    print(boot.stdout.strip()[-1200:])
    if boot.returncode != 0:
        raise LifecycleError("bootstrap failed; leaving the instance running")
    check = _remote_train_cmd(session_config, "check")
    print(check.stdout.strip()[-2000:])
    if check.returncode != 0:
        raise LifecycleError("remote_train.py check failed; leaving the instance running")

    submit = ["submit", "--task", task, "--release", release,
              "--num-envs", str(num_envs), "--max-iterations", str(max_iterations),
              "--max-hours", str(max_hours if max_hours is not None else 0.5)]
    if submit_args:
        submit += ["--extra", submit_args]
    result = _remote_train_cmd(session_config, *submit)
    print(result.stdout.strip()[-800:])
    if result.returncode != 0:
        raise LifecycleError("submit failed; leaving the instance running")
    job_id = ""
    for line in result.stdout.splitlines():
        if line.startswith("submitted "):
            job_id = line.split()[1]
            break
    if not job_id:
        raise LifecycleError("submit printed no job id")
    status = _wait_pro_job(session_config, job_id, timeout_s=30 * 60.0)
    print(f"job {job_id}: {status}")
    fetch = _remote_train_cmd(session_config, "fetch", job_id)
    print(fetch.stdout.strip()[-500:])
    if fetch.returncode != 0:
        raise LifecycleError("fetch failed; leaving the instance running")
    local = config.results_dir / job_id
    fetch_file = local / "FETCH.json"
    if not fetch_file.is_file() or not json.loads(fetch_file.read_text())["verified"]:
        raise LifecycleError("the fetched run did not verify; leaving the instance running")
    if post:
        proc = subprocess.run(
            [sys.executable, "-m", "tools.postfetch", "run", "--job-dir", str(local)],
            cwd=REPO, check=False,
        )
        if proc.returncode != 0:
            print("postfetch reported a failure; the download itself is verified")
    if config.auto_shutdown:
        print("shutting the Pro instance down")
        api_power_off(config.api)
        _log(config, "power_off", instance=config.api.instance_uuid, backend="pro")
    else:
        print("auto_shutdown is off; the Pro instance is left running")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# pending training tasks: monitor Pro only while one exists
# ──────────────────────────────────────────────────────────────────────────────

PENDING_TASK = REPO / "outputs" / "autodl" / "pending-task.json"
PRO_STATE = REPO / "outputs" / "autodl" / "pro-task-state.json"

#: Low-end Pro specs only: 4080S-32G, 4090D, 4090-48G, 3090-48G.
PRO_LOW_SPECS = ("v-32g-p", "4090D", "v-48g", "v-48g-350w")


def read_pending_task() -> dict | None:
    if not PENDING_TASK.is_file():
        return None
    try:
        task = json.loads(PENDING_TASK.read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    return task if isinstance(task, dict) and task.get("task") else None


def queue_task(task: str, *, max_hours: float | None, num_envs: int,
               max_iterations: int, submit_args: str = "") -> Path:
    PENDING_TASK.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "max_hours": max_hours,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "submit_args": submit_args,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    PENDING_TASK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            "utf-8")
    return PENDING_TASK


def _feishu_text(text: str) -> None:
    try:
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location(
            "postfetch_feishu", REPO / "tools" / "postfetch" / "feishu.py")
        module = _ilu.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module.send(module.discover(REPO), text)
        print(f"feishu: {result.get('status')}")
    except Exception as exc:  # noqa: BLE001 - notification is best effort
        print(f"feishu notification failed: {type(exc).__name__}: {exc}")


def pro_worker(config: LifecycleConfig, *, background: bool = False) -> int:
    """Run the pending task on a low-end Pro instance, if one is waiting.

    With no pending task this returns immediately: no Pro market calls and no
    notifications. With a task it tries only the allowed low-end specs, creates
    from the saved private image on the first hit, runs the task in the same
    session, then clears the queue. If no low-end card appears for six hours it
    sends one Feishu message asking whether a high-end model may be approved.
    """
    task = read_pending_task()
    if task is None:
        print("no pending training task; Pro monitoring skipped")
        return 0
    if background:
        log_path = REPO / "outputs" / "autodl" / "pro-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "pro-worker"],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"pro worker started in the background; log {log_path}")
        return 0
    allowed = [spec for spec in
               dict.fromkeys([config.api.pro_gpu_spec, *config.api.pro_gpu_specs])
               if spec in PRO_LOW_SPECS]
    if not allowed:
        raise LifecycleError(f"no low-end Pro spec configured; allowed: {PRO_LOW_SPECS}")
    config.api.pro_gpu_spec = allowed[0]
    config.api.pro_gpu_specs = tuple(allowed[1:])
    try:
        config.api.instance_uuid, _created = pro_ensure_instance(config, retry_minutes=0)
    except LifecycleError as exc:
        stock_out = "暂无库存" in str(exc) or "stock" in str(exc).lower()
        if not stock_out:
            raise
        state = {}
        if PRO_STATE.is_file():
            try:
                state = json.loads(PRO_STATE.read_text())
            except json.JSONDecodeError:
                state = {}
        first = state.get("first_no_stock_at")
        if not first:
            first = datetime.now().astimezone().isoformat(timespec="seconds")
        state.update({"first_no_stock_at": first, "task": task.get("task")})
        PRO_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                             "utf-8")
        waited = datetime.now().astimezone() - datetime.fromisoformat(first)
        if waited.total_seconds() >= 6 * 3600 and not state.get("escalated_at"):
            _feishu_text(
                "【Pro 低价卡 6 小时无库存】\n"
                f"待执行任务：{task.get('task')}\n"
                "4080S/4090D/4090-48G/3090-48G 均无库存。是否允许使用 4090 以上的"
                "高性能显卡（如 5090/PRO6000）？回复批准后我会调整规格并继续自动创建。"
            )
            state["escalated_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds")
            PRO_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2)
                                 + "\n", "utf-8")
        print(f"no low-end Pro stock for {waited}; task stays queued")
        return 0
    PRO_STATE.unlink(missing_ok=True)
    print(f"Pro instance {config.api.instance_uuid} created "
          f"({config.api.pro_gpu_spec}); running {task.get('task')}")
    try:
        rc = pro_session(
            config, task=str(task.get("task")), submit_args=str(task.get("submit_args") or ""),
            max_hours=task.get("max_hours"), post=False, retry_minutes=0,
            dry_run=False, num_envs=int(task.get("num_envs") or 64),
            max_iterations=int(task.get("max_iterations") or 20),
        )
    except Exception as exc:  # noqa: BLE001 - keep the queue for a retry/report
        print(f"task failed: {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    if rc == 0:
        PENDING_TASK.unlink(missing_ok=True)
        print("pending task cleared")
    return rc


def _console_api_config(config: LifecycleConfig) -> ApiConfig:
    """A console API view of the same login, for standard-instance actions."""
    return ApiConfig(backend="browser", app_version=config.api.app_version,
                     token_provider=lambda: browser_token(config))


def standard_power_on(config: LifecycleConfig, instance_uuid: str) -> bool:
    """Try a GPU power-on; False means the host has no free card right now."""
    api = _console_api_config(config)
    api.instance_uuid = instance_uuid
    try:
        api_power_on(api)
        return True
    except LifecycleError as exc:
        print(f"{instance_uuid}: power-on refused: {str(exc)[:160]}")
        return False


def _wait_standard_running(config: LifecycleConfig, instance_uuid: str,
                           timeout_s: float = 300.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rows = standard_instances(config)
        row = next((candidate for candidate in rows
                    if candidate.get("uuid") == instance_uuid), None)
        status = row.get("status") if row else "missing"
        if status == "running":
            return True
        if status == "missing":
            return False
        time.sleep(8.0)
    return False


def task_worker(config: LifecycleConfig, *, background: bool = False) -> int:
    """Run the queued task on an existing standard instance before anything else.

    "Reuse what is already rented" is the first rule: a running instance is used
    as is, and each shut-down standard instance is offered a GPU power-on before
    the Pro path is considered. Only when none of them can run does it fall back
    to the low-end Pro worker.
    """
    task = read_pending_task()
    if task is None:
        print("no pending training task; nothing to do")
        return 0
    if background:
        log_path = REPO / "outputs" / "autodl" / "task-worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "task-worker"],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"task worker started in the background; log {log_path}")
        return 0
    rows = standard_instances(config)
    running = [row for row in rows if row.get("status") == "running"]
    target = str(running[0].get("uuid")) if running else ""
    if target:
        print(f"using running standard instance {target}")
    else:
        for row in rows:
            if row.get("status") != "shutdown":
                continue
            uuid = str(row.get("uuid"))
            print(f"trying to start existing standard instance {uuid}")
            if standard_power_on(config, uuid) and _wait_standard_running(config, uuid):
                target = uuid
                print(f"started {uuid}")
                break
    if target:
        try:
            rc = adopt_instance(
                config, task=str(task.get("task")), max_hours=task.get("max_hours"),
                shutdown=False, instance_uuid=target,
                num_envs=int(task.get("num_envs") or 64),
                max_iterations=int(task.get("max_iterations") or 20),
            )
        except Exception as exc:  # noqa: BLE001 - report and fall through
            print(f"standard task failed: {type(exc).__name__}: {str(exc)[:300]}")
            rc = 1
        if rc == 0:
            PENDING_TASK.unlink(missing_ok=True)
            print("pending task cleared")
            return 0
    print("no standard instance could run the task; falling back to Pro low-end stock")
    return pro_worker(config)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autodl_lifecycle",
        description="Start the AutoDL instance for a training run and shut it down "
                    "after every run is fetched and verified.",
    )
    parser.add_argument("--config", help=f"lifecycle TOML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--remote-config", help=f"training TOML (default: {DEFAULT_REMOTE_CONFIG})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="instance status, SSH reachability and running jobs")
    up = sub.add_parser("up", help="power the instance on and wait for SSH")
    up.add_argument("--manual", action="store_true", help="wait for a human-started box")
    up.add_argument("--retry-minutes", type=float, default=0.0,
                    help="keep retrying a refused power-on (e.g. no free GPU) for this long")
    up.add_argument("--rent-new", action="store_true",
                    help="if the old host has no card, scan the market and show the "
                         "replacement plan (creating one is a separate, explicit step)")
    up.add_argument("--nocard", action="store_true",
                    help="power on in the console's no-card mode (0.5 CPU / 2GB RAM, "
                         "￥0.10/h) for CPU smoke tests instead of waiting for a GPU")
    up.add_argument("--dry-run", action="store_true")
    down = sub.add_parser("down", help="verify the sync gate, then power the instance off")
    down.add_argument("--force", action="store_true", help="ignore running jobs and sync gaps")
    down.add_argument("--fetch-missing", action="store_true",
                      help="fetch any terminal run that is not local before shutting down")
    down.add_argument("--dry-run", action="store_true", help="check the gate, do not shut down")
    sub.add_parser("verify", help="run the sync gate only")
    sub.add_parser("discover", help="list the account's instances and the one matching the box")
    market = sub.add_parser("market", help="scan the compute market (read-only)")
    market.add_argument("--gpu", help="comma-separated GPU types (default: autodl.gpu_types)")
    market.add_argument("--max-price", type=float, default=None,
                        help="¥/hour ceiling (default: autodl.max_price_cny)")
    market.add_argument("--min-idle", type=int, default=1)
    market.add_argument("--limit", type=int, default=20)
    market.add_argument("--json", action="store_true")
    sub.add_parser("pro-plan", help="show what the Pro API would create (read-only)")
    queue = sub.add_parser("queue-task", help="queue one training task for the Pro worker")
    queue.add_argument("--task", default="jumper.flat")
    queue.add_argument("--max-hours", type=float, default=None)
    queue.add_argument("--num-envs", type=int, default=64)
    queue.add_argument("--max-iterations", type=int, default=20)
    queue.add_argument("--submit-args", default="")
    worker = sub.add_parser("pro-worker", help="run the queued task if Pro stock exists")
    worker.add_argument("--background", action="store_true")
    tasker = sub.add_parser("task-worker", help="run the queued task on an existing instance first")
    tasker.add_argument("--background", action="store_true")
    adopt = sub.add_parser("adopt", help="take over a hand-created standard instance")
    adopt.add_argument("--task", default="jumper.flat")
    adopt.add_argument("--max-hours", type=float, default=None)
    adopt.add_argument("--shutdown", action="store_true",
                       help="shut the standard instance down after the smoke run")
    check = sub.add_parser("market-check", help="one read-only standard-card check")
    check.add_argument("--gpu", help="comma-separated GPU types")
    check.add_argument("--min-idle", type=int, default=1)
    check.add_argument("--no-notify", action="store_true")
    check.add_argument("--force", action="store_true",
                       help="run even when no training task is queued")
    auth = sub.add_parser(
        "browser-auth",
        help="open a browser, wait for a human AutoDL login, and keep the session",
    )
    auth.add_argument("--wait-seconds", type=float, default=180.0,
                      help="how long to wait for the login to complete")
    session = sub.add_parser("session", help="up -> submit -> wait -> fetch -> verify -> down")
    session.add_argument("--task", required=True)
    session.add_argument("--release", help="existing release id (default: sync and use the new one)")
    session.add_argument("--submit-args", default="", help="extra scripts/train.py arguments")
    session.add_argument("--max-hours", type=float, default=None)
    session.add_argument("--no-sync", action="store_true", help="do not sync before submitting")
    session.add_argument("--post", action="store_true", help="run the post-fetch pipeline on fetch")
    session.add_argument("--retry-minutes", type=float, default=0.0,
                         help="wait this long for a free GPU when powering on")
    session.add_argument("--rent-new", action="store_true",
                         help="show the replacement-instance plan if the old host has no card")
    session.add_argument("--auto-shutdown", action="store_true",
                         help="power off after the session (default: leave it running)")
    session.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.remote_config)
    if config.api.backend == "browser":
        config.api.token_provider = lambda: browser_token(config)
    try:
        if args.command == "status":
            module = _remote_train()
            remote = module.Remote(config.ssh)
            if config.api.backend == "browser":
                profile = config.api.browser_profile or DEFAULT_BROWSER_PROFILE
                print(f"browser profile: {profile} "
                      f"({'exists' if profile.is_dir() else 'missing; run browser-auth'})")
            if config.api.backend != "manual":
                try:
                    config.api.instance_uuid = resolve_instance_uuid(config)
                    print(f"api status: {api_status(config.api)}")
                    print(f"api instance: {config.api.instance_uuid}")
                except LifecycleError as exc:
                    print(f"api status: unavailable ({exc})")
            print(f"ssh: {config.ssh.hostname}:{config.ssh.port}")
            print(f"ssh reachable: {_ssh_ready(remote, attempts=1)}")
            try:
                alive = running_jobs(config, remote)
                print(f"running jobs: {', '.join(alive) if alive else 'none'}")
            except Exception as exc:  # noqa: BLE001
                print(f"running jobs: unknown ({type(exc).__name__})")
            return 0
        if args.command == "up":
            return up(config, manual=args.manual, dry_run=args.dry_run,
                      retry_minutes=args.retry_minutes, rent_new=args.rent_new,
                      nocard=args.nocard)
        if args.command == "down":
            return down(config, force=args.force, fetch_missing=args.fetch_missing,
                        dry_run=args.dry_run)
        if args.command == "verify":
            return verify(config)
        if args.command == "browser-auth":
            return browser_auth(config, wait_s=args.wait_seconds)
        if args.command == "adopt":
            return adopt_instance(config, task=args.task, max_hours=args.max_hours,
                                  shutdown=args.shutdown)
        if args.command == "market-check":
            gpus = tuple(part.strip() for part in (args.gpu or "").split(",")
                         if part.strip())
            return market_check(config, gpus=gpus, min_idle=args.min_idle,
                                notify=not args.no_notify, force=args.force)
        if args.command == "queue-task":
            path = queue_task(args.task, max_hours=args.max_hours,
                              num_envs=args.num_envs,
                              max_iterations=args.max_iterations,
                              submit_args=args.submit_args)
            print(f"queued {args.task} -> {path}")
            return 0
        if args.command == "task-worker":
            return task_worker(config, background=args.background)
        if args.command == "pro-worker":
            return pro_worker(config, background=args.background)
        if args.command == "pro-plan":
            print(format_pro_plan(pro_plan(config)))
            return 0
        if args.command == "adopt":
            return adopt_instance(config, task=args.task, max_hours=args.max_hours,
                                  shutdown=args.shutdown)
        if args.command == "market-check":
            gpus = tuple(part.strip() for part in (args.gpu or "").split(",")
                         if part.strip())
            return market_check(config, gpus=gpus, min_idle=args.min_idle,
                                notify=not args.no_notify, force=args.force)
        if args.command == "queue-task":
            path = queue_task(args.task, max_hours=args.max_hours,
                              num_envs=args.num_envs,
                              max_iterations=args.max_iterations,
                              submit_args=args.submit_args)
            print(f"queued {args.task} -> {path}")
            return 0
        if args.command == "task-worker":
            return task_worker(config, background=args.background)
        if args.command == "pro-worker":
            return pro_worker(config, background=args.background)
        if args.command == "pro-plan":
            print(format_pro_plan(pro_plan(config)))
            return 0
        if args.command == "market":
            gpu_types = tuple(
                part.strip() for part in (args.gpu or "").split(",") if part.strip()
            )
            plan = market_plan(config, gpu_types=gpu_types,
                               max_price_cny=args.max_price, min_idle=args.min_idle)
            if args.json:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
            else:
                print(format_market(plan["machines"], limit=args.limit))
                print(format_market_plan(plan))
            return 0
        if args.command == "discover":
            rows = api_list(config.api)
            print(f"{'uuid':<28}{'status':<10}{'ssh_port':<10}{'proxy_host':<36}name")
            for row in rows:
                uuid = str(row.get("uuid", ""))
                status = str(row.get("status", ""))
                port = str(row.get("ssh_port", ""))
                proxy = str(row.get("proxy_host", ""))
                print(f"{uuid:<28}{status:<10}{port:<10}{proxy:<36}"
                      f"{row.get('instance_name', '')}")
            matches = match_instance(rows, config.ssh, getattr(config.ssh, "hostname", ""))
            if len(matches) == 1:
                print(f"match for ssh port {config.ssh.port}: {matches[0].get('uuid')}")
            else:
                print(f"{len(matches)} instance(s) match ssh port {config.ssh.port}; "
                      "set autodl.instance_uuid explicitly")
            return 0
        if args.command == "session":
            return session(
                config, task=args.task, release=args.release, submit_args=args.submit_args,
                max_hours=args.max_hours, sync=not args.no_sync, post=args.post,
                dry_run=args.dry_run, retry_minutes=args.retry_minutes,
                rent_new=args.rent_new, auto_shutdown=args.auto_shutdown,
            )
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
