"""A2A inbound adapter: stdlib http.server (daemon thread) serving the Agent Card, /metrics and
JSON-RPC (message/send, message/stream SSE, tasks/*, push-config CRUD). Inbound tasks are framed
(security.wrap_inbound) and routed into the LIVE gateway session; ``send()`` fulfils the per-task
Future the HTTP handler blocks on. No token configured => binds 127.0.0.1 only."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import http.client
import json
import logging
import os
import re
import socket
import ssl
import threading
import time
import urllib.parse
from collections import OrderedDict, deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from pathlib import Path

from hermes_constants import get_hermes_home

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.config import Platform
from gateway.platforms._shared import coerce_port as _to_int, get_scoped_secret as _get_scoped_secret

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9900
# seconds: orphan grace floor / ceiling / watchdog period. The ceiling keeps the sweep
# meaningful when A2A_REPLY_TIMEOUT is absurd (1e18 would never fail an orphan).
_MIN_ORPHAN_TIMEOUT, _MAX_ORPHAN_TIMEOUT, _WATCHDOG_INTERVAL = 300, 86400, 60
_MAX_BODY = 1_048_576  # 1MB max request body — prevents DoS via memory exhaustion
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments
_DEFAULT_DESCRIPTION = "Hermes Agent — a general-purpose agent reachable over A2A."

_ok = protocol.jsonrpc_result
_err = protocol.jsonrpc_error

# (adapter handler, v1.0 PascalCase method per §5.3/§9.4, *legacy slash aliases still accepted)
_METHOD_TABLE = (
    ("_rpc_message_send", "SendMessage", "message/send"),
    ("_rpc_message_stream", "SendStreamingMessage", "message/stream"),
    ("_rpc_tasks_get", "GetTask", "tasks/get"),
    # Hermes-only read-only recovery extension (not an A2A v1.0 standard RPC).
    ("_rpc_tasks_get_by_message", "GetTaskByMessageId"),
    ("_rpc_tasks_list", "ListTasks", "tasks/list"),
    ("_rpc_tasks_cancel", "CancelTask", "tasks/cancel"),
    ("_rpc_tasks_subscribe", "SubscribeToTask", "tasks/subscribe"),
    ("_rpc_push_config_create", "CreateTaskPushNotificationConfig", "tasks/pushNotificationConfig/create",
     "tasks/pushNotificationConfig/set", "tasks/pushNotification/set"),
    ("_rpc_push_config_get", "GetTaskPushNotificationConfig", "tasks/pushNotificationConfig/get"),
    ("_rpc_push_config_list", "ListTaskPushNotificationConfigs", "tasks/pushNotificationConfig/list"),
    ("_rpc_push_config_delete", "DeleteTaskPushNotificationConfig", "tasks/pushNotificationConfig/delete"),
)
# JSON-RPC method -> (adapter handler name, is_v1)
_METHODS: dict[str, tuple[str, bool]] = {
    m: (handler, i == 0) for handler, *methods in _METHOD_TABLE for i, m in enumerate(methods)
}


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


def _orphan_timeout() -> float:
    """Orphan grace must never expire before a configured reply window, but stays bounded."""
    return min(float(_MAX_ORPHAN_TIMEOUT), max(float(_MIN_ORPHAN_TIMEOUT), _reply_timeout()))


def _default_agent_name() -> str:
    # Scope-aware: a secondary multiplex profile must not borrow the default profile's A2A_AGENT_NAME.
    name = _get_scoped_secret("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


def _clean_slug(value: str) -> str:
    """URL-safe-ish single-segment slug for a served agent ("" for the root agent)."""
    slug = str(value or "").strip().strip("/")
    return "" if slug in ("", "default", "root") else slug.split("/")[0]


def _join_url(base: str, prefix: str) -> str:
    base = (base or "").strip() or "/"
    base = base if base.endswith("/") else base + "/"
    prefix = (prefix or "").strip("/")
    return urllib.parse.urljoin(base, prefix + "/") if prefix else base


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.getenv("HERMES_PROFILE", "default") or "default"


def _profile_home(profile: str) -> Optional[str]:
    with contextlib.suppress(Exception):
        from hermes_cli.profiles import get_profile_dir
        return str(get_profile_dir(profile))
    if profile and profile != "default":
        return os.path.expanduser(f"~/.hermes/profiles/{profile}")
    with contextlib.suppress(Exception):
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    return None


class _ThreadingHTTPServerV6(ThreadingHTTPServer):
    """IPv6 listener. ``V6ONLY=0`` makes the ``::`` wildcard dual-stack (IPv4 peers arrive as
    ``::ffff:a.b.c.d``), so one socket serves IPv6-only private networks and IPv4 alike."""

    address_family = socket.AF_INET6

    def server_bind(self):
        with contextlib.suppress(OSError, AttributeError):  # no IPV6_V6ONLY: the bind stays v6-only
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def _http_server_class(host: str) -> type:
    """``ThreadingHTTPServer`` is AF_INET only, so an IPv6 bind (``::1``, ``::``) needs the v6 family."""
    return _ThreadingHTTPServerV6 if ":" in host else ThreadingHTTPServer


def _daemon_thread(target, name: str) -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def _safe_context_slug(value: str, max_len: int = 96) -> str:
    """Sanitize attacker-provided context ids before using in session titles."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (slug or "ctx")[:max_len]


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the A2A JSON-RPC surface; all state lives on ``self.server.adapter``."""

    @property
    def adapter(self) -> A2AAdapter:
        return self.server.adapter  # type: ignore[attr-defined]

    def parse_request(self) -> bool:
        """Without a token, refuse any request whose ``Host`` is not a loopback name, before a route
        runs: a page on a hostname re-resolved to 127.0.0.1 (DNS rebinding) is same-origin to this
        listener and could otherwise read its served-agent list and Agent Card."""
        if not super().parse_request():
            return False
        adapter = self.adapter
        if adapter._security_context.localhost_only() and not security.is_loopback_host_header(
                self.headers.get("Host"), adapter._public_url):
            self._error(403, None, protocol.ERR_UNAUTHORIZED,
                        "Host must be a loopback name or the A2A_PUBLIC_URL host (no A2A token configured)")
            return False
        return True

    def log_message(self, format, *args):
        logger.debug("A2A http: " + format, *args)  # silence the default stderr access log

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(body)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, http_code: int, req_id: Any, code: int, message: str):
        self._json(http_code, _err(req_id, code, message))

    def _client_ip(self) -> str:
        """The peer's address; an IPv4 peer of a dual-stack ``::`` bind keeps its plain IPv4 identity."""
        ip = self.client_address[0] if self.client_address else ""
        return ip.removeprefix("::ffff:") if ip.startswith("::ffff:") and "." in ip else ip

    def _request_public_url(self) -> str:
        """A2A_PUBLIC_URL > X-Forwarded-Host / Host (scheme from X-Forwarded-Proto) > "" (bind host).

        Empty means "caller has no info, fall back to bind host". See #41711.

        A2A_PUBLIC_URL is read from ``self.adapter`` (captured at construction time, inside profile
        scope) rather than os.getenv here: do_GET/do_POST run on ThreadingHTTPServer's per-connection
        OS threads, which never inherit the profile scope contextvar, so a secondary multiplex
        profile's own A2A_PUBLIC_URL would otherwise resolve to the default profile's env value.
        """
        explicit = self.adapter._public_url
        if explicit:
            return explicit
        host = (self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")).split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/" if host else ""

    def do_GET(self):
        adapter = self.adapter
        route = adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        public_url = self._request_public_url() or None
        sec = adapter._security_context
        identity = sec.authenticate(self.headers.get("Authorization"), self._client_ip())
        authorized = sec.localhost_only() or (identity is not None and sec.is_trusted_peer(identity))
        # Do not disclose a routed private agent's card, metrics, or existence
        # before checking the same credential and per-agent ACL as /health.
        # Root discovery remains public for clients bootstrapping A2A auth.
        if agent.get("slug") and (not authorized or not adapter._agent_allows_peer(agent, identity or "")):
            return self._json(404, {"error": "not found"})
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            return self._json(200, adapter._build_card(public_url, agent=agent))
        if subpath == "/metrics":
            return self._json(200, protocol.metrics.snapshot())
        if subpath not in ("/", "/health"):
            return self._json(404, {"error": "not found"})
        payload = {"status": "ok", "agent": agent.get("name") or adapter.agent_name}
        if authorized:
            payload["served_agents"] = adapter._served_agent_summary(public_url=public_url, peer=identity or "")
        self._json(200, payload)

    def do_POST(self):
        adapter = self.adapter
        # Identity comes from the credential (or the socket in localhost-only mode) — never the body.
        identity = adapter._security_context.authenticate(self.headers.get("Authorization"), self._client_ip())
        if identity is None:
            return self._error(401, None, protocol.ERR_UNAUTHORIZED, "unauthorized")
        if adapter._security_context.localhost_only() and (refusal := security.browser_post_refusal(self.headers)):
            return self._error(403, None, protocol.ERR_UNAUTHORIZED, refusal)
        try:
            length = int(self.headers.get("Content-Length", 0))
            # Negative values make rfile.read(-1) consume until EOF, allowing
            # one keep-alive request to tie up a thread with unbounded input.
            if length < 0:
                return self._error(400, None, protocol.ERR_PARSE, "invalid content length")
            if length > _MAX_BODY:
                return self._error(413, None, protocol.ERR_PARSE, "payload too large")
            req = json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8"))
        except Exception:
            return self._error(400, None, protocol.ERR_PARSE, "parse error")
        if not isinstance(req, dict):
            return self._error(400, None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object")
        req_id, method = req.get("id"), str(req.get("method", ""))
        params = req["params"] if req.get("params") is not None else {}
        version = (self.headers.get("A2A-Version") or "").strip()
        route = adapter._route_for_request(self.path, params) if isinstance(params, dict) else {}
        handler_name, is_v1 = _METHODS.get(method, ("", False))
        # Ordered, lazily-evaluated checks -> (http status, error code, message); first failure wins
        # (the rate limiter must not be consulted for requests rejected before it).
        checks = (
            (lambda: not isinstance(params, dict), 200, protocol.ERR_INVALID_PARAMS, "params must be an object"),
            (lambda: version and version not in {"1.0", "1.0.0"}, 200, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"),
            (lambda: route.get("error"), 400, protocol.ERR_INVALID_PARAMS, route.get("error")),
            (lambda: not adapter._rate_limiter.allow(identity), 429, protocol.ERR_RATE_LIMITED, "rate limit exceeded"),
            (lambda: not adapter._security_context.is_trusted_peer(identity), 403, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"),
            (lambda: not adapter._agent_allows_peer(route.get("agent") or {}, identity), 403,
             protocol.ERR_UNTRUSTED_PEER, "peer is not authorized for the requested served agent"),
            (lambda: not handler_name, 200, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"),
        )
        for failed, http, code, msg in checks:
            if failed():
                if code == protocol.ERR_RATE_LIMITED:
                    protocol.metrics.rate_limit_triggers += 1
                return self._error(http, req_id, code, msg)
        agent = route["agent"]
        if handler_name == "_rpc_message_send":
            self._json(200, adapter._rpc_message_send(req_id, params, identity, agent=agent, v1_response=is_v1))
        elif handler_name == "_rpc_message_stream":
            adapter._rpc_message_stream(self, req_id, params, identity, agent=agent)
        elif handler_name == "_rpc_tasks_subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent, peer=identity)
        else:  # plain JSON task / push-config queries
            # Always scope task operations by the authenticated transport identity.
            # A URL tenant is routing metadata, never an authorization principal.
            self._json(200, getattr(adapter, handler_name)(req_id, params, agent=agent, peer=identity))


class A2AAdapter(BasePlatformAdapter):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("a2a"))
        extra = getattr(config, "extra", {}) or {}
        # Scope-aware: a secondary multiplex profile must not borrow the default profile's bridged
        # A2A_PORT (falls closed to the module default). advertised_toolsets is deliberately unscoped.
        # (advertised_toolsets has the same env-leak shape but is left unscoped here — see the "Scope note"
        # in this fix's PR description: open PR #98937 is actively rewriting this field's None-vs-empty-list
        # semantics.)
        self._security_context = security.A2ASecurityContext.capture()
        _port_env = _get_scoped_secret("A2A_PORT")
        self.port = int(_port_env or extra.get("port", _DEFAULT_PORT))
        self.host = self._security_context.resolve_bind_host()
        self.agent_name = _default_agent_name()
        configured_toolsets = list(extra.get("advertised_toolsets") or []) or _get_scoped_secret("A2A_ADVERTISED_TOOLSETS", "").split(",")
        self._advertised_toolsets = [t.strip() for t in configured_toolsets if str(t).strip()]
        self._active_profile = _active_profile_name()
        # Captured here (construction runs inside _profile_runtime_scope), not read at request time:
        # do_GET/do_POST run on ThreadingHTTPServer's per-connection OS threads, which never inherit
        # the profile scope contextvar (same class as A2A_PORT above).
        self._public_url = _get_scoped_secret("A2A_PUBLIC_URL", "").strip()
        self._agents = self._load_served_agents(extra)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._server_thread = self._watchdog_thread = None  # type: Optional[threading.Thread]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._watchdog_stop = threading.Event()
        # Per-adapter protocol state (not module-global).
        # Capture the per-profile home while its scope is active. No request
        # thread may choose a ledger from ambient/global environment later.
        ledger_path = Path(_get_scoped_secret("A2A_TASK_LEDGER_PATH", "") or
                           (get_hermes_home() / "a2a_native_tasks.sqlite3"))
        self.tasks = protocol.TaskStore(protocol.DurableTaskLedger(ledger_path))
        self._turns, self._rate_limiter = protocol.TurnTracker(), protocol.RateLimiter()
        # Pending reply futures: task_id -> (context_id, Future). _pending_order keeps per-context
        # FIFO so adapter.send() — which only knows the context — resolves the oldest task.
        self._pending: dict[str, tuple[str, Future]] = {}
        self._pending_order: dict[str, deque[str]] = {}
        # Request ownership outlives reply Futures and also covers synchronous profile forwards.
        self._active_tasks: set[str] = set()
        # Admission ids this process already forwarded; a resend of one is a retry, not a new turn.
        self._forwarded_inputs: OrderedDict[str, None] = OrderedDict()
        self._pending_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "A2A"

    @property
    def authorization_is_upstream(self) -> bool:
        """Requests are authenticated in ``do_POST``; the gateway's A2A_ALLOWED_USERS list would
        otherwise reject peers (identity is a token-derived name or IP). Wrong credentials still 401.

        This is authorization delegated to the A2A bearer-token transport, not a fail-open: every request is
        401'd if the credential is wrong. Reported by kuangmi-bit (PR #41711 comment, Jun 27).
        """
        return True

    async def connect(self, **_kwargs) -> bool:
        # Capture the gateway loop so the HTTP thread can marshal events via run_coroutine_threadsafe.
        self._loop = asyncio.get_running_loop()
        try:
            self._httpd = _http_server_class(self.host)((self.host, self.port), A2ARequestHandler)
        except OSError as e:
            logger.error("A2A: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"A2A bind failed: {e}", retryable=True)
            return False
        self._httpd.daemon_threads = True
        self._httpd.adapter = self  # type: ignore[attr-defined]
        self._server_thread = _daemon_thread(self._httpd.serve_forever, "a2a-http")
        self._watchdog_stop.clear()  # disconnect sets it; reset for reconnection
        self._watchdog_thread = _daemon_thread(self._watchdog_loop, "a2a-watchdog")
        self._mark_connected()
        logger.info("A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)", self.host, self.port,
                    "localhost-only" if self._security_context.localhost_only() else "REMOTE (bearer auth)", self.agent_name, len(self._agents))
        sec = self._security_context
        if sec.dispatch_fails_closed():
            logger.error("A2A: exposed on non-loopback bind %s with no A2A_TRUSTED_PEERS; every dispatch will be refused (403). "
                         "Set A2A_TRUSTED_PEERS, or A2A_ALLOW_ALL_USERS=true for a trusted network.", self.host)
        if sec.bearer_token and not sec.peer_tokens:
            logger.warning("A2A: A2A_BEARER_TOKEN authenticates all callers as one shared principal; "
                           "configure unique A2A_PEER_TOKENS for independently isolated remote peers")
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        if self._httpd is not None:
            with contextlib.suppress(Exception):
                self._httpd.shutdown()
                self._httpd.server_close()
            self._httpd = None
        # Fail any in-flight replies so blocked HTTP threads don't hang.
        with self._pending_lock:
            for tid in list(self._pending):
                self._resolve_locked(tid, protocol.STATE_FAILED, "[agent shutting down]")
            self._pending.clear()
            self._pending_order.clear()
            self._active_tasks.clear()

    def _watchdog_loop(self) -> None:
        """Background thread that fails orphaned tasks (keeps them queryable)."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                self._fail_orphans_once()
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    def _fail_orphans_once(self) -> list[str]:
        """Fail stale tasks that no request still owns."""
        with self._pending_lock:
            active_tasks = set(self._active_tasks)
        timeout = _orphan_timeout()
        failed = self.tasks.fail_orphans(timeout, exclude=active_tasks)
        for tid in failed:
            logger.warning("A2A: orphaned task %s marked failed (timeout %gs)", tid, timeout)
            protocol.metrics.tasks_failed += 1
        return failed

    def _load_served_agents(self, extra: dict) -> dict[str, dict]:
        """Served-agent routing from ``platforms.a2a.extra.agents`` (top-level ``a2a_served_agents``
        fallback for scripts/tests). Root/default always maps to the live gateway session."""
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            try:
                from hermes_cli.config import load_config
                cfg = load_config() or {}
            except Exception:
                cfg = {}
            cfg = cfg if isinstance(cfg, dict) else {}
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")
        # Scope-aware like port: a secondary profile must not inherit A2A_AGENT_DESCRIPTION.
        default_desc = _get_scoped_secret("A2A_AGENT_DESCRIPTION", _DEFAULT_DESCRIPTION)
        agents: dict[str, dict] = {"": {
            "slug": "", "path": "", "tenant": "", "profile": self._active_profile, "local": True,
            "name": self.agent_name, "description": default_desc, "advertised_toolsets": self._advertised_toolsets,
        }}
        reserved = {"health", "metrics", ".well-known"}
        tenants: dict[str, str] = {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw or []) if isinstance(raw, list) else []
        for key, val in items:
            if not isinstance(val, dict):
                continue
            slug = _clean_slug(str(val.get("slug") or val.get("id") or key))
            if not slug:
                continue
            path_segment = _clean_slug(str(val.get("path") or slug))
            if not path_segment or path_segment in reserved:
                logger.warning("A2A: ignoring served agent %r with reserved/invalid path %r", slug, path_segment)
                continue
            profile = str(val.get("profile") or slug).strip()
            toolsets = val.get("advertised_toolsets") or val.get("toolsets") or val.get("capabilities") or []
            if isinstance(toolsets, str):
                toolsets = [t.strip() for t in toolsets.split(",") if t.strip()]
            tenant = str(val.get("tenant") or slug).strip()
            allowed = val.get("allowed_peers")
            if isinstance(allowed, str):
                allowed = [p.strip() for p in allowed.split(",") if p.strip()]
            if allowed is not None and not isinstance(allowed, (list, tuple, set)):
                logger.warning("A2A: ignoring invalid allowed_peers for served agent %r", slug)
                continue
            if tenant:
                if tenant in tenants:
                    logger.warning("A2A: ignoring served agent %r with duplicate tenant %r already used by %r",
                                   slug, tenant, tenants[tenant])
                    continue
                tenants[tenant] = slug
            agents[slug] = {
                "slug": slug, "path": "/" + path_segment, "tenant": tenant, "profile": profile or slug,
                "local": bool(val.get("local")) or profile in ("", "default", self._active_profile),
                "name": str(val.get("name") or f"Hermes {slug}"),
                "description": str(val.get("description") or f"Hermes profile '{profile or slug}' exposed over A2A."),
                "advertised_toolsets": list(toolsets or []),
                # Explicit ACL required for remotely exposed non-root tenants.
                "allowed_peers": frozenset(str(p).strip() for p in allowed if str(p).strip()) if allowed is not None else None,
                "timeout": int(val.get("timeout") or _reply_timeout()),
            }
        return agents

    def _base_url(self, public_url: Optional[str]) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host  # IPv6 literal
        return (public_url or "").strip() or f"http://{host}:{self.port}/"

    def _served_agent_summary(self, public_url: Optional[str] = None, peer: str = "") -> list[dict]:
        """Expose only agents visible to this peer, never private profile or ACL metadata."""
        base = self._base_url(public_url)
        return [{"slug": a["slug"] or "default", "name": a.get("name"), "url": _join_url(base, a.get("path", ""))}
                for a in self._agents.values() if self._agent_allows_peer(a, peer)]

    def _route_for_path(self, raw_path: str) -> dict:
        path = urllib.parse.urlsplit(raw_path or "/").path or "/"
        # Longest prefix wins. Default/root agent is the fallback.
        for agent in sorted(self._agents.values(), key=lambda a: len(a.get("path", "")), reverse=True):
            if (prefix := agent.get("path", "") or "") and (path == prefix or path.startswith(prefix + "/")):
                return {"agent": agent, "subpath": path[len(prefix):] or "/"}
        return {"agent": self._agents[""], "subpath": path}

    def _route_for_request(self, raw_path: str, params: dict) -> dict:
        route = self._route_for_path(raw_path)
        agent = route["agent"]
        tenant = str((params or {}).get("tenant") or "")
        # If no URL prefix chose a non-default agent, allow v1.0 tenant routing.
        if agent.get("slug") == "" and tenant and (matches := [a for a in self._agents.values() if a.get("tenant") == tenant]):
            agent = matches[0]
            route = {"agent": agent, "subpath": route["subpath"]}
        expected = str(agent.get("tenant") or "")
        if tenant and expected and tenant != expected:
            return {"error": f"tenant {tenant!r} does not match routed agent {agent.get('slug') or 'default'}"}
        return route

    def _build_card(self, public_url: Optional[str] = None, agent: Optional[dict] = None) -> dict:
        # Per-request public URL beats the bind host so peers behind a reverse proxy can call back.
        agent = agent or self._agents[""]
        return protocol.build_agent_card(
            name=agent.get("name") or self.agent_name, url=_join_url(self._base_url(public_url), agent.get("path", "")),
            description=agent.get("description") or _DEFAULT_DESCRIPTION, skills=self._advertised_skills(agent),
            streaming=bool(agent.get("local", True)), push_notifications=True,
            auth_required=not self._security_context.localhost_only(), tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
        """Agent Card skills from the live tool registry, restricted by ``advertised_toolsets``;
        static fallback without a registry."""
        configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
        try:
            from tools.registry import registry as tool_registry
            allowed = set(configured or []) or None
            mapping = {n: tool_registry.get_tool_names_for_toolset(n)
                       for n in tool_registry.get_registered_toolset_names() if allowed is None or n in allowed}
            if mapping:
                return protocol.skills_from_toolsets(mapping)
        except Exception:
            logger.debug("A2A: tool registry unavailable for Agent Card", exc_info=True)
        return protocol.skills_from_toolsets(configured or [])

    def _add_pending(self, task_id: str, context_id: str) -> Future:
        fut: Future = Future()
        with self._pending_lock:
            self._active_tasks.add(task_id)
            self._pending[task_id] = (context_id, fut)
            self._pending_order.setdefault(context_id, deque()).append(task_id)
        return fut

    def _activate_task(self, task_id: str) -> None:
        with self._pending_lock:
            self._active_tasks.add(task_id)

    _MAX_FORWARDED_INPUTS = 1000

    def _note_forwarded_input(self, input_id: str) -> bool:
        """Record a forwarded admission id; True when this process already sent it (a retry)."""
        with self._pending_lock:
            seen = input_id in self._forwarded_inputs
            self._forwarded_inputs[input_id] = None
            self._forwarded_inputs.move_to_end(input_id)
            while len(self._forwarded_inputs) > self._MAX_FORWARDED_INPUTS:
                self._forwarded_inputs.popitem(last=False)
        return seen

    def _forget_forwarded_input(self, input_id: Optional[str]) -> None:
        """Drop a fresh id a pre-admission guard refused, so its resend is refused again."""
        with self._pending_lock:
            self._forwarded_inputs.pop(input_id, None)

    def _pop_pending(self, task_id: str) -> None:
        with self._pending_lock:
            self._active_tasks.discard(task_id)
            entry = self._pending.pop(task_id, None)
            order = self._pending_order.get(entry[0]) if entry else None
            if order and task_id in order:
                order.remove(task_id)
            if order is not None and not order:
                self._pending_order.pop(entry[0], None)

    def _resolve_locked(self, task_id: str, state: str, text: str) -> bool:
        entry = self._pending.get(task_id)
        if not entry or entry[1].done():
            return False
        entry[1].set_result((state, text))
        return True

    def _resolve_task(self, task_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            return self._resolve_locked(task_id, state, text)

    def _resolve_oldest_for_context(self, context_id: str, state: str, text: str,
                                    *, reply_to: Optional[str] = None) -> Optional[str]:
        """Resolve the exact inbound task, never a different task's late reply.

        The gateway carries MessageEvent.message_id back as ``reply_to`` on a
        final send.  An old/canceled task may reply after a newer task began in
        the same context; FIFO would irrevocably complete the newer task with
        the stale result.  Legacy unkeyed sends are safe only with *one*
        unresolved task in the context.  Ambiguity must fail closed.
        """
        with self._pending_lock:
            if reply_to is not None:
                task_id = str(reply_to)
                pending = self._pending.get(task_id)
                if pending and pending[0] == context_id and self._resolve_locked(task_id, state, text):
                    return task_id
                return None
            unresolved = [tid for tid in self._pending_order.get(context_id, ())
                          if tid in self._pending and not self._pending[tid][1].done()]
            if len(unresolved) == 1 and self._resolve_locked(unresolved[0], state, text):
                return unresolved[0]
            return None

    def _scope_for_agent(self, agent: Optional[dict]) -> tuple[str, str]:
        return tuple(str((agent or self._agents[""]).get(k) or "") for k in ("slug", "tenant"))

    def _agent_allows_peer(self, agent: dict, peer: str) -> bool:
        """A routed tenant is NOT itself proof that this credential may access it.

        In remote mode a non-root agent needs a per-agent allow-list, unless the
        operator explicitly opted into A2A_ALLOW_ALL_USERS. The localhost-only
        listener remains an intentionally single-machine trust domain.
        """
        if not agent.get("slug"):
            return True
        sec = self._security_context
        if sec.localhost_only() or getattr(sec, "allow_all_users", False):
            return True
        allowed = agent.get("allowed_peers")
        return allowed is not None and peer in allowed

    @staticmethod
    def _execution_context_id(agent: dict, peer: str, external_context_id: str) -> str:
        """Private gateway/session/history key; the caller's contextId is never a local chat id.

        A2A contextIds are chosen by remote callers, so two peers (or served tenants)
        may legitimately submit the same string. Using that string as a Hermes
        chat_id would merge their gateway sessions, pending callbacks and JSONL
        histories. Keep it public in Task receipts but never as a local state key.
        The full identity tuple makes the same peer's conversation stable on restart.
        """
        scope = "\0".join((str(agent.get("slug") or ""), str(agent.get("tenant") or ""),
                            str(peer), str(external_context_id)))
        return "a2a-session-" + hashlib.sha256(scope.encode("utf-8")).hexdigest()

    def _end_task(self, rec: dict, state: str, text: str, stored_reply: Optional[str] = None) -> tuple[dict, None]:
        """Complete a task immediately (rejected / not ready) and build its terminal Task."""
        # Store the explanatory result too: an exact HTTP retry after a crash
        # must receive the same rejection reason as the first attempt.
        # A concurrent duplicate HTTP admission may already have claimed this
        # reservation for backend dispatch. Never overwrite its uncertain
        # WORKING state with a pre-dispatch validation rejection.
        if self.tasks.complete(rec["task_id"], state, text if stored_reply is None else stored_reply,
                               require_reserved=True) is None:
            authoritative = self.tasks.get(rec["task_id"])
            return protocol.TaskStore.to_task(authoritative or rec), None
        protocol.metrics.tasks_failed += state == protocol.STATE_FAILED
        return protocol.build_task(rec["task_id"], rec["context_id"], state, text, created_at=rec["created_iso"]), None

    def _prepare_task(self, params: dict, peer: str, agent: Optional[dict] = None) -> tuple[Optional[dict], Optional[dict]]:
        """Validate, register, and dispatch an inbound message (HTTP worker thread). Returns
        (terminal_task, None) when it ends immediately, else (None, pending) with the future to wait on."""
        agent = agent or self._agents[""]
        text = protocol.extract_text(params)
        forwarded = not agent.get("local", True)
        message_id = protocol.extract_message_id(params)
        if not message_id:
            raise ValueError("A2A message.messageId is required for crash-safe admission")
        context_id = protocol.extract_context_id(params)
        if not context_id:
            # A retry of a first send repeats the messageId and omits the contextId it never received.
            scope = "\0".join((*self._scope_for_agent(agent), peer))
            context_id = protocol.message_context_id(scope, message_id)
        execution_context_id = self._execution_context_id(agent, peer, context_id)
        # The exact durable owner input identity is stable across process restarts.
        scope = "\0".join((*self._scope_for_agent(agent), peer, context_id, message_id))
        input_id = "a2a-msg:" + hashlib.sha256(scope.encode()).hexdigest()
        rec, retry = self.tasks.admit(peer, message_id, params, context_id,
                                      *self._scope_for_agent(agent), input_id=input_id)
        task_id = rec["task_id"]
        if retry and rec.get("effect_state") != "reserved":
            # The original dispatch was accepted or its outcome is uncertain.
            # Never create a second task or re-dispatch on an HTTP retry.
            return protocol.TaskStore.to_task(self.tasks.get(task_id) or rec), None
        # A replay of a never-dispatched reservation can safely claim the work.
        # It is not a new turn against the ping-pong budget.
        turn = 0 if retry else self._turns.track(execution_context_id)
        max_turns = protocol.max_pingpong_turns()
        if turn > max_turns:
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)", context_id, turn, max_turns)
            return self._end_task(rec, protocol.STATE_REJECTED, f"Anti-loop protection: context {context_id} exceeded "
                                  f"{max_turns} turns. Start a new context or increase A2A_MAX_PINGPONG_TURNS.")
        if not text:
            return self._end_task(rec, protocol.STATE_REJECTED, "Empty task — nothing to do.")
        if not forwarded and (self._loop is None or self._message_handler is None):
            return self._end_task(rec, protocol.STATE_FAILED, "Agent gateway not ready to accept A2A tasks.")
        if not self.tasks.claim_dispatch(task_id):
            return protocol.TaskStore.to_task(self.tasks.get(task_id) or rec), None
        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        if not protocol.logged_input(execution_context_id, input_id):
            protocol.persist_message(execution_context_id, "user", text, task_id, input_id=input_id or "")
            protocol.metrics.inbound_total += 1
        self._register_inline_push(task_id, params, agent=agent)
        if forwarded:
            self._activate_task(task_id)
            # Rakazo and other v1.0 clients use returnImmediately to obtain the
            # durable task id before a long-running owner response. The owner
            # admission remains identified by the same input_id on restart.
            config = params.get("configuration") or {}
            if isinstance(config, dict) and config.get("returnImmediately") is True:
                def settle_forward() -> None:
                    try:
                        reply, state = self._forward_to_profile(
                            agent, peer, execution_context_id, framed, input_id=input_id)
                        if state != protocol.STATE_WORKING:
                            self._record_outcome(task_id, context_id, peer, state, reply)
                    except Exception:
                        logger.warning("A2A: forwarded task %s remains uncertain", task_id, exc_info=True)
                    finally:
                        self._pop_pending(task_id)
                threading.Thread(target=settle_forward, name="a2a-forward", daemon=True).start()
                return protocol.TaskStore.to_task(self.tasks.get(task_id) or rec), None
            try:
                reply, state = self._forward_to_profile(agent, peer, execution_context_id, framed, input_id=input_id or task_id)
                if state != protocol.STATE_WORKING:
                    self._record_outcome(task_id, context_id, peer, state, reply)
                return protocol.build_task(task_id, context_id, state, reply, created_at=rec["created_iso"]), None
            finally:
                self._pop_pending(task_id)
        fut = self._add_pending(task_id, execution_context_id)
        event = MessageEvent(text=framed, message_type=MessageType.TEXT, message_id=task_id,
                             source=self.build_source(chat_id=execution_context_id, chat_name=f"a2a:{peer}", chat_type="dm", user_id=peer, user_name=peer))
        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception:
            # The claim is durable; a scheduling exception might race a handler.
            # Do not synthesize a terminal FAILED receipt or replay the request.
            logger.warning("A2A: gateway dispatch outcome unknown for task %s", task_id, exc_info=True)
            self._pop_pending(task_id)
            return protocol.TaskStore.to_task(self.tasks.get(task_id) or rec), None
        return None, {"task_id": task_id, "context_id": context_id, "peer": peer, "future": fut, "created_iso": rec["created_iso"], "started": time.time()}

    def _forward_to_profile(self, agent: dict, peer: str, context_id: str, framed_text: str,
                            *, input_id: Optional[str] = None) -> tuple[str, str]:
        """Forward to the destination owner; never open its canonical database here."""
        from gateway.session_a2a import forward_to_owner
        profile = str(agent.get("profile") or agent.get("slug") or "").strip()
        home = _profile_home(profile)
        if not home:
            return "[profile unavailable]", protocol.STATE_FAILED
        try:
            receipt = asyncio.run(forward_to_owner(
                home, agent=str(agent.get("slug") or profile or "agent"),
                tenant=str(agent.get("tenant") or ""), peer=peer, context_id=context_id,
                input_id=input_id or protocol.new_task_id(), text=framed_text,
                timeout=int(agent.get("timeout") or _reply_timeout())))
        except TimeoutError:
            return "[profile reply unverified; accepted work was not cancelled]", protocol.STATE_WORKING
        except Exception as exc:
            # Boundary for a network peer: the full error stays in the local log, the reply is redacted.
            logger.warning("A2A: dispatch to profile %r unverified", profile, exc_info=True)
            return "[profile dispatch status uncertain; reconcile using the original inputId]", protocol.STATE_WORKING
        result = receipt.get('result') or {}
        completed = receipt.get('outcome') == 'completed' and not result.get('failed')
        if receipt.get("status") == "unknown":
            return "[owner execution unknown; do not re-dispatch]", protocol.STATE_WORKING
        return (security.redact_outbound(result.get('final_response') or ''),
                protocol.STATE_COMPLETED if completed else protocol.STATE_FAILED)

    def _record_outcome(self, task_id: str, context_id: str, peer: str, state: str, reply: str,
                        started: Optional[float] = None) -> bool:
        """Settle once, durably, even when the original HTTP caller is gone."""
        if state == protocol.STATE_WORKING:
            return False
        reply = security.redact_outbound(reply or "")
        stripped = reply.lstrip()
        if state == protocol.STATE_COMPLETED and stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
            state, reply = protocol.STATE_INPUT_REQUIRED, stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()
        # Commit first; duplicate outcomes must not duplicate audit, log or push.
        if self.tasks.complete(task_id, state, reply) is None:
            return False
        rec = self.tasks.get(task_id)
        if rec:
            session_id = self._execution_context_id(
                {"slug": rec.get("agent_slug", ""), "tenant": rec.get("tenant", "")},
                rec.get("peer", peer), rec["context_id"])
            protocol.persist_message(session_id, "agent", reply, task_id)
        security.audit("outbound", peer, task_id, reply)
        m = protocol.metrics
        if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
            m.outbound_total, m.tasks_completed = m.outbound_total + 1, m.tasks_completed + 1
            if started is not None:
                m.record_latency(time.time() - started)
        else:
            m.tasks_failed += 1
        self._send_push_notification(task_id, context_id, reply, state)
        return True

    def _finalize_task(self, pending: dict, state: str, reply: str) -> tuple[str, str]:
        """Record a dispatched task's outcome; returns (state, reply) after redaction and
        input-required detection (a leading marker flags a clarification request)."""
        task_id, context_id, peer = pending["task_id"], pending["context_id"], pending["peer"]
        if state == protocol.STATE_WORKING:
            # A timeout is not a backend failure. Keep the pending future
            # available for a late, durable settlement from send()/completion.
            return state, reply
        try:
            self._record_outcome(task_id, context_id, peer, state, reply, started=pending["started"])
            rec = self.tasks.get(task_id)
            return (rec["state"], rec.get("reply", "")) if rec else (state, reply)
        finally:
            self._pop_pending(task_id)

    @staticmethod
    def _await_future(fut: Future, deadline: float, keepalive, on_timeout: tuple[str, str]) -> tuple[str, str]:
        """Block until ``fut`` resolves or ``deadline`` passes (-> ``on_timeout``). ``keepalive`` runs
        every _SSE_KEEPALIVE seconds while waiting; if it raises, the client is gone and we stop."""
        while True:
            try:
                return fut.result(timeout=_SSE_KEEPALIVE if keepalive else max(0.0, deadline - time.time()))
            except FuturesTimeout:
                if time.time() >= deadline:
                    return on_timeout
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        # A disconnected SSE client does not cancel accepted
                        # backend work. The authoritative task stays WORKING
                        # until an actual callback or owner receipt settles it.
                        return (protocol.STATE_WORKING, "[stream disconnected; backend outcome unverified]")
            except Exception:
                return on_timeout

    def _await_reply(self, pending: dict, keepalive=None) -> tuple[str, str]:
        return self._await_future(pending["future"], pending["started"] + _reply_timeout(), keepalive,
                                  (protocol.STATE_WORKING, "[backend outcome unverified; task remains active]") )

    def _rpc_message_send(self, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None, v1_response: bool = False) -> dict:
        try:
            task, pending = self._prepare_task(params, peer, agent=agent)
        except protocol.IdempotencyConflict:
            return _err(req_id, protocol.ERR_IDEMPOTENCY_CONFLICT, "messageId conflicts with existing reservation")
        except ValueError as exc:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, str(exc))
        if task is None:
            config = params.get("configuration") or {}
            if isinstance(config, dict) and config.get("returnImmediately") is True:
                # The dispatch claim is already committed; callback settlement
                # is independent of this HTTP connection.
                state, reply = protocol.STATE_WORKING, ""
            else:
                state, reply = self._await_reply(pending)
                if state != protocol.STATE_WORKING:
                    state, reply = self._finalize_task(pending, state, reply)
            # Always return the authoritative persisted state (another callback
            # may have settled it after this request started waiting).
            rec = self.tasks.get(pending["task_id"])
            task = protocol.TaskStore.to_task(rec) if rec else protocol.build_task(
                pending["task_id"], pending["context_id"], state, reply)
        return _ok(req_id, protocol.send_message_response(task) if v1_response else task)

    @staticmethod
    def _sse_headers(handler) -> None:
        handler.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache")):
            handler.send_header(k, v)
        handler.end_headers()
        handler.close_connection = True  # v1.0: stream closure signals the terminal state

    @staticmethod
    def _sse_write(handler, chunk: str) -> None:
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    @classmethod
    def _keepalive(cls, handler):
        return lambda: cls._sse_write(handler, ": keepalive\n\n")

    def _emit_terminal(self, handler, task_id: str, context_id: str, state: str, reply: str, req_id: Any = None) -> None:
        """Emit the current artifact/status events. Only signal intentional stream
        completion for terminal/interrupted states, never an uncertain WORKING
        task after an HTTP timeout. ``req_id`` threads into the JSON-RPC SSE
        envelope (§9.4)."""
        completed = bool(reply) and state == protocol.STATE_COMPLETED
        events = ([protocol.artifact_update(task_id, context_id, reply)] if completed else []) + [
            protocol.status_update(task_id, context_id, state, "" if completed else reply)]
        for ev in events:
            self._sse_write(handler, protocol.sse_data(ev, req_id))
        if state in protocol.TERMINAL_STATES or state == protocol.STATE_INPUT_REQUIRED:
            self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(self, handler, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None) -> None:
        """message/stream as an SSE response of JSON-RPC-wrapped StreamResponse events (§9.4)."""
        protocol.metrics.streams_started += 1
        pending = None
        try:
            terminal, pending = self._prepare_task(params, peer, agent=agent)
        except protocol.IdempotencyConflict:
            return handler._json(200, _err(req_id, protocol.ERR_IDEMPOTENCY_CONFLICT,
                                           "messageId conflicts with existing reservation"))
        except ValueError as exc:
            return handler._json(200, _err(req_id, protocol.ERR_INVALID_PARAMS, str(exc)))
        self._sse_headers(handler)
        try:
            if terminal is not None:
                return self._emit_terminal(handler, terminal["id"], terminal["contextId"], terminal["status"]["state"],
                                           protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}), req_id=req_id)
            task_id, context_id = pending["task_id"], pending["context_id"]
            submitted = protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(submitted), req_id))
            self._sse_write(handler, protocol.sse_data(protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))
            state, reply = self._await_reply(pending, keepalive=self._keepalive(handler))
            if state != protocol.STATE_WORKING:
                state, reply = self._finalize_task(pending, state, reply)
                pending = None
            self._emit_terminal(handler, task_id, context_id, state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            # Client loss is transport-only, not proof that accepted work failed.
            logger.debug("A2A: stream client disconnected")

    def _await_subscribed_task(self, task_id: str, fut: Future, agent: Optional[dict],
                               peer: str, keepalive) -> tuple[str, str]:
        """Wait for local notification OR another process's durable SQLite settlement.

        TaskStore.watch() only notifies local Futures. Periodic scoped ledger
        reads observe remote completions without replaying backend work.
        """
        slug, tenant = self._scope_for_agent(agent)
        deadline = time.monotonic() + _reply_timeout()
        last_keepalive = time.monotonic()
        while True:
            rec = self.tasks.get(task_id, slug, tenant, peer)
            if rec is None:
                return protocol.STATE_WORKING, "[task scope changed; backend outcome unverified]"
            state, reply = rec["state"], rec.get("reply", "")
            if state in protocol.TERMINAL_STATES or state == protocol.STATE_INPUT_REQUIRED:
                return state, reply
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return state, reply
            interval = min(0.25, max(0.01, _SSE_KEEPALIVE), remaining)
            if fut.done():
                # An exceptional local wakeup must not cause a busy loop.
                time.sleep(interval)
            else:
                try:
                    fut.result(timeout=interval)
                except FuturesTimeout:
                    pass
                except Exception:
                    # Future errors are not proof of backend settlement.
                    pass
            if keepalive and time.monotonic() - last_keepalive >= _SSE_KEEPALIVE:
                keepalive()  # transport failure does not cancel accepted work
                last_keepalive = time.monotonic()

    def _rpc_tasks_subscribe(self, handler, req_id: Any, params: dict, agent: Optional[dict] = None,
                             peer: str = "") -> None:
        """Reconnect to a task, including durable settlement from another worker."""
        task_id, rec, error = self._find_task(req_id, params, agent, peer)
        if error:
            return handler._json(200, error)
        fut = self.tasks.watch(task_id, *self._scope_for_agent(agent), peer=peer)
        if fut is None:
            return handler._json(200, _err(req_id, protocol.ERR_TASK_NOT_FOUND, "task not found"))
        self._sse_headers(handler)
        try:
            # Initial persisted Task snapshot must precede status updates.
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(
                protocol.TaskStore.to_task(rec)), req_id))
            state, reply = self._await_subscribed_task(task_id, fut, agent, peer, self._keepalive(handler))
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")
        finally:
            self.tasks.unwatch(task_id, fut)

    def _find_task(self, req_id: Any, params: dict, agent: Optional[dict], peer: str = "") -> tuple[str, Optional[dict], Optional[dict]]:
        """(task_id, record, None) for a visible task, else (task_id, None, jsonrpc_error)."""
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent), peer=peer)
        return task_id, rec, None if rec else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")

    def _rpc_tasks_get(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        _task_id, rec, error = self._find_task(req_id, params, agent, peer)
        return error or _ok(req_id, protocol.TaskStore.to_task(rec))

    def _rpc_tasks_get_by_message(self, req_id: Any, params: dict,
                                  agent: Optional[dict] = None, peer: str = "") -> dict:
        message_id = str(params.get("messageId") or "")
        if not message_id:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "messageId required")
        rec = self.tasks.get_by_message(peer, message_id, *self._scope_for_agent(agent))
        if rec is None:
            return _err(req_id, protocol.ERR_TASK_NOT_FOUND, "task not found for messageId")
        return _ok(req_id, protocol.TaskStore.to_task(rec))

    def _rpc_tasks_list(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        offset = _to_int(params.get("pageToken") or 0, 0)
        page_size = _to_int(params.get("pageSize") or 50, 50)
        agent_slug, tenant = self._scope_for_agent(agent)
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""), state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size, offset=max(0, offset), agent_slug=agent_slug, tenant=tenant,
            with_total=True, peer=peer)
        include_artifacts = bool(params.get("includeArtifacts", False))
        return _ok(req_id, {"tasks": [protocol.TaskStore.to_task(r, include_artifacts=include_artifacts) for r in recs],
                            "nextPageToken": str(next_offset) if next_offset else "",
                            "pageSize": max(1, min(page_size, 100)), "totalSize": total})

    def _rpc_tasks_cancel(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        task_id, rec, error = self._find_task(req_id, params, agent, peer)
        if error:
            return error
        if rec["state"] in protocol.TERMINAL_STATES:
            return _err(req_id, protocol.ERR_TASK_NOT_CANCELABLE, f"task {task_id} already {rec['state']}")
        # No backend cancellation / confirmed stop receipt is available.  A
        # superficial state transition can claim CANCELED while work continues.
        # Reject until true backend settlement is implemented.
        return _err(req_id, protocol.ERR_TASK_NOT_CANCELABLE,
                    f"task {task_id} cannot be canceled without backend stop confirmation")

    def _register_inline_push(self, task_id: str, params: dict, agent: Optional[dict] = None) -> None:
        """v1.0: message/send can carry configuration.taskPushNotificationConfig."""
        cfg = (params.get("configuration") or {}).get("taskPushNotificationConfig") or {}
        url = (cfg.get("url") or (cfg.get("pushNotificationConfig") or {}).get("url") or "") if isinstance(cfg, dict) else ""
        if url:
            self.tasks.set_push_config(task_id, str(url), *self._scope_for_agent(agent))

    def _rpc_push_config_create(self, req_id: Any, params: dict, agent: Optional[dict] = None,
                                peer: str = "") -> dict:
        task_id = str(params.get("taskId") or "")
        url = str((params.get("pushNotificationConfig") or params.get("config") or {}).get("url") or "")
        if not task_id or not url:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope_for_agent(agent), peer=peer)
        return _ok(req_id, stored) if stored is not None else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")

    def _push_config_op(self, req_id: Any, params: dict, agent: Optional[dict], op, render,
                        peer: str = "") -> dict:
        """Shared get/list/delete: ``op(task_id, config_id, slug, tenant)`` falsy => not found."""
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        found = op(task_id, str(params.get("id") or params.get("configId") or ""),
                   *self._scope_for_agent(agent), peer=peer)
        return _ok(req_id, render(found)) if found else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"push config not found for task: {task_id}")

    def _rpc_push_config_get(self, req_id: Any, params: dict, agent: Optional[dict] = None,
                             peer: str = "") -> dict:
        return self._push_config_op(req_id, params, agent, self.tasks.get_push_config, lambda cfg: cfg, peer=peer)

    def _rpc_push_config_list(self, req_id: Any, params: dict, agent: Optional[dict] = None,
                              peer: str = "") -> dict:
        # An empty list is a valid (non-error) result, hence the ``or [[]]`` sentinel through the shared op.
        return self._push_config_op(req_id, params, agent,
                                    lambda tid, _cid, *scope, peer="": self.tasks.list_push_configs(tid, *scope, peer=peer) or [[]],
                                    lambda found: {"configs": [c for c in found if c], "nextPageToken": ""}, peer=peer)

    def _rpc_push_config_delete(self, req_id: Any, params: dict, agent: Optional[dict] = None,
                                peer: str = "") -> dict:
        return self._push_config_op(req_id, params, agent, self.tasks.delete_push_config,
                                    lambda _: {"deleted": True}, peer=peer)

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str) -> None:
        """POST a v1.0 StreamResponse payload to the task's registered callback (SSRF-checked URL)."""
        def fail(msg: str, *args) -> None:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s " + msg, task_id, *args)

        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return
        endpoint = security.resolve_safe_callback_target(
            callback_url, localhost_mode=self._security_context.localhost_only())
        if endpoint is None:
            return fail("blocked — unsafe callback destination")
        payload = protocol.status_update(task_id, context_id, state, (reply or "")[:2000])
        headers = {"Content-Type": "application/json"}
        if signature := self._security_context.sign_push_payload(payload):
            headers["X-A2A-Signature"] = signature
        scheme, hostname, port, approved_ip, request_target = endpoint
        # Connect to the vetted IP. urllib.urlopen would resolve the hostname
        # AGAIN (DNS-rebinding SSRF), honor untrusted proxy settings and follow
        # redirects to private addresses. http.client does none of those.
        display_host = f"[{hostname}]" if ":" in hostname else hostname
        headers["Host"] = display_host + (f":{port}" if port != (443 if scheme == "https" else 80) else "")
        connection = (http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection)(
            hostname, port, timeout=10)
        try:
            connection.sock = socket.create_connection((approved_ip, port), timeout=10)
            if scheme == "https":
                # Keep verification and SNI tied to the ORIGINAL authenticated
                # DNS name, not to the resolved address used on the wire.
                connection.sock = ssl.create_default_context().wrap_socket(
                    connection.sock, server_hostname=hostname)
            connection.request("POST", request_target, json.dumps(payload).encode("utf-8"), headers=headers)
            status = connection.getresponse().status
        except Exception as e:
            return fail("failed: %s", e)
        finally:
            connection.close()
        if not 200 <= status < 300:
            return fail("got HTTP %d", status)
        protocol.metrics.push_sent += 1
        logger.debug("A2A: push notification sent for task %s", task_id)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None):
        """Fulfil the task identified by ``reply_to`` (``chat_id`` = private A2A session key).
        Only sends carrying ``metadata['notify']`` (the base adapter's final-reply marker) satisfy
        the caller; progress/status/preview sends must not. Unkeyed sends can
        settle only a single unambiguous pending task."""
        if not (metadata or {}).get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
        else:
            task_id = self._resolve_oldest_for_context(
                chat_id, protocol.STATE_COMPLETED, content or "", reply_to=reply_to)
            if task_id:
                rec = self.tasks.get(task_id)
                if rec:
                    try:
                        await asyncio.to_thread(self._record_outcome, task_id, rec["context_id"],
                                                rec["peer"], protocol.STATE_COMPLETED, content or "")
                    finally:
                        self._pop_pending(task_id)
            else:
                logger.debug("A2A: ignored uncorrelated final send for context %s (reply_to=%s)",
                             chat_id, reply_to)
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Resolve the task future when processing ends without a reply send (failures,
        cancellations, empty runs, or a reply the gateway already streamed to the user) so the
        HTTP thread returns promptly."""
        task_id = str(getattr(event, "message_id", "") or "")
        if task_id:
            # A streamed turn never calls send() with notify=True (the gateway suppresses the
            # normal final send once streaming delivered the body), so the SUCCESS default must
            # not resolve with "" — that strands every A2A streaming reply as an empty completed
            # task (#116944). _streamed_final_response is the same stash _final_text_for_post_turn_hooks
            # reads for /goal and /loop.
            _streamed = getattr(event, "_streamed_final_response", "")
            default = (protocol.STATE_COMPLETED, _streamed if isinstance(_streamed, str) else "")
            state, reply = {
                ProcessingOutcome.FAILURE: (protocol.STATE_FAILED, "[agent processing failed]"),
                ProcessingOutcome.CANCELLED: (protocol.STATE_CANCELED, ""),
            }.get(outcome, default)
            if self._resolve_task(task_id, state, reply):
                rec = self.tasks.get(task_id)
                if rec:
                    try:
                        await asyncio.to_thread(self._record_outcome, task_id, rec["context_id"],
                                                rec["peer"], state, reply)
                    finally:
                        self._pop_pending(task_id)
