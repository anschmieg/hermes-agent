"""Dependency-free native A2A peer-isolation smoke tests.

Run in an isolated Python interpreter (not imported by pytest):
    python tests/scripts/verify_native_a2a_peer_scope_stdlib.py

The environment shims below replace only unrelated gateway dependencies.  The
REAL plugins.platforms.a2a.protocol and adapter are imported and exercised.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _module(name: str, package: bool = False):
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = []
    sys.modules[name] = mod
    return mod


_module("gateway", True)
_module("gateway.platforms", True)
shared = _module("gateway.platforms._shared")
shared.coerce_port = lambda value, default: int(value) if str(value).isdigit() else default
shared.get_scoped_secret = lambda name: None
shared.profile_scoped = lambda: False
base = _module("gateway.platforms.base")
base.BasePlatformAdapter = type("BasePlatformAdapter", (), {})
base.SendResult = type("SendResult", (), {})
event = _module("gateway.platforms.event")
for name in ("MessageEvent", "MessageType", "ProcessingOutcome"):
    setattr(event, name, type(name, (), {}))
config = _module("gateway.config")
config.Platform = type("Platform", (), {})
_module("agent", True)
proxy = _module("agent.proxy_bypass")
proxy.is_loopback_host = lambda host: host in {"localhost", "127.0.0.1", "::1"}
constants = _module("hermes_constants")
constants.get_hermes_home = lambda: Path(tempfile.gettempdir())

from plugins.platforms.a2a import adapter as native_adapter  # noqa: E402
from plugins.platforms.a2a import protocol  # noqa: E402


class NativePeerOwnership(unittest.TestCase):
    def setUp(self):
        self.adapter = native_adapter.A2AAdapter.__new__(native_adapter.A2AAdapter)
        self.adapter.tasks = protocol.TaskStore()
        self.agent = {"slug": "research", "tenant": "tenant-1"}
        self.adapter._agents = {"": self.agent}
        self.adapter.tasks.create("t1", "shared-context", "alice", "research", "tenant-1")
        self.adapter.tasks.set_state("t1", protocol.STATE_WORKING)

    def test_native_http_dispatch_binds_authenticated_identity(self):
        source = inspect.getsource(native_adapter.A2ARequestHandler.do_POST)
        self.assertIn("peer=identity", source)

    def test_get_watch_and_list_deny_foreign_peer(self):
        params = {"taskId": "t1"}
        self.assertEqual(self.adapter._rpc_tasks_get(1, params, self.agent, "alice")["result"]["id"], "t1")
        self.assertEqual(self.adapter._rpc_tasks_get(2, params, self.agent, "bob")["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertIsNone(self.adapter.tasks.watch("t1", "research", "tenant-1", peer="bob"))
        self.assertEqual(self.adapter._rpc_tasks_list(3, {}, self.agent, "bob")["result"]["tasks"], [])
        self.assertEqual(self.adapter._rpc_tasks_list(4, {}, self.agent, "alice")["result"]["totalSize"], 1)

    def test_cancel_never_fakes_backend_stop(self):
        params = {"taskId": "t1"}
        self.assertEqual(self.adapter._rpc_tasks_cancel(1, params, self.agent, "bob")["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(self.adapter._rpc_tasks_cancel(2, params, self.agent, "alice")["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        self.assertEqual(self.adapter.tasks.get("t1")["state"], protocol.STATE_WORKING)

    def test_push_config_is_peer_scoped(self):
        params = {"taskId": "t1"}
        push = {**params, "pushNotificationConfig": {"url": "https://example.org/callback"}}
        self.assertEqual(self.adapter._rpc_push_config_create(1, push, self.agent, "bob")["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(self.adapter._rpc_push_config_create(2, push, self.agent, "alice")["result"]["taskId"], "t1")
        self.assertEqual(self.adapter._rpc_push_config_get(3, params, self.agent, "bob")["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(self.adapter._rpc_push_config_list(4, params, self.agent, "bob")["result"]["configs"], [])
        self.assertEqual(self.adapter._rpc_push_config_delete(5, params, self.agent, "bob")["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(self.adapter._rpc_push_config_get(6, params, self.agent, "alice")["result"]["taskId"], "t1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
