"""Native A2A cross-worker SSE and /health ACL regressions (no live inference).

    python tests/scripts/verify_native_a2a_cross_worker.py

Uses actual Hermes adapter/protocol, ephemeral HTTP and a subprocess that
settles the same on-disk SQLite task ledger as a separate worker.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time
import unittest
import urllib.error
import urllib.request

from verify_native_a2a_durable_stdlib import NativeHttpAcceptance, PYTHON
from plugins.platforms.a2a import adapter as native, protocol


class CrossWorkerSubscriptionAndHealth(NativeHttpAcceptance):
    def test_external_process_completion_wakes_subscribed_http_client(self):
        prior_timeout = native._reply_timeout
        prior_handler = self.adapter.handle_message
        native._reply_timeout = lambda: 2.5

        async def accepted_without_local_callback(_event):
            # The owner that admitted and dispatched the task never completes it.
            return None

        self.adapter.handle_message = accepted_without_local_callback
        try:
            params = self.message("other-process-completes")
            params["configuration"] = {"returnImmediately": True}
            task_id = self.post("SendMessage", params)["result"]["task"]["id"]

            with ThreadPoolExecutor(max_workers=1) as pool:
                subscription = pool.submit(self.stream, "SubscribeToTask", {"id": task_id})
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if self.adapter.tasks._watchers.get(task_id):
                        break
                    time.sleep(.01)
                else:
                    self.fail("HTTP subscriber never registered local fast-path watcher")

                writer = (
                    "from plugins.platforms.a2a.ledger import DurableTaskLedger; "
                    "import sys; l=DurableTaskLedger(sys.argv[1]); "
                    "r=l.update(sys.argv[2],state='TASK_STATE_COMPLETED',"
                    "reply='completed by different process',effect_state='settled'); "
                    "assert r is not None, 'remote durable settlement failed'; "
                    "print(r['state'],flush=True)"
                )
                proc = subprocess.run(
                    [PYTHON, "-c", writer, str(self.adapter.tasks._ledger.path), task_id],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(protocol.STATE_COMPLETED, proc.stdout)
                output = subscription.result(timeout=2.0)

            self.assertIn('"task":', output, "SubscribeToTask must start with persisted Task")
            self.assertIn(protocol.STATE_COMPLETED, output)
            self.assertIn("completed by different process", output)
            self.assertIn(": done", output, "durably completed task must close SSE")
            self.assertEqual(self.post("GetTask", {"id": task_id})["result"]["status"]["state"],
                             protocol.STATE_COMPLETED)
            self.assertFalse(self.adapter.tasks._watchers.get(task_id),
                             "remote completion must release process-local waiter")
        finally:
            native._reply_timeout = prior_timeout
            self.adapter.handle_message = prior_handler

    def test_remote_settlement_never_crosses_peer_or_tenant(self):
        self.adapter._agents["dev"] = {
            "slug": "dev", "tenant": "dev", "path": "/dev",
            "profile": "secret-internal-profile", "name": "private-agent",
            "local": True, "allowed_peers": {"alice"},
        }
        prior_handler = self.adapter.handle_message

        async def accepted_without_local_callback(_event):
            return None

        self.adapter.handle_message = accepted_without_local_callback
        try:
            params = self.message("private-other-worker")
            params["configuration"] = {"returnImmediately": True}
            params["tenant"] = "dev"
            task_id = self.post("SendMessage", params, peer="alice")["result"]["task"]["id"]
            for meth_params in ({"id": task_id}, {"id": task_id, "tenant": "dev"}):
                denied = self.post("SubscribeToTask", meth_params, peer="bob")
                self.assertEqual(denied["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
            root_route = self.post("SubscribeToTask", {"id": task_id}, peer="alice")
            self.assertEqual(root_route["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
            # The allowed route is still usable for authenticated subscribers.
            receipt = self.post("GetTask", {"id": task_id, "tenant": "dev"}, peer="alice")
            self.assertEqual(receipt["result"]["id"], task_id)
        finally:
            self.adapter.handle_message = prior_handler

    def test_health_never_exposes_unapproved_agent_or_internal_profile(self):
        self.adapter._agents["dev"] = {
            "slug": "dev", "tenant": "private-tenant", "path": "/dev",
            "profile": "secret-internal-profile", "name": "private-agent",
            "local": False, "allowed_peers": {"alice"},
        }

        def get(path: str, peer: str | None):
            headers = {"Authorization": f"Bearer {peer}-token"} if peer else {}
            req = urllib.request.Request(self.base.rstrip("/") + path, headers=headers)
            with urllib.request.urlopen(req, timeout=6) as response:
                return json.load(response)

        anonymous = get("/health", None)
        self.assertNotIn("served_agents", anonymous)
        bob = get("/health", "bob")
        self.assertNotIn("private-agent", json.dumps(bob))
        self.assertNotIn("secret-internal-profile", json.dumps(bob))
        self.assertNotIn("dev", {r["slug"] for r in bob.get("served_agents", [])})
        with self.assertRaises(urllib.error.HTTPError) as err:
            get("/dev/health", "bob")
        self.assertEqual(err.exception.code, 404)
        alice = get("/health", "alice")
        self.assertIn("dev", {r["slug"] for r in alice["served_agents"]})
        self.assertNotIn("secret-internal-profile", json.dumps(alice))
        self.assertNotIn("private-tenant", json.dumps(alice))
        self.assertEqual(get("/dev/health", "alice")["agent"], "private-agent")


if __name__ == "__main__":
    # The base fixture intentionally contains its own older tests; run only
    # these new adversarial cases, not the inherited suite a second time.
    names = (
        "test_external_process_completion_wakes_subscribed_http_client",
        "test_remote_settlement_never_crosses_peer_or_tenant",
        "test_health_never_exposes_unapproved_agent_or_internal_profile",
    )
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(CrossWorkerSubscriptionAndHealth(name) for name in names))
    raise SystemExit(0 if result.wasSuccessful() else 1)
