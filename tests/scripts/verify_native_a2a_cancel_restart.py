"""Adversarial native A2A cancellation/restart safety checks (stdlib only).

Uses exact Hermes adapter/protocol/ledger with the existing test-only gateway
shims, a private SQLite WAL journal and real SIGKILL. No live dispatch.
Run: python tests/scripts/verify_native_a2a_cancel_restart.py
"""

from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import verify_native_a2a_peer_scope_stdlib as shim  # noqa: F401
import verify_native_a2a_durable_stdlib as http_shim
from plugins.platforms.a2a import adapter, protocol
from plugins.platforms.a2a.ledger import DurableTaskLedger, IdempotencyConflict


class DurableCancelSafety(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "a2a.sqlite3"
        self.alice = {"slug": "private", "tenant": "tenant-a"}
        self.bob_agent = {"slug": "private", "tenant": "tenant-b"}
        self.payload = {"message": {"messageId": "same-id", "role": "ROLE_USER",
                                    "parts": [{"text": "run once"}], "contextId": "same-context"}}
        self.adapter = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
        self.adapter.tasks = protocol.TaskStore(DurableTaskLedger(self.db))

    def admit(self):
        return self.adapter.tasks.admit("alice", "same-id", self.payload,
                                        "same-context", "private", "tenant-a",
                                        input_id="owner:fixed-input")

    def cancel(self, task_id, agent=None, peer="alice"):
        return self.adapter._rpc_tasks_cancel(1, {"taskId": task_id}, agent or self.alice, peer)

    def _claim_then_sigkill(self, task_id):
        code = (
            "import os,signal,sys; from plugins.platforms.a2a.ledger import DurableTaskLedger; "
            "l=DurableTaskLedger(sys.argv[1]); "
            "assert l.update(sys.argv[2],claim=True)['effect_state']=='uncertain'; "
            "os.kill(os.getpid(),signal.SIGKILL)"
        )
        proc = subprocess.run([sys.executable or "/usr/bin/python3", "-c", code, str(self.db), task_id],
                              capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stderr)

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "requires POSIX SIGKILL")
    def test_crash_restart_cancel_is_rejected_without_second_dispatch(self):
        rec, replay = self.admit()
        self.assertFalse(replay)
        self._claim_then_sigkill(rec["task_id"])
        self.adapter.tasks = protocol.TaskStore(DurableTaskLedger(self.db))
        result = self.cancel(rec["task_id"])
        self.assertEqual(result["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        before = self.adapter.tasks.get(rec["task_id"])
        self.assertEqual((before["state"], before["effect_state"], before["input_id"]),
                         (protocol.STATE_WORKING, "uncertain", "owner:fixed-input"))
        same, replay = self.admit()
        self.assertTrue(replay)
        self.assertEqual(same["task_id"], rec["task_id"])
        self.assertFalse(self.adapter.tasks.claim_dispatch(rec["task_id"]))
        self.assertEqual(self.adapter.tasks.get(rec["task_id"])["state"], protocol.STATE_WORKING)

    def test_cancel_is_peer_and_tenant_private_even_when_context_matches(self):
        rec, _ = self.admit()
        task = rec["task_id"]
        for peer, agent in (("bob", self.alice), ("alice", self.bob_agent),
                            ("alice", {"slug": "", "tenant": ""}),
                            ("alice", {"slug": "other", "tenant": "tenant-a"})):
            self.assertEqual(self.cancel(task, agent, peer)["error"]["code"],
                             protocol.ERR_TASK_NOT_FOUND)
            self.assertEqual(self.adapter._rpc_tasks_get(1, {"taskId": task}, agent, peer)
                             ["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(self.cancel(task)["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        bob, replay = self.adapter.tasks.admit("bob", "same-id", self.payload,
                                               "same-context", "private", "tenant-a")
        self.assertFalse(replay)
        self.assertNotEqual(bob["task_id"], task)
        self.assertEqual(self.cancel(bob["task_id"], self.alice, "bob")
                         ["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        with self.assertRaises(IdempotencyConflict):
            self.adapter.tasks.admit("alice", "same-id", self.payload,
                                     "same-context", "private", "tenant-b")

    def test_late_completion_from_other_process_is_immutable_after_cancel(self):
        rec, _ = self.admit()
        task = rec["task_id"]
        self.assertTrue(self.adapter.tasks.claim_dispatch(task))
        self.assertEqual(self.cancel(task)["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        code = (
            "import sys; sys.path.insert(0,sys.argv[3]); "
            "import verify_native_a2a_peer_scope_stdlib; "
            "from plugins.platforms.a2a.ledger import DurableTaskLedger; "
            "from plugins.platforms.a2a.protocol import TaskStore,STATE_COMPLETED; "
            "s=TaskStore(DurableTaskLedger(sys.argv[1])); "
            "assert s.complete(sys.argv[2],STATE_COMPLETED,'owner confirmed result')"
        )
        proc = subprocess.run([sys.executable or "/usr/bin/python3", "-c", code, str(self.db),
                               task, str(Path(__file__).parent)],
                              capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.adapter.tasks = protocol.TaskStore(DurableTaskLedger(self.db))
        self.assertEqual(self.cancel(task)["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        self.assertIsNone(self.adapter.tasks.complete(task, protocol.STATE_CANCELED, "late cancel"))
        self.assertIsNone(self.adapter.tasks.complete(task, protocol.STATE_FAILED, "late error"))
        final = self.adapter.tasks.get(task, "private", "tenant-a", "alice")
        self.assertEqual((final["state"], final["reply"], final["effect_state"]),
                         (protocol.STATE_COMPLETED, "owner confirmed result", "settled"))

    def test_concurrent_cancel_calls_cannot_override_success(self):
        rec, _ = self.admit()
        task = rec["task_id"]
        self.assertTrue(self.adapter.tasks.claim_dispatch(task))
        def reject():
            result = self.cancel(task)
            return result["error"]["code"]
        with ThreadPoolExecutor(max_workers=12) as pool:
            pending = [pool.submit(reject) for _ in range(30)]
            self.assertIsNotNone(self.adapter.tasks.complete(task, protocol.STATE_COMPLETED, "ACK"))
        self.assertEqual(set(f.result() for f in pending), {protocol.ERR_TASK_NOT_CANCELABLE})
        self.assertEqual(self.adapter.tasks.get(task)["reply"], "ACK")


class IsolatedHttpCancelSafety(unittest.TestCase):
    def test_forwarded_unknown_receipt_keeps_original_input_id_and_never_retries(self):
        h = http_shim.NativeHttpAcceptance("test_timeout_retry_and_late_terminal_receipt")
        h.setUp()
        self.addCleanup(h.doCleanups)
        h.adapter._agents["dev"] = {"slug": "dev", "tenant": "dev", "path": "/dev",
                                    "local": False, "profile": "dev", "name": "dev",
                                    "allowed_peers": {"alice"}}
        calls = []
        finished = threading.Event()

        def fake_unknown(_agent, _peer, _context, _framed, *, input_id):
            calls.append(input_id)
            finished.set()
            return "[owner execution unverified]", protocol.STATE_WORKING

        h.adapter._forward_to_profile = fake_unknown
        params = h.message("forwarded-timeout-unknown")
        params["tenant"] = "dev"
        params["configuration"] = {"returnImmediately": True}
        first = h.post("SendMessage", params)
        task_id = first["result"]["task"]["id"]
        self.assertTrue(finished.wait(3), "isolated forwarded owner was never reached")
        self.assertEqual(h.post("SendMessage", params)["result"]["task"]["id"], task_id)
        self.assertEqual(len(calls), 1, "unknown receipt must never cause second dispatch")
        self.assertTrue(calls[0].startswith("a2a-msg:"))
        rec = h.adapter.tasks.get(task_id, "dev", "dev", "alice")
        self.assertEqual(rec["input_id"], calls[0])
        self.assertEqual(rec["state"], protocol.STATE_WORKING)
        self.assertEqual(h.post("CancelTask", {"id": task_id, "tenant": "dev"})
                         ["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        with self.assertRaises(urllib.error.HTTPError) as denied:
            h.post("GetTask", {"id": task_id, "tenant": "dev"}, peer="bob")
        self.assertEqual(denied.exception.code, 403)
        self.assertEqual(h.adapter.tasks.get(task_id)["state"], protocol.STATE_WORKING)

    def test_loopback_jsonrpc_rejects_cancel_until_fake_backend_completion(self):
        # Exercise real native HTTP/JSON-RPC routing with a deterministic fake
        # gateway inference handler; no external service or live task involved.
        h = http_shim.NativeHttpAcceptance("test_timeout_retry_and_late_terminal_receipt")
        h.setUp()
        self.addCleanup(h.doCleanups)
        old_timeout = adapter._reply_timeout
        adapter._reply_timeout = lambda: 0.025
        self.addCleanup(setattr, adapter, "_reply_timeout", old_timeout)
        h.delay = 0.3
        payload = h.message("cancel-http-unique")
        first = h.post("SendMessage", payload)
        task_id = first["result"]["task"]["id"]
        self.assertEqual(first["result"]["task"]["status"]["state"], protocol.STATE_WORKING)
        self.assertEqual(h.post("CancelTask", {"id": task_id}, peer="bob")
                         ["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        for _ in range(6):
            self.assertEqual(h.post("CancelTask", {"id": task_id})
                             ["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            result = h.post("GetTask", {"id": task_id})["result"]
            if result["status"]["state"] == protocol.STATE_COMPLETED:
                break
            time.sleep(0.025)
        self.assertEqual(result["status"]["state"], protocol.STATE_COMPLETED, h.errors)
        self.assertEqual(protocol.extract_text(result["artifacts"][0]), "ACK")
        self.assertEqual(h.post("CancelTask", {"id": task_id})
                         ["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        self.assertEqual(h.post("SendMessage", payload)["result"]["task"]["id"], task_id)
        self.assertEqual(h.count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
