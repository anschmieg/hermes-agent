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


def _callback_cancel_worker(db: str, task_id: str, outcome: str, stale: str) -> None:
    """Exact original adapter + SQLite callback, then actual SIGKILL; no external calls."""
    import asyncio
    import os
    from enum import Enum
    from types import SimpleNamespace
    from unittest.mock import patch

    # The stdlib-only fixture stubs gateway.platforms.event to avoid importing
    # the full gateway. These values mirror the real core ProcessingOutcome enum,
    # whose CANCELLED is an expected asyncio.CancelledError, NOT a STOP receipt.
    class CoreOutcome(Enum):
        SUCCESS = "success"
        FAILURE = "failure"
        CANCELLED = "cancelled"

    client = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
    client.tasks = protocol.TaskStore(DurableTaskLedger(Path(db)))
    client._pending_lock = threading.RLock()
    client._pending = {}
    client._pending_order = {}
    client._active_tasks = set()
    rec = client.tasks.get(task_id)
    assert rec is not None and rec["state"] == protocol.STATE_WORKING
    future = client._add_pending(task_id, rec["context_id"])
    context = client._execution_context_id(
        {"slug": rec["agent_slug"], "tenant": rec["tenant"]}, rec["peer"], rec["context_id"])
    forged = stale == "stale"
    event = SimpleNamespace(
        message_id=task_id,
        source=SimpleNamespace(chat_id="foreign-session" if forged else context,
                               user_id="mallory" if forged else rec["peer"]),
        metadata={"tenant": "tenant-b" if forged else rec["tenant"],
                  "peer": "mallory" if forged else rec["peer"],
                  "contextId": "foreign-context" if forged else rec["context_id"],
                  "taskId": "foreign-task" if forged else task_id,
                  "inputId": "foreign-input" if forged else rec["input_id"],
                  "sessionId": "foreign-session" if forged else context,
                  "generation": "stale" if forged else "current"},
        _streamed_final_response="normal streamed response",
    )
    client._send_push_notification = lambda *_args: None
    with patch.object(adapter, "ProcessingOutcome", CoreOutcome), \
         patch.object(protocol, "persist_message", lambda *_args, **_kwargs: None), \
         patch.object(adapter.security, "audit", lambda *_args, **_kwargs: None):
        asyncio.run(client.on_processing_complete(event, CoreOutcome[outcome]))
    assert future.done(), "completion hook must release waiting HTTP future"
    os.kill(os.getpid(), signal.SIGKILL)


@unittest.skipUnless(hasattr(signal, "SIGKILL"), "requires actual SIGKILL")
class UnverifiedCallbackCancelSafety(unittest.TestCase):
    """A local core CANCELLED event is not independent authenticated backend STOP."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="a2a-callback-cancel-")
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "tasks.sqlite3"
        self.store = protocol.TaskStore(DurableTaskLedger(self.db))
        self.scope = {"peer": "alice", "agent_slug": "private", "tenant": "tenant-a",
                      "input_id": "a2a-msg:" + "c" * 64}

    def new_task(self, message_id: str):
        rec, replay = self.store.admit(
            "alice", message_id,
            {"message": {"messageId": message_id, "role": "ROLE_USER",
                         "parts": [{"text": "local fake only"}]}},
            "ctx-" + message_id, "private", "tenant-a",
            input_id=self.scope["input_id"])
        self.assertFalse(replay)
        self.assertTrue(self.store.claim_dispatch(rec["task_id"]))
        return rec["task_id"]

    def killed_callback(self, task_id: str, outcome: str, stale: str):
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--callback-worker",
             str(self.db), task_id, outcome, stale],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stderr)
        # Force real process restart / WAL reopen, not just an in-memory snapshot.
        self.store = protocol.TaskStore(DurableTaskLedger(self.db))
        return self.store.get(task_id, "private", "tenant-a", "alice")

    def test_local_cancelled_without_stop_authority_survives_sigkill_and_late_callback(self):
        task_id = self.new_task("unverified-cancel")
        for variant in ("legitimate-core-cancel", "stale-late-cancel"):
            rec = self.killed_callback(task_id, "CANCELLED",
                                       "stale" if variant == "stale-late-cancel" else "current")
            self.assertEqual((rec["state"], rec["effect_state"], rec["input_id"]),
                             (protocol.STATE_WORKING, "uncertain", self.scope["input_id"]),
                             variant)
            self.assertEqual(self.store.get_by_message(
                "alice", "unverified-cancel", "private", "tenant-a")["state"],
                protocol.STATE_WORKING)
            self.assertIsNone(self.store.get(task_id, "private", "tenant-b", "alice"))
            self.assertIsNone(self.store.get(task_id, "private", "tenant-a", "mallory"))
            self.assertFalse(self.store.claim_dispatch(task_id),
                             "stale local cancel must never permit re-dispatch")

    def test_direct_outcome_sink_rejects_unverified_canceled_without_side_effects(self):
        from unittest.mock import patch
        task_id = self.new_task("forged-direct-cancel")
        client = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
        client.tasks = self.store
        client._send_push_notification = lambda *_args: None
        with patch.object(protocol, "persist_message", lambda *_args, **_kwargs: None), \
             patch.object(adapter.security, "audit", lambda *_args, **_kwargs: None):
            self.assertFalse(client._record_outcome(
                task_id, "ctx-forged-direct-cancel", "alice",
                protocol.STATE_CANCELED, "no independently confirmed backend STOP"))
        rec = protocol.TaskStore(DurableTaskLedger(self.db)).get(
            task_id, "private", "tenant-a", "alice")
        self.assertEqual((rec["state"], rec["effect_state"], rec["reply"]),
                         (protocol.STATE_WORKING, "uncertain", ""))
        self.assertFalse(self.store.claim_dispatch(task_id))

    def test_normal_success_and_failure_outcomes_remain_terminal(self):
        for name, outcome, expected in (
            ("ok", "SUCCESS", protocol.STATE_COMPLETED),
            ("error", "FAILURE", protocol.STATE_FAILED),
        ):
            with self.subTest(outcome=outcome):
                task_id = self.new_task(name)
                rec = self.killed_callback(task_id, outcome, "current")
                self.assertEqual((rec["state"], rec["effect_state"]), (expected, "settled"))
                if outcome == "SUCCESS":
                    self.assertEqual(rec["reply"], "normal streamed response")

if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--callback-worker":
        _callback_cancel_worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
