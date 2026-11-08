"""Adversarial correlation of local A2A final replies and cancellation.

No live backend or production ports. Uses the exact native adapter and a
temporary SQLite WAL store; `reply_to` is the gateway event's task id.
Run: python tests/scripts/verify_native_a2a_late_reply_binding.py
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import verify_native_a2a_peer_scope_stdlib  # noqa: F401 - gateway import shims
from plugins.platforms.a2a import adapter, protocol
from plugins.platforms.a2a.ledger import DurableTaskLedger


class LateReplyBinding(unittest.TestCase):
    def setUp(self):
        # The peer-scope import shim uses a non-constructible SendResult stub;
        # replace only this test's return-value type, not the adapter logic.
        result_patch = patch.object(adapter, "SendResult", lambda **kw: SimpleNamespace(**kw))
        result_patch.start()
        self.addCleanup(result_patch.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = protocol.TaskStore(DurableTaskLedger(Path(tmp.name) / "tasks.sqlite3"))
        self.adapter = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
        self.adapter.tasks = self.store
        self.adapter._pending_lock = threading.Lock()
        self.adapter._pending = {}
        self.adapter._pending_order = {}
        self.adapter._active_tasks = set()
        self.recorded = []

        def record(task_id, _context, _peer, state, reply):
            self.recorded.append((task_id, state, reply))
            return self.store.complete(task_id, state, reply) is not None

        self.adapter._record_outcome = record

    def admit(self, message_id, peer="alice", tenant="tenant-a", context="same"):
        agent = {"slug": "private", "tenant": tenant}
        execution_context = adapter.A2AAdapter._execution_context_id(agent, peer, context)
        payload = {"message": {"messageId": message_id, "contextId": context,
                               "role": "ROLE_USER", "parts": [{"text": message_id}]}}
        rec, replay = self.store.admit(peer, message_id, payload, context,
                                       "private", tenant, input_id="test:" + message_id)
        self.assertFalse(replay)
        self.assertTrue(self.store.claim_dispatch(rec["task_id"]))
        self.adapter._add_pending(rec["task_id"], execution_context)
        return rec["task_id"], execution_context

    def finish(self, context, text, reply_to=None):
        return asyncio.run(self.adapter.send(context, text, reply_to=reply_to,
                                             metadata={"notify": True}))

    def test_second_task_finishing_first_does_not_settle_oldest(self):
        first, context = self.admit("first")
        second, _ = self.admit("second")
        self.finish(context, "second result", reply_to=second)
        self.assertEqual(self.store.get(second)["reply"], "second result")
        self.assertEqual(self.store.get(first)["state"], protocol.STATE_WORKING)
        self.assertEqual([entry[0] for entry in self.recorded], [second])

    def test_late_reply_after_confirmed_cancel_cannot_complete_next_task(self):
        first, context = self.admit("cancel-me")
        second, _ = self.admit("keep-working")
        # Simulate the gateway's authoritative per-task completion callback;
        # our test import shim does not implement its ProcessingOutcome enum.
        self.assertTrue(self.adapter._resolve_task(first, protocol.STATE_CANCELED, ""))
        self.assertIsNotNone(self.store.complete(first, protocol.STATE_CANCELED, ""))
        self.adapter._pop_pending(first)
        self.assertEqual(self.store.get(first)["state"], protocol.STATE_CANCELED)
        self.finish(context, "stale result", reply_to=first)
        self.assertEqual(self.store.get(second)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.store.get(first)["state"], protocol.STATE_CANCELED)

    def test_task_id_from_different_tenant_context_cannot_settle_peer_task(self):
        first, context = self.admit("alice-work", peer="alice", tenant="tenant-a")
        other, other_context = self.admit("bob-work", peer="bob", tenant="tenant-b")
        self.assertNotEqual(context, other_context)
        self.finish(context, "bad cross-scope result", reply_to=other)
        self.assertEqual(self.store.get(first)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.store.get(other)["state"], protocol.STATE_WORKING)

    def test_uncorrelated_reply_with_multiple_live_tasks_fails_closed(self):
        first, context = self.admit("unkeyed-1")
        second, _ = self.admit("unkeyed-2")
        self.finish(context, "no task id")
        self.assertEqual(self.store.get(first)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.store.get(second)["state"], protocol.STATE_WORKING)

    def test_unkeyed_reply_with_one_durable_pending_fails_closed(self):
        only, context = self.admit("durable-unkeyed")
        self.finish(context, "unverifiable result")
        self.assertEqual(self.store.get(only)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.recorded, [])

    def test_late_unkeyed_reply_after_worker_restart_cannot_settle_new_task(self):
        old, context = self.admit("pre-crash-uncertain")
        self.adapter._pop_pending(old)  # emulate lost local Future, not lost ledger
        newer, _ = self.admit("post-restart-working")
        self.finish(context, "stale old reply without task identity")
        self.assertEqual(self.store.get(old)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.store.get(newer)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.recorded, [])

    def test_legacy_unkeyed_reply_with_one_in_memory_pending_still_works(self):
        self.adapter.tasks = protocol.TaskStore()
        self.adapter._add_pending("legacy-task", "legacy-session")
        result = self.adapter._resolve_oldest_for_context(
            "legacy-session", protocol.STATE_COMPLETED, "legacy reply")
        self.assertEqual(result, "legacy-task")


if __name__ == "__main__":
    unittest.main(verbosity=2)
