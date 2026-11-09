"""SIGKILL / SQLite regression for foreign forwarded-owner completion receipts.

Uses the real Hermes A2A adapter and ledger, with only an in-process FAKE
owner transport. No Chief, network, backend STOP or external profiles.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import verify_native_a2a_peer_scope_stdlib  # noqa: F401 - gateway import shims only
from plugins.platforms.a2a import adapter, protocol
from plugins.platforms.a2a.ledger import DurableTaskLedger

SCOPE = dict(peer="alice", agent_slug="private", tenant="tenant-a",
             input_id="a2a-msg:" + "a" * 64)
CONTEXT = "a2a-session-" + "c" * 64
AGENT = dict(slug="private", tenant="tenant-a", profile="fake-profile")


def worker(db: str, task_id: str, variant: str) -> None:
    async def fake_owner(_home, *, agent, tenant, peer, context_id, input_id, text, timeout):
        assert text == "FAKE only"
        receipt = dict(status="terminal", outcome="completed",
                       result={"final_response": "FOREIGN OWNER RESULT"},
                       agent=agent, tenant=tenant, peer=peer,
                       context_id=context_id, input_id=input_id)
        if variant == "missing-input":
            del receipt["input_id"]
        elif variant != "valid":
            field = {"wrong-input": "input_id", "wrong-tenant": "tenant",
                     "wrong-peer": "peer", "wrong-agent": "agent",
                     "wrong-execution": "context_id"}[variant]
            receipt[field] = "foreign-owner-binding"
        return receipt

    fake_gateway = types.ModuleType("gateway.session_a2a")
    fake_gateway.forward_to_owner = fake_owner
    forwarded = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
    with patch.dict(sys.modules, {"gateway.session_a2a": fake_gateway}), \
            patch.object(adapter, "_profile_home", return_value=Path(db).parent):
        reply, state = forwarded._forward_to_profile(
            AGENT, SCOPE["peer"], CONTEXT, "FAKE only", input_id=SCOPE["input_id"])
    if state != protocol.STATE_WORKING:
        protocol.TaskStore(DurableTaskLedger(Path(db))).complete(task_id, state, reply)
    os.kill(os.getpid(), signal.SIGKILL)


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires SIGKILL")
class ForwardOwnerBindingCrash(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="a2a-forward-binding-")
        self.addCleanup(tmp.cleanup)
        self.db = str(Path(tmp.name) / "tasks.sqlite3")
        self.ledger = DurableTaskLedger(Path(self.db))
        task, replay = self.ledger.reserve(
            message_id="owner-binding", payload={"message": "one"},
            context_id="external-context", **SCOPE)
        self.assertFalse(replay)
        self.task_id = task["task_id"]
        self.assertIsNotNone(self.ledger.update(self.task_id, claim=True))

    def invoke_killed_worker(self, variant):
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--worker", self.db, self.task_id, variant],
            cwd=Path(__file__).resolve().parents[2], capture_output=True,
            text=True, timeout=30)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stderr)
        return DurableTaskLedger(Path(self.db)).get(self.task_id)

    def test_forged_foreign_or_unbound_owner_receipt_never_finishes(self):
        for variant in ("wrong-input", "wrong-tenant", "wrong-peer",
                        "wrong-agent", "wrong-execution", "missing-input"):
            with self.subTest(variant=variant):
                rec = self.invoke_killed_worker(variant)
                self.assertEqual((rec["state"], rec["effect_state"]),
                                 (protocol.STATE_WORKING, "uncertain"))
                self.assertIsNone(self.ledger.update(self.task_id, claim=True))

    def test_matching_owner_receipt_still_completes_durably(self):
        rec = self.invoke_killed_worker("valid")
        self.assertEqual((rec["state"], rec["effect_state"], rec["reply"]),
                         (protocol.STATE_COMPLETED, "settled", "FOREIGN OWNER RESULT"))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
