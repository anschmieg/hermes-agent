"""Fail closed on nonterminal or ambiguous forwarded owner receipts (FAKE owner).

Each case executes the production A2A adapter and SQLite ledger in a child
process, writes any terminal outcome, then SIGKILLs itself. No Chief, external
gateway, live backend, or remote STOP; never evidence of Chief authority.
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

import verify_native_a2a_peer_scope_stdlib  # noqa: F401 - dependency shims only
from plugins.platforms.a2a import adapter, protocol
from plugins.platforms.a2a.ledger import DurableTaskLedger

SCOPE = dict(peer="alice", agent_slug="private", tenant="tenant-a",
             input_id="a2a-msg:" + "e" * 64)
CONTEXT = "a2a-session-" + "c" * 64
AGENT = dict(slug="private", tenant="tenant-a", profile="fake-profile")


def worker(db: str, task_id: str, variant: str) -> None:
    async def fake_owner(_home, *, agent, tenant, peer, context_id, input_id, text, timeout):
        assert text == "FAKE only"
        status, outcome = {
            "queued-completed": ("queued", "completed"),
            "started-completed": ("started", "completed"),
            "unknown-completed": ("unknown", "completed"),
            "missing-status": (None, "completed"),
            "terminal-missing-outcome": ("terminal", None),
            "terminal-invalid-outcome": ("terminal", "foreign"),
            "terminal-completed": ("terminal", "completed"),
            "terminal-failed": ("terminal", "failed"),
            "terminal-interrupted": ("terminal", "interrupted"),
        }[variant]
        receipt = dict(
            outcome=outcome, result={"final_response": "FAKE OWNER RESULT"},
            agent=agent, tenant=tenant, peer=peer,
            context_id=context_id, input_id=input_id)
        if status is not None:
            receipt["status"] = status
        return receipt

    gateway = types.ModuleType("gateway.session_a2a")
    gateway.forward_to_owner = fake_owner
    client = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
    with patch.dict(sys.modules, {"gateway.session_a2a": gateway}), \
            patch.object(adapter, "_profile_home", return_value=Path(db).parent):
        reply, state = client._forward_to_profile(
            AGENT, SCOPE["peer"], CONTEXT, "FAKE only", input_id=SCOPE["input_id"])
    if state != protocol.STATE_WORKING:
        protocol.TaskStore(DurableTaskLedger(Path(db))).complete(task_id, state, reply)
    os.kill(os.getpid(), signal.SIGKILL)


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires SIGKILL")
class ForwardTerminalReceiptGate(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="a2a-forward-terminal-")
        self.addCleanup(temp.cleanup)
        self.db = str(Path(temp.name) / "tasks.sqlite3")
        ledger = DurableTaskLedger(Path(self.db))
        rec, replay = ledger.reserve(
            message_id="owner-status", payload={"text": "once"},
            context_id="external-context", **SCOPE)
        self.assertFalse(replay)
        self.task_id = rec["task_id"]
        self.assertIsNotNone(ledger.update(self.task_id, claim=True))

    def killed(self, variant):
        p = subprocess.run([sys.executable, "-u", __file__, "--worker",
                            self.db, self.task_id, variant],
                           cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, -signal.SIGKILL, p.stderr)
        return DurableTaskLedger(Path(self.db)).get(self.task_id)

    def test_nonterminal_owner_receipt_cannot_forge_completed(self):
        for variant in ("queued-completed", "started-completed",
                        "unknown-completed", "missing-status"):
            with self.subTest(variant=variant):
                rec = self.killed(variant)
                self.assertEqual((rec["state"], rec["effect_state"]),
                                 (protocol.STATE_WORKING, "uncertain"))

    def test_ambiguous_terminal_outcome_cannot_forge_failed(self):
        for variant in ("terminal-missing-outcome", "terminal-invalid-outcome"):
            with self.subTest(variant=variant):
                rec = self.killed(variant)
                self.assertEqual((rec["state"], rec["effect_state"]),
                                 (protocol.STATE_WORKING, "uncertain"))

    def test_verified_terminal_outcomes_remain_durable(self):
        expectations = {
            "terminal-completed": protocol.STATE_COMPLETED,
            "terminal-failed": protocol.STATE_FAILED,
            "terminal-interrupted": protocol.STATE_FAILED,
        }
        for variant, state in expectations.items():
            with self.subTest(variant=variant):
                # Independent task per positive outcome to avoid mutating a terminal row.
                db = Path(self.db).with_name(variant + ".sqlite3")
                ledger = DurableTaskLedger(db)
                rec, _ = ledger.reserve(message_id=variant, payload={"text": variant},
                                        context_id="external-context", **SCOPE)
                self.assertIsNotNone(ledger.update(rec["task_id"], claim=True))
                p = subprocess.run([sys.executable, "-u", __file__, "--worker",
                                    str(db), rec["task_id"], variant],
                                   cwd=Path(__file__).resolve().parents[2],
                                   capture_output=True, text=True, timeout=30)
                self.assertEqual(p.returncode, -signal.SIGKILL, p.stderr)
                actual = DurableTaskLedger(db).get(rec["task_id"])
                self.assertEqual((actual["state"], actual["effect_state"], actual["reply"]),
                                 (state, "settled", "FAKE OWNER RESULT"))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
