"""Real SIGKILL + synthetic Chief read-only receipt scope contract.

Only local SQLite, exact native ledger and a FAKE owner lookup. No live Chief
read, SendMessage, GetTask, CancelTask, backend inference or external traffic.
Run: python tests/scripts/verify_native_a2a_stop_receipt_scope.py
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import verify_native_a2a_peer_scope_stdlib as shim  # noqa: F401 - native import shims
from plugins.platforms.a2a.chief_receipt import lookup_stop_receipt
from plugins.platforms.a2a.ledger import DurableTaskLedger

SCOPE = dict(peer="alice", agent_slug="private", tenant="tenant-a",
             input_id="a2a-msg:" + "a" * 64)
PAYLOAD = {"message": {"messageId": "receipt-gap", "role": "ROLE_USER",
                       "parts": [{"text": "side effect once"}]}}


class FakeChiefReadOnly:
    """Synthetic interface, NOT an authenticated authoritative receipt."""
    def __init__(self, receipt=None):
        self.receipt = receipt
        self.calls = []

    def lookup_input(self, **scope):
        self.calls.append(scope)
        return self.receipt

    def CancelTask(self, *_a, **_kw):
        raise AssertionError("read-only path called CancelTask")

    def SendMessage(self, *_a, **_kw):
        raise AssertionError("read-only path called SendMessage")


def worker(db, task_id, trace):
    ledger = DurableTaskLedger(db)
    assert ledger.claim_stop_attempt(task_id, **SCOPE)["stop_attempt_state"] == "uncertain"
    with open(trace, "a", encoding="utf-8") as output:
        output.write("FAKE_STOP_ACK\n")
        output.flush()
        os.fsync(output.fileno())
    os.kill(os.getpid(), signal.SIGKILL)


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "POSIX SIGKILL required")
class ReceiptAfterAckSigkill(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="native-receipt-gap-")
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "state.db"
        self.trace = Path(temp.name) / "effects.log"
        ledger = DurableTaskLedger(self.db)
        rec, replay = ledger.reserve(peer=SCOPE["peer"], message_id="receipt-gap",
                                     payload=PAYLOAD, context_id="private-context",
                                     agent_slug=SCOPE["agent_slug"], tenant=SCOPE["tenant"],
                                     input_id=SCOPE["input_id"])
        self.assertFalse(replay)
        self.task_id = rec["task_id"]
        self.assertIsNotNone(ledger.update(self.task_id, claim=True))
        child = subprocess.run([sys.executable, "-u", __file__, "--worker", str(self.db),
                                self.task_id, str(self.trace)],
                               cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, -signal.SIGKILL, child.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])
        self.ledger = DurableTaskLedger(self.db)
        self.attempted = self.ledger.get(self.task_id)["stop_attempted_at"]

    def receipt(self, **changes):
        value = dict(inputId=SCOPE["input_id"], peer=SCOPE["peer"],
                     agent=SCOPE["agent_slug"], tenant=SCOPE["tenant"],
                     contextId="private-context", status="terminal",
                     outcome="canceled", updatedAt=time.time())
        value.update(changes)
        return value

    def lookup(self, reader, **changes):
        return lookup_stop_receipt(self.ledger, self.task_id,
                                   **{**SCOPE, **changes}, reader=reader)

    def test_post_crash_observation_is_never_backend_stop_confirmation(self):
        fake = FakeChiefReadOnly(self.receipt(stopConfirmed=True, authority="chief"))
        result = self.lookup(fake)
        self.assertEqual(result, dict(task_id=self.task_id, input_id=SCOPE["input_id"],
                                      owner_status="terminal", task_state="TASK_STATE_WORKING",
                                      stop_confirmed=False))
        self.assertEqual(fake.calls, [dict(input_id=SCOPE["input_id"], peer="alice",
                                           agent_slug="private", tenant="tenant-a",
                                           context_id="private-context")])
        self.assertEqual(self.ledger.get(self.task_id)["state"], "TASK_STATE_WORKING")
        self.assertIsNone(self.ledger.claim_stop_attempt(self.task_id, **SCOPE))
        self.assertIsNone(self.ledger.update(self.task_id, claim=True))
        again, replay = self.ledger.reserve(peer="alice", message_id="receipt-gap",
                                            payload=PAYLOAD, context_id="private-context",
                                            agent_slug="private", tenant="tenant-a",
                                            input_id=SCOPE["input_id"])
        self.assertTrue(replay)
        self.assertEqual(again["task_id"], self.task_id)
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])

    def test_wrong_local_scope_never_queries_owner(self):
        fake = FakeChiefReadOnly(self.receipt())
        for key, value in (("peer", "bob"), ("tenant", "tenant-b"),
                           ("agent_slug", "other"), ("agent_slug", ""),
                           ("input_id", "a2a-msg:" + "b" * 64), ("input_id", "")):
            with self.subTest(key=key, value=value):
                self.assertIsNone(self.lookup(fake, **{key: value}))
        self.assertEqual(fake.calls, [])

    def test_missing_stale_and_wrong_receipt_fields_fail_closed(self):
        bad = [None, {}, self.receipt(updatedAt=self.attempted - 10),
               self.receipt(peer="bob"), self.receipt(tenant="tenant-b"),
               self.receipt(agent="other"), self.receipt(inputId="a2a-msg:" + "b" * 64),
               self.receipt(contextId="other"), self.receipt(status="unknown"),
               self.receipt(updatedAt=time.time() + 3600),
               self.receipt(updatedAt="2026-11-08"), self.receipt(updatedAt=float("nan"))]
        for receipt in bad:
            with self.subTest(receipt=receipt):
                self.assertIsNone(self.lookup(FakeChiefReadOnly(receipt)))
        self.assertEqual(self.ledger.get(self.task_id)["state"], "TASK_STATE_WORKING")

    def test_reader_missing_or_transport_failure_does_not_settle(self):
        self.assertIsNone(self.lookup(None))
        class Broken:
            def lookup_input(self, **_kwargs):
                raise ConnectionError("Chief unavailable")
        self.assertIsNone(self.lookup(Broken()))
        self.assertEqual(self.ledger.get(self.task_id)["stop_attempt_state"], "uncertain")

    def test_read_only_retries_cannot_resend_cancel_or_settle(self):
        fake = FakeChiefReadOnly(self.receipt())
        for _ in range(3):
            self.assertIsNotNone(self.lookup(fake))
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])
        rec = self.ledger.get(self.task_id)
        self.assertEqual((rec["state"], rec["effect_state"], rec["stop_attempt_state"]),
                         ("TASK_STATE_WORKING", "uncertain", "uncertain"))



    def test_ambiguous_and_forged_response_metadata_is_not_observation(self):
        """A single matching top-level identity cannot bless conflicting owner rows."""
        competing = self.receipt(peer="mallory", status="running", outcome=None)
        forged = [
            self.receipt(matches=[self.receipt(), competing]),
            self.receipt(receipts=[self.receipt(), competing]),
            self.receipt(query={"peer": "mallory"}),
            self.receipt(status="running", outcome="canceled"),
            self.receipt(status="queued", outcome="canceled"),
            self.receipt(status="queued", outcome=None, stopConfirmed=True),
        ]
        for value in forged:
            with self.subTest(value=value):
                reader = FakeChiefReadOnly(value)
                self.assertIsNone(self.lookup(reader))
                self.assertEqual(len(reader.calls), 1)
        rec = self.ledger.get(self.task_id)
        self.assertEqual((rec["state"], rec["effect_state"], rec["stop_attempt_state"]),
                         ("TASK_STATE_WORKING", "uncertain", "uncertain"))
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])

    def test_forged_chief_authority_endpoint_and_stop_provenance_fail_closed(self):
        """A matching top-level inputId must not bless forged or contradictory claims."""
        original = self.ledger.get(self.task_id)
        forged = [
            self.receipt(authority={
                "name": "chief", "inputId": SCOPE["input_id"],
                "tenant": "tenant-b", "endpoint": "https://untrusted.invalid",
                "backendStopConfirmed": True}),
            self.receipt(authority=["chief", "backend-stop"]),
            self.receipt(authority=True),
            self.receipt(authority="untrusted-endpoint"),
            self.receipt(outcome="completed", stopConfirmed=True),
            self.receipt(outcome="failed", stopConfirmed=True),
            self.receipt(outcome=None, stopConfirmed=True),
            self.receipt(inputId="a2a-msg:" + "b" * 64),
            self.receipt(tenant="tenant-b"),
            self.receipt(endpoint="https://untrusted.invalid"),
        ]
        for candidate in forged:
            with self.subTest(candidate=candidate):
                fake = FakeChiefReadOnly(candidate)
                self.assertIsNone(self.lookup(fake))
                self.assertEqual(len(fake.calls), 1)
        self.assertEqual(self.ledger.get(self.task_id), original)
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])

    def test_once_fresh_owner_receipt_becomes_old_it_fails_closed(self):
        """A later replay must not turn a stale owner snapshot into fresh evidence."""
        from unittest.mock import patch
        candidate = self.receipt(updatedAt=self.attempted + 1)
        with patch("plugins.platforms.a2a.chief_receipt.time.time",
                   return_value=self.attempted + 601):
            self.assertIsNone(self.lookup(FakeChiefReadOnly(candidate)))
        self.assertIsNone(self.ledger.claim_stop_attempt(self.task_id, **SCOPE))
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])

if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
