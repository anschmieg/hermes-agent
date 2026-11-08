"""Real SIGKILL after a fake remote stop ACK, before durable terminal commit.

Only exact Hermes modules, isolated SQLite, loopback-free fake backend.
Neither the ACK nor this test proves authoritative Chief stop confirmation.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

import verify_native_a2a_peer_scope_stdlib  # noqa: F401 - gateway shims only
from plugins.platforms.a2a import adapter, protocol
from plugins.platforms.a2a.ledger import DurableTaskLedger

SCOPE = {"peer": "alice", "agent_slug": "private", "tenant": "tenant-a",
         "input_id": "a2a-msg:" + "a" * 64}
PAYLOAD = {"message": {"messageId": "stop-gap", "role": "ROLE_USER",
                       "parts": [{"text": "original once"}]}}

def worker(db, trace, task_id):
    ledger = DurableTaskLedger(db)
    claim = ledger.claim_stop_attempt(task_id, **SCOPE)
    assert claim and claim["stop_attempt_state"] == "uncertain"
    with open(trace, "a", encoding="utf-8") as f:
        f.write("FAKE_STOP_ACK\n")
        f.flush()
        os.fsync(f.fileno())
    os.kill(os.getpid(), signal.SIGKILL)

@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires SIGKILL")
class RemoteStopAckCrashGap(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="native-stop-ack-")
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "ledger.db"
        self.trace = Path(tmp.name) / "trace.log"
        self.ledger = DurableTaskLedger(self.db)
        rec, replay = self.ledger.reserve(
            peer="alice", message_id="stop-gap", payload=PAYLOAD,
            context_id="ctx", agent_slug="private", tenant="tenant-a",
            input_id=SCOPE["input_id"])
        self.assertFalse(replay)
        self.task_id = rec["task_id"]
        self.assertIsNotNone(self.ledger.update(self.task_id, claim=True))

    def test_ack_then_sigkill_before_terminal_commit_is_uncertain_and_never_replayed(self):
        proc = subprocess.run(
            [sys.executable or "/usr/bin/python3", "-u", __file__, "--worker",
             str(self.db), str(self.trace), self.task_id],
            cwd=Path(__file__).resolve().parents[2], capture_output=True,
            text=True, timeout=20)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])
        restarted = DurableTaskLedger(self.db)
        rec = restarted.get(self.task_id)
        self.assertEqual((rec["state"], rec["effect_state"],
                          rec["stop_attempt_state"], rec["input_id"]),
                         (protocol.STATE_WORKING, "uncertain", "uncertain", SCOPE["input_id"]))
        self.assertIsNone(restarted.claim_stop_attempt(self.task_id, **SCOPE))
        self.assertIsNone(restarted.update(self.task_id, claim=True))
        replayed, replay = restarted.reserve(
            peer="alice", message_id="stop-gap", payload=PAYLOAD, context_id="ctx",
            agent_slug="private", tenant="tenant-a", input_id=SCOPE["input_id"])
        self.assertTrue(replay)
        self.assertEqual(replayed["task_id"], self.task_id)
        a = adapter.A2AAdapter.__new__(adapter.A2AAdapter)
        a.tasks = protocol.TaskStore(restarted)
        response = a._rpc_tasks_cancel(1, {"taskId": self.task_id},
                                       {"slug": "private", "tenant": "tenant-a"}, "alice")
        self.assertEqual(response["error"]["code"], protocol.ERR_TASK_NOT_CANCELABLE)
        self.assertEqual(a.tasks.get(self.task_id)["state"], protocol.STATE_WORKING)
        self.assertEqual(self.trace.read_text().splitlines(), ["FAKE_STOP_ACK"])

    def test_fences_peer_tenant_agent_and_authoritative_input_identity(self):
        for bad in ({**SCOPE, "peer": "bob"}, {**SCOPE, "tenant": "tenant-b"},
                    {**SCOPE, "agent_slug": ""}, {**SCOPE, "agent_slug": "other"},
                    {**SCOPE, "input_id": "a2a-msg:" + "b" * 64},
                    {**SCOPE, "input_id": ""}):
            self.assertIsNone(self.ledger.claim_stop_attempt(self.task_id, **bad))
        self.assertNotIn("stop_attempt_state", self.ledger.get(self.task_id))
        self.assertIsNotNone(self.ledger.claim_stop_attempt(self.task_id, **SCOPE))
        self.assertIsNone(self.ledger.claim_stop_attempt(self.task_id, **SCOPE))

    def test_multiple_independent_processes_get_only_one_stop_claim(self):
        code = (
            "from plugins.platforms.a2a.ledger import DurableTaskLedger; import sys; "
            "r=DurableTaskLedger(sys.argv[1]).claim_stop_attempt("
            "sys.argv[2],peer='alice',agent_slug='private',tenant='tenant-a',"
            "input_id='a2a-msg:'+'a'*64); print(1 if r else 0,flush=True)"
        )
        def race(_):
            proc = subprocess.run(
                [sys.executable or "/usr/bin/python3", "-c", code,
                 str(self.db), self.task_id],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return int(proc.stdout.strip())
        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(race, range(8)))
        self.assertEqual(sum(result), 1)
        self.assertEqual(self.ledger.get(self.task_id)["stop_attempt_state"], "uncertain")

    def test_unclaimed_dispatch_and_completed_tasks_cannot_start_stop(self):
        fresh, _ = self.ledger.reserve(
            peer="alice", message_id="another", payload=PAYLOAD, context_id="ctx",
            agent_slug="private", tenant="tenant-a", input_id="a2a-msg:" + "b"*64)
        self.assertIsNone(self.ledger.claim_stop_attempt(
            fresh["task_id"], **{**SCOPE, "input_id": "a2a-msg:" + "b"*64}))
        self.ledger.update(self.task_id, state=protocol.STATE_COMPLETED,
                           reply="owner completion", effect_state="settled")
        self.assertIsNone(self.ledger.claim_stop_attempt(self.task_id, **SCOPE))
        self.assertEqual(self.ledger.get(self.task_id)["state"], protocol.STATE_COMPLETED)

if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
