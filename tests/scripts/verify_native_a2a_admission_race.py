"""Native A2A admission-rejection vs concurrent replay dispatch race.

    python tests/scripts/verify_native_a2a_admission_race.py

Uses real adapter._end_task, TaskStore and SQLite ledger. No inference or live traffic.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest

import verify_native_a2a_peer_scope_stdlib as shim
from plugins.platforms.a2a import adapter, protocol
from plugins.platforms.a2a.ledger import DurableTaskLedger

class AdmissionRejectionRace(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db=Path(self.tmp.name)/"tasks.sqlite3"
        self.tasks=protocol.TaskStore(DurableTaskLedger(self.db))
        self.owner=SimpleNamespace(tasks=self.tasks)
        self.payload={"message":{"messageId":"race-1","role":"ROLE_USER","parts":[{"text":"hello"}]}}
        self.rec,first=self.tasks.admit("alice","race-1",self.payload,"shared-context")
        self.assertFalse(first)

    def reject(self,reason="too many turns"):
        return adapter.A2AAdapter._end_task(
            self.owner,self.rec,protocol.STATE_REJECTED,reason)[0]

    def test_reserved_rejection_settles_before_dispatch(self):
        receipt=self.reject()
        self.assertEqual(receipt["status"]["state"],protocol.STATE_REJECTED)
        self.assertEqual(self.tasks.get(self.rec["task_id"])["effect_state"],"settled")
        self.assertFalse(self.tasks.claim_dispatch(self.rec["task_id"]))
        self.assertEqual(self.tasks.get(self.rec["task_id"])["reply"],"too many turns")

    def test_concurrent_duplicate_claim_precedes_original_validation_rejection(self):
        duplicate,replayed=self.tasks.admit("alice","race-1",self.payload,"shared-context")
        self.assertTrue(replayed)
        self.assertEqual(duplicate["task_id"],self.rec["task_id"])
        self.assertTrue(self.tasks.claim_dispatch(self.rec["task_id"]))
        receipt=self.reject()
        self.assertEqual(receipt["status"]["state"],protocol.STATE_WORKING)
        self.assertEqual(self.tasks.get(self.rec["task_id"])["effect_state"],"uncertain")
        self.assertIsNotNone(self.tasks.complete(self.rec["task_id"],protocol.STATE_COMPLETED,"actual backend result"))
        self.assertEqual(self.tasks.get(self.rec["task_id"])["reply"],"actual backend result")
        self.assertIsNone(self.tasks.complete(self.rec["task_id"],protocol.STATE_REJECTED,"late rejection"))

    def test_separate_process_claim_precedes_local_validation_rejection(self):
        code=(
            "from plugins.platforms.a2a.ledger import DurableTaskLedger; import sys; "
            "r=DurableTaskLedger(sys.argv[1]).update(sys.argv[2],claim=True); "
            "assert r and r['effect_state']=='uncertain',r"
        )
        proc=subprocess.run(
            ["/usr/bin/python3","-c",code,str(self.db),self.rec["task_id"]],
            capture_output=True,text=True,timeout=15)
        self.assertEqual(proc.returncode,0,proc.stderr)
        receipt=self.reject()
        self.assertEqual(receipt["status"]["state"],protocol.STATE_WORKING)
        self.assertEqual(self.tasks.get(self.rec["task_id"])["effect_state"],"uncertain")
        recovered=protocol.TaskStore(DurableTaskLedger(self.db))
        self.assertIsNotNone(recovered.complete(self.rec["task_id"],protocol.STATE_COMPLETED,"late ACK"))
        self.assertEqual(self.tasks.get(self.rec["task_id"])["reply"],"late ACK")

if __name__=="__main__":
    unittest.main(verbosity=2)

