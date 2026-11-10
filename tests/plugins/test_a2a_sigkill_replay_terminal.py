"""Real-process SQLite replay/terminal survival across two SIGKILLs.

Unlike the HTTP adapter crash test, this specifically exercises concurrent
independent recovery clients and a second crash immediately after committing
the terminal receipt. It uses the production durable ledger, no live backend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from plugins.platforms.a2a.ledger import DurableTaskLedger, IdempotencyConflict

PAYLOAD = {"message": {"messageId": "original", "role": "ROLE_USER", "parts": [{"text": "work"}]}}
SCOPE = {"peer": "alice", "message_id": "original", "payload": PAYLOAD,
         "context_id": "ctx", "agent_slug": "private", "tenant": "tenant-a",
         "input_id": "a2a-msg:original"}


def worker(mode: str, db: str, trace: str) -> None:
    ledger = DurableTaskLedger(db)
    if mode == "admit":
        rec, replay = ledger.reserve(**SCOPE)
        assert not replay
        claimed = ledger.update(rec["task_id"], claim=True)
        assert claimed is not None and claimed["effect_state"] == "uncertain"
        with open(trace, "a", encoding="utf-8") as log:
            log.write(rec["task_id"] + "\n")
            log.flush()
            os.fsync(log.fileno())
        print(json.dumps({"ack": rec["task_id"], "effect": claimed["effect_state"]}), flush=True)
        signal.pause()
    elif mode == "replay":
        rec, replay = ledger.reserve(**SCOPE)
        claimed = ledger.update(rec["task_id"], claim=True)
        print(json.dumps({"id": rec["task_id"], "replay": replay,
                          "claimed": claimed is not None, "effect": rec["effect_state"],
                          "state": rec["state"]}), flush=True)
    elif mode == "negative":
        try:
            ledger.reserve(**{**SCOPE, "tenant": "tenant-b"})
        except IdempotencyConflict:
            collision = True
        else:
            collision = False
        print(json.dumps({"cross_tenant_conflict": collision,
                          "other_peer_lookup": ledger.get_by_message("bob", "original") is None}), flush=True)
    elif mode == "settle":
        rec = ledger.get_by_message("alice", "original")
        assert rec is not None
        done = ledger.update(rec["task_id"], state="TASK_STATE_COMPLETED", reply="confirmed",
                             effect_state="settled")
        assert done and done["state"] == "TASK_STATE_COMPLETED"
        print(json.dumps({"id": done["task_id"], "state": done["state"], "reply": done["reply"]}), flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    elif mode == "terminal":
        rec, replay = ledger.reserve(**SCOPE)
        claimed = ledger.update(rec["task_id"], claim=True)
        overwritten = ledger.update(rec["task_id"], state="TASK_STATE_CANCELED", reply="FALSE CANCEL")
        print(json.dumps({"id": rec["task_id"], "replay": replay,
                          "claimed": claimed is not None, "overwrite": overwritten is not None,
                          "state": rec["state"], "reply": rec["reply"]}), flush=True)
    else:
        raise ValueError(mode)


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires POSIX SIGKILL")
class RecoveryAcrossTwoRealSigkills(unittest.TestCase):
    def test_parallel_replays_then_terminal_receipt_survive_second_sigkill(self):
        with tempfile.TemporaryDirectory(prefix="hermes-a2a-sigkill-") as tmp:
            db = str(Path(tmp) / "ledger.db")
            trace = str(Path(tmp) / "dispatch.log")
            root = Path(__file__).resolve().parents[2]

            def launch(mode):
                return subprocess.Popen([sys.executable, "-u", __file__, "--worker", mode, db, trace],
                                        cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        env={**os.environ, "PYTHONPATH": str(root)})

            first = launch("admit")
            try:
                line = first.stdout.readline()
                self.assertTrue(line, "worker failed to emit ACK")
                admitted = json.loads(line)
                self.assertEqual(admitted["effect"], "uncertain")
                self.assertEqual(Path(trace).read_text().splitlines(), [admitted["ack"]])
            finally:
                if first.poll() is None:
                    os.kill(first.pid, signal.SIGKILL)
            self.assertEqual(first.wait(timeout=8), -signal.SIGKILL)
            first.stdout.close()
            first.stderr.close()

            # Real independently started processes race to replay one identity.
            retries = [launch("replay") for _ in range(12)]
            observations = []
            for proc in retries:
                out, err = proc.communicate(timeout=15)
                self.assertEqual(proc.returncode, 0, err)
                observations.append(json.loads(out))
            self.assertEqual(len(observations), 12)
            for rec in observations:
                self.assertEqual(rec, {"id": admitted["ack"], "replay": True,
                                       "claimed": False, "effect": "uncertain",
                                       "state": "TASK_STATE_WORKING"})
            self.assertEqual(Path(trace).read_text().splitlines(), [admitted["ack"]])

            negative = launch("negative")
            out, err = negative.communicate(timeout=10)
            self.assertEqual(negative.returncode, 0, err)
            self.assertEqual(json.loads(out), {"cross_tenant_conflict": True,
                                               "other_peer_lookup": True})

            finisher = launch("settle")
            out, err = finisher.communicate(timeout=10)
            self.assertEqual(finisher.returncode, -signal.SIGKILL, err)
            self.assertEqual(json.loads(out), {"id": admitted["ack"],
                                               "state": "TASK_STATE_COMPLETED", "reply": "confirmed"})

            third = launch("terminal")
            out, err = third.communicate(timeout=10)
            self.assertEqual(third.returncode, 0, err)
            self.assertEqual(json.loads(out), {"id": admitted["ack"], "replay": True,
                                               "claimed": False, "overwrite": False,
                                               "state": "TASK_STATE_COMPLETED", "reply": "confirmed"})
            self.assertEqual(Path(trace).read_text().splitlines(), [admitted["ack"]])


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        unittest.main(verbosity=2)
