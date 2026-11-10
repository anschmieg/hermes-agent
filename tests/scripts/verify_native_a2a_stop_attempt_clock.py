"""Adversarial SQLite/SIGKILL regression: remote STOP lease time is write-once.

Test-only local processes and fake effects; no Chief, network or backend STOP.
Run: python tests/scripts/verify_native_a2a_stop_attempt_clock.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'plugins/platforms/a2a/ledger.py'
spec = importlib.util.spec_from_file_location('exact_sha_stop_clock_ledger', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Ledger = module.DurableTaskLedger
SCOPE = dict(peer='auth-peer', agent_slug='private', tenant='tenant-a',
             input_id='original-chief-input')


def _worker(db: str, task_id: str) -> None:
    lease = Ledger(db).claim_stop_attempt(task_id, **SCOPE)
    assert lease and lease['stop_attempt_state'] == 'uncertain'
    os.kill(os.getpid(), signal.SIGKILL)


class StopAttemptClockSafety(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='a2a-stop-clock-')
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / 'journal.sqlite'
        self.ledger = Ledger(self.db)
        self.payload = {'part': 'fake local effect'}
        record, replay = self.ledger.reserve(message_id='original-message',
                                             payload=self.payload,
                                             context_id='context-a', **SCOPE)
        self.assertFalse(replay)
        self.task = record['task_id']
        self.assertIsNotNone(self.ledger.update(self.task, claim=True))

    def lease(self):
        claimed = self.ledger.claim_stop_attempt(self.task, **SCOPE)
        self.assertIsNotNone(claimed)
        return claimed['stop_attempted_at']

    @unittest.skipUnless(os.name == 'posix' and hasattr(signal, 'SIGKILL'), 'POSIX only')
    def test_sigkill_retry_cannot_rebase_first_stop_attempt_time(self):
        proc = subprocess.run([sys.executable, '-u', __file__, '--worker',
                               str(self.db), self.task],
                              capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stderr)
        restarted = Ledger(self.db)
        original = restarted.get(self.task)
        old = original['stop_attempted_at']
        self.assertIsNone(restarted.update(self.task, stop_attempted_at=old + 600))
        self.assertEqual(restarted.get(self.task), original)
        self.assertIsNone(restarted.claim_stop_attempt(self.task, **SCOPE))
        same, replay = restarted.reserve(message_id='original-message',
                                         payload=self.payload,
                                         context_id='context-a', **SCOPE)
        self.assertTrue(replay)
        self.assertEqual(same, original)
        self.assertIsNone(restarted.update(self.task, claim=True))

    def test_stale_worker_cannot_preseed_stop_receipt_clock_before_stop(self):
        before = self.ledger.get(self.task)
        self.assertNotIn('stop_attempted_at', before)
        self.assertIsNone(self.ledger.update(self.task, stop_attempted_at=1.0))
        self.assertEqual(self.ledger.get(self.task), before)
        stamped = self.lease()
        self.assertGreater(stamped, 1.0)

    def test_allow_terminal_override_cannot_redate_claimed_stop(self):
        stamped = self.lease()
        self.assertIsNotNone(self.ledger.update(
            self.task, state='TASK_STATE_COMPLETED', effect_state='settled'))
        terminal = self.ledger.get(self.task)
        self.assertIsNone(self.ledger.update(self.task, allow_terminal=True,
                                             stop_attempted_at=stamped + 1000))
        self.assertEqual(self.ledger.get(self.task), terminal)

    def test_legitimate_completion_keeps_original_stop_clock(self):
        stamped = self.lease()
        settled = self.ledger.update(self.task, state='TASK_STATE_COMPLETED',
                                     reply='owner result', effect_state='settled')
        self.assertIsNotNone(settled)
        self.assertEqual(settled['stop_attempted_at'], stamped)
        self.assertEqual(Ledger(self.db).get(self.task), settled)
        self.assertIsNone(self.ledger.claim_stop_attempt(self.task, **SCOPE))


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        _worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
