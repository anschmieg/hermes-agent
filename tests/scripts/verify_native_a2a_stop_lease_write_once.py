"""Adversarial SQLite regression: a STOP attempt lease is minted once.

FAKE backend only: no network, no Chief receipt, no remote STOP confirmation.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'plugins/platforms/a2a/ledger.py'
spec = importlib.util.spec_from_file_location('exact_sha_a2a_ledger', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Ledger = module.DurableTaskLedger
SCOPE = dict(peer='auth-peer', agent_slug='private', tenant='tenant-a', input_id='trusted-chief-input')

class StopLeaseWriteOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='a2a-stop-lease-')
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'tasks.sqlite'
        self.a = Ledger(self.db)
        self.record, replay = self.a.reserve(message_id='same-message', payload={'kind':'test'},
                                             context_id='ctx-a', **SCOPE)
        self.assertFalse(replay)
        self.task = self.record['task_id']
        self.assertIsNotNone(self.a.update(self.task, claim=True))
        self.claimed = self.a.claim_stop_attempt(self.task, **SCOPE)
        self.assertIsNotNone(self.claimed)

    def test_stop_lease_cannot_be_cleared_by_generic_update(self):
        before = self.a.get(self.task)
        self.assertIsNone(self.a.update(self.task, stop_attempt_state=''))
        self.assertEqual(self.a.get(self.task), before)
        self.assertIsNone(self.a.claim_stop_attempt(self.task, **SCOPE))

    def test_stop_lease_cannot_be_overwritten_by_generic_update(self):
        self.assertIsNone(self.a.update(self.task, stop_attempt_state='settled'))
        self.assertEqual(self.a.get(self.task)['stop_attempt_state'], 'uncertain')

    def test_forged_claim_cannot_clear_stop_lease(self):
        self.assertIsNone(self.a.update(self.task, claim=True, stop_attempt_state=''))
        self.assertIsNone(self.a.claim_stop_attempt(self.task, **SCOPE))

    def test_other_process_cannot_reopen_stop_lease(self):
        b = Ledger(self.db)
        self.assertIsNone(b.update(self.task, stop_attempt_state=None))
        self.assertIsNone(b.claim_stop_attempt(self.task, **SCOPE))
        self.assertEqual(self.a.get(self.task)['stop_attempt_state'], 'uncertain')

    def test_terminal_override_cannot_reopen_stop_lease(self):
        self.assertIsNotNone(self.a.update(self.task, state='TASK_STATE_COMPLETED', effect_state='settled'))
        self.assertIsNone(self.a.update(self.task, allow_terminal=True, stop_attempt_state=''))
        self.assertEqual(self.a.get(self.task)['stop_attempt_state'], 'uncertain')

    def test_completion_without_lease_mutation_still_works(self):
        self.assertIsNotNone(self.a.update(self.task, state='TASK_STATE_COMPLETED', effect_state='settled'))
        self.assertEqual(self.a.get(self.task)['state'], 'TASK_STATE_COMPLETED')
        self.assertIsNone(self.a.claim_stop_attempt(self.task, **SCOPE))

    def test_different_tenant_cannot_claim_stop(self):
        self.assertIsNone(self.a.claim_stop_attempt(self.task, **{**SCOPE, 'tenant':'tenant-b'}))
        self.assertEqual(self.a.get(self.task)['stop_attempt_state'], 'uncertain')

if __name__ == '__main__':
    unittest.main(verbosity=2)
