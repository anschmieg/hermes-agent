"""Adversarial real-SQLite regression: task admission identity is immutable.

Runs only the exact on-disk a2a ledger source; no live Chief, remote peer,
or network. No backend stop confirmation can be inferred from these tests.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'plugins/platforms/a2a/ledger.py'
spec = importlib.util.spec_from_file_location('isolated_a2a_ledger', SOURCE)
assert spec is not None and spec.loader is not None
ledger_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger_module)
DurableTaskLedger = ledger_module.DurableTaskLedger

IMMUTABLE = {
    'context_id': 'context-forged',
    'peer': 'peer-forged',
    'agent_slug': 'agent-forged',
    'tenant': 'tenant-forged',
    'message_id': 'message-forged',
    'input_id': 'input-forged',
}

class LedgerImmutableScope(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='a2a-immutable-')
        self.addCleanup(temporary.cleanup)
        self.ledger = DurableTaskLedger(Path(temporary.name) / 'journal.sqlite')
        self.original, replay = self.ledger.reserve(
            peer='authenticated-peer', message_id='msg-once', payload={'text': 'hello'},
            context_id='ctx', agent_slug='test-agent', tenant='trusted-tenant',
            input_id='trusted-input-id')
        self.assertFalse(replay)
        self.task_id = self.original['task_id']

    def assert_immutable(self, field, proposed):
        updated = self.ledger.update(self.task_id, **{field: proposed})
        self.assertIsNone(updated, f'generic update must reject immutable {field}')
        self.assertEqual(self.ledger.get(self.task_id), self.original)
        self.assertEqual(self.ledger.get_by_message('authenticated-peer', 'msg-once'), self.original)

    def test_claim_with_forged_tenant_does_not_dispatch(self):
        self.assertIsNone(self.ledger.update(self.task_id, claim=True, tenant='forged-tenant'))
        self.assertEqual(self.ledger.get(self.task_id), self.original)

    def test_normal_claim_remains_available(self):
        record = self.ledger.update(self.task_id, claim=True)
        self.assertEqual((record['state'], record['effect_state']),
                         ('TASK_STATE_WORKING', 'uncertain'))

    def test_normal_completion_remains_available(self):
        self.assertIsNotNone(self.ledger.update(self.task_id, claim=True))
        result = self.ledger.update(self.task_id, state='TASK_STATE_COMPLETED',
                                    reply='done', effect_state='settled')
        self.assertEqual(result['state'], 'TASK_STATE_COMPLETED')

for field, proposed in IMMUTABLE.items():
    def make_test(f, v):
        def test(self):
            self.assert_immutable(f, v)
        return test
    setattr(LedgerImmutableScope, 'test_reject_mutation_of_' + field,
            make_test(field, proposed))

if __name__ == '__main__':
    unittest.main(verbosity=2)
