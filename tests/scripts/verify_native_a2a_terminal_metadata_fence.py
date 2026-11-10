"""Real-SQLite SIGKILL regression: privileged metadata updates never rewrite outcomes.

Runs the exact A2A ledger source with a synthetic local writer and no remote
Chief/owner transport, backend STOP, network or launch-capable startup.
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

SOURCE = Path(__file__).resolve().parents[2] / "plugins/platforms/a2a/ledger.py"
spec = importlib.util.spec_from_file_location("a2a_exact_ledger", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Ledger = module.DurableTaskLedger
SCOPE = dict(peer="trusted-peer", agent_slug="agent-a", tenant="tenant-a",
             input_id="a2a-msg:" + "a" * 64)
TERMINAL = {"completed": "TASK_STATE_COMPLETED", "failed": "TASK_STATE_FAILED"}


def committed_writer(db: str, task_id: str, outcome: str) -> None:
    ledger = Ledger(db)
    assert ledger.update(task_id, claim=True)
    assert ledger.update(task_id, state=TERMINAL[outcome], reply="authoritative:" + outcome,
                         effect_state="settled", completed_at=123.0)
    os.kill(os.getpid(), signal.SIGKILL)


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires real SIGKILL")
class TerminalMetadataFence(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="a2a-terminal-metadata-")
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "tasks.sqlite3"
        self.ledger = Ledger(self.path)
        self.rec, replay = self.ledger.reserve(
            message_id="msg-1", payload={"text": "once"}, context_id="ctx-1", **SCOPE)
        self.assertFalse(replay)
        self.task = self.rec["task_id"]

    def crash_after_commit(self, outcome="completed"):
        p = subprocess.run([sys.executable, "-u", __file__, "--writer", str(self.path),
                            self.task, outcome], cwd=SOURCE.parents[3],
                           capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, -signal.SIGKILL, p.stderr)
        self.ledger = Ledger(self.path)
        result = self.ledger.get(self.task)
        self.assertEqual((result["state"], result["reply"], result["effect_state"]),
                         (TERMINAL[outcome], "authoritative:" + outcome, "settled"))
        return result

    def test_stale_failure_cannot_replace_completed_after_restart(self):
        original = self.crash_after_commit("completed")
        self.assertIsNone(self.ledger.update(
            self.task, allow_terminal=True, state="TASK_STATE_FAILED",
            reply="stale generation", effect_state="settled"))
        self.assertEqual(self.ledger.get(self.task), original)

    def test_stale_success_cannot_replace_failed_after_restart(self):
        original = self.crash_after_commit("failed")
        self.assertIsNone(self.ledger.update(
            self.task, allow_terminal=True, state="TASK_STATE_COMPLETED",
            reply="old delayed receipt", effect_state="settled"))
        self.assertEqual(self.ledger.get(self.task), original)

    def test_privileged_metadata_path_cannot_forge_cancel(self):
        original = self.crash_after_commit("completed")
        self.assertIsNone(self.ledger.update(
            self.task, allow_terminal=True, state="TASK_STATE_CANCELED",
            reply="unverified fake stop", effect_state="settled"))
        self.assertEqual(self.ledger.get(self.task), original)

    def test_allow_terminal_push_metadata_remains_supported(self):
        original = self.crash_after_commit()
        updated = self.ledger.update(self.task, allow_terminal=True,
                                     push_url="https://example.invalid/push",
                                     push_config_id="cfg-test")
        self.assertIsNotNone(updated)
        for key in ("state", "reply", "effect_state", "input_id", "tenant", "peer"):
            self.assertEqual(updated[key], original[key])
        self.assertEqual(updated["push_config_id"], "cfg-test")

    def test_normal_stale_update_and_replay_remain_fenced(self):
        original = self.crash_after_commit()
        self.assertIsNone(self.ledger.update(self.task, state="TASK_STATE_FAILED"))
        self.assertIsNone(self.ledger.update(self.task, claim=True))
        replayed, reused = self.ledger.reserve(
            message_id="msg-1", payload={"text": "once"}, context_id="ctx-1", **SCOPE)
        self.assertTrue(reused)
        self.assertEqual(replayed, original)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--writer":
        committed_writer(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
