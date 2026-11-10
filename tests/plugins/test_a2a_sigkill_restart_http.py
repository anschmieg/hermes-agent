"""Process-bound native A2A crash/retry/tenant/reply-correlation regression.

A private worker serves the production A2A HTTP handler, adapter, TaskStore and
SQLite ledger. Parent kills that *OS process* after an HTTP ACK and a recorded
handler dispatch, then starts a fresh worker against the same database.
Only gateway inference and unrelated imports use existing test-only shims.
No live backend, credentials, or production session is accessed.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


HERE = Path(__file__).resolve()


def _worker(db: str, dispatch_log: str) -> None:
    # Import in the separate process only: these existing stdlib shims must not
    # pollute pytest's gateway modules in the parent interpreter.
    sys.path.insert(0, str(HERE.parents[1] / "scripts"))
    from verify_native_a2a_durable_stdlib import NativeHttpAcceptance
    from plugins.platforms.a2a import protocol
    from plugins.platforms.a2a.ledger import DurableTaskLedger

    harness = NativeHttpAcceptance("test_timeout_retry_and_late_terminal_receipt")
    harness.setUp()
    a = harness.adapter
    a.tasks = protocol.TaskStore(DurableTaskLedger(db))
    a._agents["private"] = {"slug": "private", "tenant": "tenant-a", "local": True,
                             "name": "private", "path": "/private",
                             "allowed_peers": {"alice", "bob"}}
    a._agents["foreign"] = {"slug": "foreign", "tenant": "tenant-b", "local": True,
                             "name": "foreign", "path": "/foreign",
                             "allowed_peers": {"alice", "bob"}}

    async def blocked_inference(event):
        # This marker is emitted only *after* the production adapter has
        # admitted/claimed and scheduled the actual gateway handler callback.
        with open(dispatch_log, "a", encoding="utf-8") as out:
            out.write(event.message_id + "\n")
            out.flush()
            os.fsync(out.fileno())
        await asyncio.sleep(3600)

    a.handle_message = blocked_inference
    print(f"READY {harness.server.server_address[1]}", flush=True)
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["op"] == "reply":
                rec = a.tasks.get(command["task_id"], "private", "tenant-a", "alice")
                assert rec is not None
                context = a._execution_context_id(a._agents["private"], "alice", rec["context_id"])
                asyncio.run(a.send(context, command["text"], reply_to=command.get("reply_to"),
                                   metadata={"notify": True}))
                print(json.dumps({"state": a.tasks.get(rec["task_id"])["state"]}), flush=True)
            elif command["op"] == "inspect":
                rec = a.tasks.get(command["task_id"], "private", "tenant-a", "alice")
                assert rec is not None
                print(json.dumps({"id": rec["task_id"], "state": rec["state"],
                                  "effect_state": rec["effect_state"],
                                  "input_id": rec["input_id"]}), flush=True)
            elif command["op"] == "stop":
                print("STOPPED", flush=True)
                break
            else:
                raise AssertionError(command)
    finally:
        harness.doCleanups()


@unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires real POSIX SIGKILL")
class NativeSigkillHttpRestart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / "a2a.sqlite3")
        self.trace = Path(self.tmp.name) / "dispatches.txt"
        self.workers: list[subprocess.Popen] = []
        self.addCleanup(self.cleanup_workers)

    def cleanup_workers(self):
        for proc in reversed(self.workers):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe:
                    pipe.close()

    def read_line(self, proc, timeout=12):
        readable, _, _ = select.select([proc.stdout], [], [], timeout)
        self.assertTrue(readable, f"worker did not respond (exit={proc.poll()})")
        line = proc.stdout.readline()
        self.assertTrue(line, f"worker exited early (exit={proc.poll()})")
        return line.strip()

    def start_worker(self):
        proc = subprocess.Popen([sys.executable, "-u", str(HERE), "--worker", self.db, str(self.trace)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1, cwd=str(HERE.parents[2]))
        self.workers.append(proc)
        ready = self.read_line(proc)
        self.assertTrue(ready.startswith("READY "), ready)
        return proc, "http://127.0.0.1:" + ready.split()[1] + "/"

    def command(self, proc, **payload):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        return json.loads(self.read_line(proc))

    def rpc(self, url, method, params, peer="alice"):
        data = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        request = urllib.request.Request(url, data=data, method="POST",
            headers={"Authorization": f"Bearer {peer}-token", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    @staticmethod
    def message(message_id, *, text="work", tenant="tenant-a"):
        return {"tenant": tenant, "configuration": {"returnImmediately": True},
                "message": {"messageId": message_id, "contextId": "shared-context",
                            "role": "ROLE_USER", "parts": [{"text": text}]}}

    def await_dispatches(self, count):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            ids = self.trace.read_text().splitlines() if self.trace.exists() else []
            if len(ids) >= count:
                self.assertEqual(len(ids), count, ids)
                return ids
            time.sleep(.025)
        self.fail(f"expected {count} real handler dispatches; observed {ids}")

    def test_sigkill_recovery_is_private_idempotent_and_reply_correlated(self):
        first_worker, base = self.start_worker()
        payload = self.message("original-unique")
        first = self.rpc(base, "SendMessage", payload)["result"]["task"]
        old_id = first["id"]
        self.assertEqual(first["status"]["state"], "TASK_STATE_WORKING")
        self.assertEqual(self.await_dispatches(1), [old_id])
        before = self.command(first_worker, op="inspect", task_id=old_id)
        self.assertEqual((before["state"], before["effect_state"]),
                         ("TASK_STATE_WORKING", "uncertain"))
        self.assertTrue(before["input_id"].startswith("a2a-msg:"))

        # SIGKILL a real admitted/ACKed worker; no cooperative teardown and no
        # _pop_pending() / synthetic exception stands in for process death.
        os.kill(first_worker.pid, signal.SIGKILL)
        self.assertEqual(first_worker.wait(timeout=6), -signal.SIGKILL)

        second_worker, base = self.start_worker()
        after = self.command(second_worker, op="inspect", task_id=old_id)
        self.assertEqual(after, before)
        replay = self.rpc(base, "SendMessage", payload)["result"]["task"]
        self.assertEqual(replay["id"], old_id)
        self.assertEqual(replay["status"]["state"], "TASK_STATE_WORKING")
        self.assertEqual(self.trace.read_text().splitlines(), [old_id])
        self.assertEqual(self.rpc(base, "GetTaskByMessageId",
                                  {"messageId": "original-unique", "tenant": "tenant-a"})
                         ["result"]["id"], old_id)

        # Negative controls: same identity/ID cannot pierce tenant scope;
        # distinct peer cannot read or cancel the accepted task.
        for peer, tenant in (("bob", "tenant-a"), ("alice", "tenant-b")):
            self.assertEqual(self.rpc(base, "GetTask", {"id": old_id, "tenant": tenant}, peer)
                             ["error"]["code"], -32001)
            self.assertEqual(self.rpc(base, "GetTaskByMessageId",
                                      {"messageId": "original-unique", "tenant": tenant}, peer)
                             ["error"]["code"], -32001)
        self.assertEqual(self.rpc(base, "SendMessage", self.message("original-unique", tenant="tenant-b"))
                         ["error"]["code"], -32053)
        self.assertEqual(self.rpc(base, "CancelTask", {"id": old_id, "tenant": "tenant-a"})
                         ["error"]["code"], -32002)

        newer = self.rpc(base, "SendMessage", self.message("new-unique"))["result"]["task"]
        new_id = newer["id"]
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(newer["status"]["state"], "TASK_STATE_WORKING")
        self.assertEqual(self.await_dispatches(2), [old_id, new_id])
        # Uncorrelated final and a correlated *old* final must not settle new.
        for reply_to in (None, old_id):
            outcome = self.command(second_worker, op="reply", task_id=new_id,
                                   reply_to=reply_to, text="stale unconfirmed result")
            self.assertEqual(outcome["state"], "TASK_STATE_WORKING")
            self.assertEqual(self.rpc(base, "GetTask", {"id": new_id, "tenant": "tenant-a"})
                             ["result"]["status"]["state"], "TASK_STATE_WORKING")
        self.assertEqual(self.rpc(base, "CancelTask", {"id": new_id, "tenant": "tenant-a"})
                         ["error"]["code"], -32002)
        settled = self.command(second_worker, op="reply", task_id=new_id,
                               reply_to=new_id, text="correlated final")
        self.assertEqual(settled["state"], "TASK_STATE_COMPLETED")
        got = self.rpc(base, "GetTask", {"id": new_id, "tenant": "tenant-a"})["result"]
        self.assertEqual(got["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(got["artifacts"][0]["parts"][0]["text"], "correlated final")
        self.assertEqual(self.rpc(base, "CancelTask", {"id": new_id, "tenant": "tenant-a"})
                         ["error"]["code"], -32002)
        self.assertEqual(self.rpc(base, "GetTask", {"id": old_id, "tenant": "tenant-a"})
                         ["result"]["status"]["state"], "TASK_STATE_WORKING")
        self.assertEqual(self.trace.read_text().splitlines(), [old_id, new_id])
        self.assertEqual(self.rpc(base, "SendMessage", self.message("new-unique"))
                         ["result"]["task"]["id"], new_id)
        self.assertEqual(self.trace.read_text().splitlines(), [old_id, new_id])
        second_worker.stdin.write('{"op":"stop"}\n')
        second_worker.stdin.flush()
        self.assertEqual(self.read_line(second_worker), "STOPPED")
        self.assertEqual(second_worker.wait(timeout=8), 0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
