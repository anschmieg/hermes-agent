"""Standalone native A2A v1.0 over HTTP + SQLite recovery acceptance tests.

Uses the actual Hermes protocol, ledger and HTTP adapter with only unrelated
gateway dependencies replaced by test shims. No live endpoint or credentials.

    python tests/scripts/verify_native_a2a_durable_stdlib.py
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
import unittest
import urllib.request

import verify_native_a2a_peer_scope_stdlib as shim  # stdlib import shim + real modules
from plugins.platforms.a2a import protocol, security
from plugins.platforms.a2a import adapter as native
from plugins.platforms.a2a.ledger import DurableTaskLedger, IdempotencyConflict
from http.server import ThreadingHTTPServer

PYTHON = sys.executable or shutil.which("python3") or "/usr/bin/python3"


class SqliteDurability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "native-a2a.sqlite3"
        self.payload = {"message": {"role": "ROLE_USER", "parts": [{"text": "hello"}], "messageId": "id-1"}}

    def reserve(self, **kw):
        return DurableTaskLedger(self.path).reserve(peer="alice", message_id="id-1",
                                                    payload=self.payload, context_id="ctx-1", **kw)

    def test_restart_and_scope_conflict(self):
        rec, replay = self.reserve(agent_slug="agent-a", tenant="tenant-a")
        self.assertFalse(replay)
        same, replay = self.reserve(agent_slug="agent-a", tenant="tenant-a")
        self.assertTrue(replay)
        self.assertEqual(same["task_id"], rec["task_id"])
        self.assertEqual(same["state"], protocol.STATE_SUBMITTED)
        with self.assertRaises(IdempotencyConflict):
            self.reserve(agent_slug="agent-b", tenant="tenant-a")
        with self.assertRaises(IdempotencyConflict):
            self.reserve(agent_slug="agent-a", tenant="tenant-b")
        with self.assertRaises(IdempotencyConflict):
            DurableTaskLedger(self.path).reserve(peer="alice", message_id="id-1",
                payload={"message": {"parts": [{"text": "CHANGED"}]}},
                context_id="ctx-1", agent_slug="agent-a", tenant="tenant-a")

    def test_claim_once_then_durable_terminal(self):
        task, _ = self.reserve()
        ledger = DurableTaskLedger(self.path)
        self.assertIsNotNone(ledger.update(task["task_id"], claim=True))
        self.assertIsNone(DurableTaskLedger(self.path).update(task["task_id"], claim=True))
        recovered = DurableTaskLedger(self.path).get(task["task_id"])
        self.assertEqual(recovered["state"], protocol.STATE_WORKING)
        self.assertEqual(recovered["effect_state"], "uncertain")
        self.assertEqual(DurableTaskLedger(self.path).update(task["task_id"],
            state=protocol.STATE_COMPLETED, reply="final")["reply"], "final")
        self.assertIsNone(DurableTaskLedger(self.path).update(task["task_id"],
            state=protocol.STATE_FAILED, reply="incorrect overwrite"))
        self.assertEqual(DurableTaskLedger(self.path).get(task["task_id"])["reply"], "final")

    def test_two_processes_racing_same_message_id(self):
        code = (
            "from plugins.platforms.a2a.ledger import DurableTaskLedger; import sys; "
            "r,replayed=DurableTaskLedger(sys.argv[1]).reserve(peer='alice',message_id='id-1',"
            "payload={'message': {'role': 'ROLE_USER', 'parts': [{'text': 'hello'}], 'messageId': 'id-1'}},"
            "context_id='ctx-1'); print(r['task_id'],replayed, flush=True)"
        )
        jobs = [subprocess.Popen([PYTHON, "-c", code, str(self.path)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(6)]
        output = [p.communicate(timeout=30) for p in jobs]
        self.assertTrue(all(p.returncode == 0 for p in jobs), output)
        self.assertEqual(len({out.split()[0] for out, _err in output}), 1)
        self.assertEqual(sum(out.split()[1] == "False" for out, _err in output), 1)

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "requires SIGKILL")
    def test_real_sigkill_after_dispatch_claim_no_reexecution(self):
        code = (
            "from plugins.platforms.a2a.ledger import DurableTaskLedger; import os,signal,sys; "
            "l=DurableTaskLedger(sys.argv[1]); "
            "r,_=l.reserve(peer='alice',message_id='id-1',"
            "payload={'message': {'role': 'ROLE_USER', 'parts': [{'text': 'hello'}], 'messageId': 'id-1'}},"
            "context_id='ctx-1'); l.update(r['task_id'],claim=True); "
            "os.kill(os.getpid(),signal.SIGKILL)"
        )
        p = subprocess.run([PYTHON, "-c", code, str(self.path)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        self.assertEqual(p.returncode, -signal.SIGKILL, p.stderr)
        restored, replay = self.reserve()
        self.assertTrue(replay)
        self.assertEqual(restored["state"], protocol.STATE_WORKING)
        self.assertEqual(restored["effect_state"], "uncertain")
        self.assertIsNone(DurableTaskLedger(self.path).update(restored["task_id"], claim=True))

    def test_orphan_sweep_preserves_uncertain_effects(self):
        store = protocol.TaskStore(DurableTaskLedger(self.path))
        rec, _ = store.admit("alice", "id-1", self.payload, "ctx-1")
        self.assertTrue(store.claim_dispatch(rec["task_id"]))
        self.assertEqual(store.fail_orphans(timeout_seconds=-1), [])
        self.assertEqual(protocol.TaskStore(DurableTaskLedger(self.path)).get(rec["task_id"])["state"],
                         protocol.STATE_WORKING)

    def test_reserved_before_dispatch_remains_recoverable(self):
        store = protocol.TaskStore(DurableTaskLedger(self.path))
        rec, _ = store.admit("alice", "id-1", self.payload, "ctx-1")
        self.assertEqual(store.fail_orphans(timeout_seconds=-1), [])
        restarted = protocol.TaskStore(DurableTaskLedger(self.path))
        replay, already = restarted.admit("alice", "id-1", self.payload, "ctx-1")
        self.assertTrue(already)
        self.assertEqual(replay["task_id"], rec["task_id"])
        self.assertTrue(restarted.claim_dispatch(rec["task_id"]))
        self.assertFalse(store.claim_dispatch(rec["task_id"]))

    def test_push_config_survives_restart_and_is_claimed_once(self):
        store = protocol.TaskStore(DurableTaskLedger(self.path))
        rec, _ = store.admit("alice", "id-1", self.payload, "ctx-1", "a", "t")
        self.assertIsNone(store.set_push_config(rec["task_id"], "https://example.com", "a", "t", peer="bob"))
        cfg = store.set_push_config(rec["task_id"], "https://example.com", "a", "t", peer="alice")
        self.assertIsNotNone(cfg)
        restored = protocol.TaskStore(DurableTaskLedger(self.path))
        self.assertEqual(restored.get_push_config(rec["task_id"], agent_slug="a", tenant="t", peer="alice")["configId"],
                         cfg["configId"])
        self.assertEqual(restored.pop_push_url(rec["task_id"]), "https://example.com")
        self.assertEqual(store.pop_push_url(rec["task_id"]), "")


class NativeHttpAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.adapter = native.A2AAdapter.__new__(native.A2AAdapter)
        a = self.adapter
        a.tasks = protocol.TaskStore(DurableTaskLedger(Path(self.tmp.name) / "tasks.db"))
        a._agents = {"": {"slug": "", "tenant": "", "local": True, "name": "hermes-test", "path": ""}}
        a._public_url, a.host, a.port, a.agent_name = "", "127.0.0.1", 0, "hermes-test"
        a._security_context = SimpleNamespace(
            authenticate=lambda header, ip: {"Bearer alice-token": "alice", "Bearer bob-token": "bob"}.get(header),
            localhost_only=lambda: False,
            is_trusted_peer=lambda peer: peer in {"alice", "bob"},
            sign_push_payload=lambda payload: "",
        )
        a._rate_limiter, a._turns = SimpleNamespace(allow=lambda peer: True), protocol.TurnTracker()
        a._pending, a._pending_order, a._active_tasks = {}, {}, set()
        a._pending_lock = threading.Lock()
        a._message_handler = object()
        self.count = 0
        self.errors = []
        self.delay = .17
        a.build_source = lambda **kwargs: SimpleNamespace(**kwargs)
        a._register_inline_push = lambda *args, **kwargs: None
        a._send_push_notification = lambda *args, **kwargs: None
        self.orig_audit, self.orig_persist, self.orig_logged, self.orig_redact = (
            security.audit, protocol.persist_message, protocol.logged_input, security.redact_outbound)
        security.audit = lambda *args, **kwargs: None
        security.redact_outbound = lambda text: text
        protocol.persist_message = lambda *args, **kwargs: None
        protocol.logged_input = lambda *args, **kwargs: False
        self.addCleanup(self.restore)

        async def fake_handle(event):
            self.count += 1
            try:
                await asyncio.sleep(self.delay)
                await a.send(event.source.chat_id, "ACK", metadata={"notify": True})
            except Exception:
                self.errors.append(traceback.format_exc())

        # Keep production adapter/protocol and real HTTP, only replace the
        # expensive gateway inference handler by a deterministic fake agent.
        class MessageEvent:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        native.MessageEvent = MessageEvent
        native.MessageType.TEXT = "text"
        native.SendResult = lambda **kwargs: SimpleNamespace(**kwargs)
        a.handle_message = fake_handle
        self.loop = asyncio.new_event_loop()
        a._loop = self.loop
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), native.A2ARequestHandler)
        self.server.adapter = a
        self.server.daemon_threads = True
        self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.http_thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def restore(self):
        security.audit, protocol.persist_message, protocol.logged_input = self.orig_audit, self.orig_persist, self.orig_logged
        security.redact_outbound = self.orig_redact
        self.server.shutdown()
        self.server.server_close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=3)

    def post(self, method, params, peer="alice", request_id=1):
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}).encode()
        request = urllib.request.Request(self.base, data=body, method="POST",
            headers={"Authorization": f"Bearer {peer}-token", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.load(response)

    def stream(self, method, params, peer="alice"):
        body = json.dumps({"jsonrpc": "2.0", "id": 17, "method": method, "params": params}).encode()
        request = urllib.request.Request(self.base, data=body, method="POST",
            headers={"Authorization": f"Bearer {peer}-token", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=8) as response:
            self.assertEqual(response.headers.get("Content-Type"), "text/event-stream")
            return response.read().decode("utf-8")

    def message(self, message_id="msg-1"):
        return {"message": {"role": protocol.ROLE_USER, "parts": [{"text": "do work"}], "messageId": message_id}}

    def test_timeout_retry_and_late_terminal_receipt(self):
        original_timeout = native._reply_timeout
        native._reply_timeout = lambda: .03
        try:
            first = self.post("SendMessage", self.message())
            self.assertEqual(first["result"]["task"]["status"]["state"], protocol.STATE_WORKING)
            task_id = first["result"]["task"]["id"]
            replay = self.post("SendMessage", self.message())
            self.assertEqual(replay["result"]["task"]["id"], task_id)
            self.assertEqual(self.count, 1)
            self.assertEqual(self.post("GetTask", {"id": task_id}, peer="bob")["error"]["code"],
                             protocol.ERR_TASK_NOT_FOUND)
            self.assertEqual(self.post("CancelTask", {"id": task_id})["error"]["code"],
                             protocol.ERR_TASK_NOT_CANCELABLE)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                got = self.post("GetTask", {"id": task_id})["result"]
                if got["status"]["state"] == protocol.STATE_COMPLETED:
                    break
                time.sleep(.035)
            self.assertEqual(got["status"]["state"], protocol.STATE_COMPLETED,
                             f"callback errors={self.errors} pending={list(self.adapter._pending)}")
            self.assertEqual(protocol.extract_text(got["artifacts"][0]), "ACK")
            self.assertEqual(self.count, 1)
            after_restart = protocol.TaskStore(DurableTaskLedger(self.adapter.tasks._ledger.path))
            self.assertEqual(after_restart.get(task_id)["state"], protocol.STATE_COMPLETED)
        finally:
            native._reply_timeout = original_timeout

    def test_concurrent_duplicate_messages_only_dispatch_once(self):
        native._reply_timeout, previous = (lambda: .03), native._reply_timeout
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                responses = list(pool.map(lambda _i: self.post("SendMessage", self.message("concurrent")), range(6)))
            task_ids = {r["result"]["task"]["id"] for r in responses}
            self.assertEqual(len(task_ids), 1)
            self.assertEqual(self.count, 1)
        finally:
            native._reply_timeout = previous

    def test_disconnected_sse_keepalive_never_marks_backend_failed(self):
        prior = native._SSE_KEEPALIVE
        native._SSE_KEEPALIVE = .01
        try:
            callback_attempts = []
            def client_disconnected():
                callback_attempts.append(True)
                raise BrokenPipeError("client disconnected")
            state, _ = native.A2AAdapter._await_future(
                Future(), time.time() + 2, client_disconnected,
                (protocol.STATE_WORKING, "unverified"))
            self.assertEqual(state, protocol.STATE_WORKING)
            self.assertEqual(len(callback_attempts), 1)
        finally:
            native._SSE_KEEPALIVE = prior

    def test_conflicting_http_retry_rejected_without_second_dispatch(self):
        original_timeout = native._reply_timeout
        native._reply_timeout = lambda: .03
        try:
            first = self.post("SendMessage", self.message("conflict"))
            self.assertIn("task", first["result"])
            changed = self.message("conflict")
            changed["message"]["parts"][0]["text"] = "different command"
            conflict = self.post("SendMessage", changed)
            self.assertEqual(conflict["error"]["code"], protocol.ERR_IDEMPOTENCY_CONFLICT)
            missing = self.post("SendMessage", {"message": {"role": protocol.ROLE_USER,
                "parts": [{"text": "missing message id"}]}})
            self.assertEqual(missing["error"]["code"], protocol.ERR_INVALID_PARAMS)
            self.assertEqual(self.count, 1)
            self.assertEqual(self.post("ListTasks", {}, peer="bob")["result"]["tasks"], [])
        finally:
            native._reply_timeout = original_timeout

    def test_read_only_recovery_by_message_id_cannot_cross_peers(self):
        params = self.message("recover-lost-response")
        params["configuration"] = {"returnImmediately": True}
        task = self.post("SendMessage", params)["result"]["task"]
        found = self.post("GetTaskByMessageId", {"messageId": "recover-lost-response"})["result"]
        self.assertEqual(found["id"], task["id"])
        self.assertEqual(self.post("GetTaskByMessageId", {"messageId": "recover-lost-response"},
                                   peer="bob")["error"]["code"], protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(self.post("GetTaskByMessageId", {"messageId": "nonexistent"})["error"]["code"],
                         protocol.ERR_TASK_NOT_FOUND)
        self.assertEqual(protocol.TaskStore(DurableTaskLedger(self.adapter.tasks._ledger.path)).get_by_message(
            "alice", "recover-lost-response")["task_id"], task["id"])

    def test_sse_timeout_stays_working_then_subscribe_shows_settlement(self):
        original_timeout = native._reply_timeout
        original_keepalive = native._SSE_KEEPALIVE
        native._reply_timeout = lambda: .03
        native._SSE_KEEPALIVE = .01
        try:
            stream = self.stream("SendStreamingMessage", self.message("stream-one"))
            self.assertIn(protocol.STATE_WORKING, stream)
            self.assertNotIn(protocol.STATE_FAILED, stream)
            self.assertNotIn(": done", stream,
                             "an unverified WORKING task must not send an SSE completion marker")
            tasks = self.post("ListTasks", {})["result"]["tasks"]
            self.assertEqual(len(tasks), 1)
            task_id = tasks[0]["id"]
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if self.post("GetTask", {"id": task_id})["result"]["status"]["state"] == protocol.STATE_COMPLETED:
                    break
                time.sleep(.035)
            else:
                self.fail("streamed task never settled after late response")
            subscription = self.stream("SubscribeToTask", {"id": task_id})
            self.assertIn(protocol.STATE_COMPLETED, subscription)
            self.assertIn('"task":', subscription,
                          "SubscribeToTask must begin with the persisted Task snapshot")
            self.assertEqual(self.count, 1)
        finally:
            native._reply_timeout = original_timeout
            native._SSE_KEEPALIVE = original_keepalive

    def test_subscribe_nonterminal_starts_with_task_and_does_not_claim_done(self):
        original_timeout = native._reply_timeout
        original_keepalive = native._SSE_KEEPALIVE
        native._reply_timeout = lambda: .03
        native._SSE_KEEPALIVE = .01
        try:
            params = self.message("subscribe-still-working")
            params["configuration"] = {"returnImmediately": True}
            task_id = self.post("SendMessage", params)["result"]["task"]["id"]
            stream = self.stream("SubscribeToTask", {"id": task_id})
            self.assertIn('"task":', stream)
            self.assertIn(protocol.STATE_WORKING, stream)
            self.assertNotIn(": done", stream,
                             "an active task cannot be presented as a terminal stream")
            self.assertEqual(self.post("GetTask", {"id": task_id})["result"]["id"], task_id)
        finally:
            native._reply_timeout = original_timeout
            native._SSE_KEEPALIVE = original_keepalive

    def test_return_immediately_persists_task_then_late_reply(self):
        params = self.message("return-now")
        params["configuration"] = {"returnImmediately": True}
        started = time.monotonic()
        result = self.post("SendMessage", params)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, .15, "A2A HTTP send blocked on the agent despite returnImmediately")
        task = result["result"]["task"]
        self.assertEqual(task["status"]["state"], protocol.STATE_WORKING)
        task_id = task["id"]
        for _ in range(80):
            got = self.post("GetTask", {"id": task_id})["result"]
            if got["status"]["state"] == protocol.STATE_COMPLETED:
                break
            time.sleep(.035)
        self.assertEqual(got["status"]["state"], protocol.STATE_COMPLETED)
        self.assertEqual(self.count, 1)
        retry = self.post("SendMessage", params)["result"]["task"]
        self.assertEqual(retry["id"], task_id)

    def test_forwarded_owner_return_immediately_claims_once(self):
        self.adapter._agents["dev"] = {"slug": "dev", "tenant": "dev", "path": "/dev",
                                      "local": False, "profile": "dev", "name": "dev",
                                      "allowed_peers": {"alice"}}
        calls = []
        def owner(agent, peer, context_id, framed, *, input_id):
            calls.append(input_id)
            time.sleep(.15)
            return "OWNER_ACK", protocol.STATE_COMPLETED
        self.adapter._forward_to_profile = owner
        params = self.message("forwarded-one")
        params["tenant"] = "dev"
        params["configuration"] = {"returnImmediately": True}
        started = time.monotonic()
        response = self.post("SendMessage", params)
        self.assertLess(time.monotonic() - started, .15)
        task_id = response["result"]["task"]["id"]
        self.assertEqual(self.post("SendMessage", params)["result"]["task"]["id"], task_id)
        for _ in range(80):
            got = self.post("GetTask", {"id": task_id, "tenant": "dev"})["result"]
            if got["status"]["state"] == protocol.STATE_COMPLETED:
                break
            time.sleep(.035)
        self.assertEqual(got["status"]["state"], protocol.STATE_COMPLETED)
        self.assertEqual(calls, [calls[0]])
        self.assertEqual(protocol.extract_text(got["artifacts"][0]), "OWNER_ACK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
