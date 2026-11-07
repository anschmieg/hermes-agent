"""Adversarial isolation checks for externally supplied A2A context IDs.

Run: python tests/scripts/verify_native_a2a_context_isolation.py
Tests actual Hermes JSON-RPC HTTP with only the inference backend mocked.
"""
from __future__ import annotations

import time
import unittest

from verify_native_a2a_durable_stdlib import NativeHttpAcceptance
from plugins.platforms.a2a import protocol, security


class PeerContextIsolation(NativeHttpAcceptance):
    def test_same_context_id_does_not_share_gateway_session_or_history(self):
        sources = []
        history_contexts = []
        orig_source = self.adapter.build_source
        self.adapter.build_source = lambda **kw: (sources.append(kw), orig_source(**kw))[1]
        protocol.persist_message = lambda context, *args, **kw: history_contexts.append(context)

        def send(peer, message_id, tenant=""):
            params = self.message(message_id)
            params["message"]["contextId"] = "client-chosen-same-id"
            params["configuration"] = {"returnImmediately": True}
            if tenant:
                params["tenant"] = tenant
            return self.post("SendMessage", params, peer=peer)["result"]["task"]

        first = send("alice", "msg-a")
        second = send("bob", "msg-b")
        self.assertEqual(first["contextId"], second["contextId"])
        self.assertEqual(first["contextId"], "client-chosen-same-id")
        self.assertEqual(len(sources), 2)
        self.assertNotEqual(sources[0]["chat_id"], sources[1]["chat_id"],
                            "two authenticated peers must never share a Hermes session")
        self.assertEqual(len(history_contexts), 2)
        self.assertNotEqual(history_contexts[0], history_contexts[1],
                            "two peers must not share conversation JSONL")
        # Verify async replies also persist under the same isolated internal
        # context, and public Task receipts keep the caller's contextId.
        for task, peer in ((first, "alice"), (second, "bob")):
            for _ in range(90):
                receipt = self.post("GetTask", {"id": task["id"]}, peer=peer)["result"]
                if receipt["status"]["state"] == protocol.STATE_COMPLETED:
                    break
                time.sleep(.03)
            else:
                self.fail(f"{peer} task never settled")
            self.assertEqual(receipt["contextId"], "client-chosen-same-id")
        self.assertEqual(len(history_contexts), 4)
        self.assertEqual(set(history_contexts), {x["chat_id"] for x in sources})

    def test_same_context_id_across_served_tenants_isolated(self):
        sources = []
        orig_source = self.adapter.build_source
        self.adapter.build_source = lambda **kw: (sources.append(kw), orig_source(**kw))[1]
        self.adapter._agents["dev"] = {"slug": "dev", "tenant": "dev", "path": "/dev",
                                       "local": True, "name": "dev", "allowed_peers": {"alice"}}
        for tenant, msg_id in (("", "same-root"), ("dev", "same-dev")):
            params = self.message(msg_id)
            params["message"]["contextId"] = "same-context"
            params["configuration"] = {"returnImmediately": True}
            if tenant:
                params["tenant"] = tenant
            result = self.post("SendMessage", params, peer="alice")["result"]["task"]
            self.assertEqual(result["contextId"], "same-context")
        self.assertEqual(len(sources), 2)
        self.assertNotEqual(sources[0]["chat_id"], sources[1]["chat_id"],
                            "separate served tenants must never share Hermes session")

    def test_foreign_peer_cannot_route_into_another_tenant(self):
        import urllib.error
        self.adapter._agents["dev"] = {"slug": "dev", "tenant": "dev", "path": "/dev",
                                       "local": True, "name": "dev", "allowed_peers": {"bob"}}
        params = self.message("tenant-bypass")
        params["tenant"] = "dev"
        params["configuration"] = {"returnImmediately": True}
        with self.assertRaises(urllib.error.HTTPError) as err:
            self.post("SendMessage", params, peer="alice")
        self.assertEqual(err.exception.code, 403)
        self.assertEqual(self.count, 0)
        permitted = self.post("SendMessage", params, peer="bob")["result"]["task"]
        self.assertEqual(permitted["contextId"] != "", True)


class BearerPrincipalIsolation(unittest.TestCase):
    def test_shared_bearer_is_one_principal_not_reverse_proxy_ip(self):
        sec = security.A2ASecurityContext(
            bearer_token="some-shared-secret", peer_tokens=(),
            trusted_peers=frozenset(), allow_all_users=True,
            requested_host="0.0.0.0", push_secret="")
        self.assertEqual(sec.authenticate("Bearer some-shared-secret", "10.0.0.1"),
                         sec.authenticate("Bearer some-shared-secret", "10.0.0.2"))
        self.assertNotIn("ip:", sec.authenticate("Bearer some-shared-secret", "10.0.0.1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
