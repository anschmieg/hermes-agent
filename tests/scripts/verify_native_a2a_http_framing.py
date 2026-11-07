"""A2A HTTP framing regressions against the actual stdlib request handler.

Run standalone: python tests/scripts/verify_native_a2a_http_framing.py
Uses only an ephemeral loopback native HTTP fixture, never real credentials.
"""

import http.client
import json
import unittest
from urllib.parse import urlsplit

from verify_native_a2a_durable_stdlib import NativeHttpAcceptance


class NativeHttpFraming(unittest.TestCase):
    def setUp(self):
        fixture = NativeHttpAcceptance("test_return_immediately_persists_task_then_late_reply")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        url = urlsplit(fixture.base)
        self.host, self.port = url.hostname, url.port

    def request_with_length(self, content_length):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
        self.addCleanup(conn.close)
        conn.putrequest("POST", "/")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Authorization", "Bearer alice-token")
        conn.putheader("Content-Length", content_length)
        conn.endheaders()
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    def test_negative_content_length_rejected_without_waiting_for_eof(self):
        for malformed in ("-1", "-1024"):
            with self.subTest(length=malformed):
                status, body = self.request_with_length(malformed)
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["code"], -32700)

    def test_oversized_content_length_rejected_without_read(self):
        status, body = self.request_with_length("1048577")
        self.assertEqual(status, 413)
        self.assertEqual(body["error"]["code"], -32700)

    def test_valid_jsonrpc_content_length_still_works(self):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=2.0)
        self.addCleanup(conn.close)
        payload = json.dumps({
            "jsonrpc": "2.0", "id": 7, "method": "GetTask",
            "params": {"id": "missing"},
        })
        conn.request("POST", "/", payload, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer alice-token",
        })
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["error"]["code"], -32001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
