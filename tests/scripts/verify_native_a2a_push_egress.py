"""Fail-closed A2A push egress: tested without real DNS, sockets or credentials.

python tests/scripts/verify_native_a2a_push_egress.py
"""
from __future__ import annotations

import socket
import unittest
from types import SimpleNamespace
from unittest import mock

# This file's companion harness substitutes only the unrelated gateway runtime.
import verify_native_a2a_durable_stdlib  # noqa: F401
from plugins.platforms.a2a import security
from plugins.platforms.a2a.adapter import A2AAdapter


class PushEgressSafety(unittest.TestCase):
    def _send(self, url: str):
        tasks = SimpleNamespace(pop_push_url=lambda _id: url)
        ctx = SimpleNamespace(localhost_only=lambda: False, sign_push_payload=lambda _: "")
        adapter = SimpleNamespace(tasks=tasks, _security_context=ctx)
        return A2AAdapter._send_push_notification(adapter, "t1", "ctx1", "done", "TASK_STATE_COMPLETED")

    @staticmethod
    def _dns(*addresses):
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET,
                 socket.SOCK_STREAM, 6, "", (ip, 443 if ":" in ip else 80))
                for ip in addresses]

    def test_private_dns_cannot_reach_http_transport(self):
        with mock.patch("socket.getaddrinfo", return_value=self._dns("127.0.0.1")), \
             mock.patch("urllib.request.urlopen") as old_transport, \
             mock.patch("socket.create_connection") as new_transport:
            self._send("http://internal.example.test/metadata")
            old_transport.assert_not_called()
            new_transport.assert_not_called()

    def test_mixed_public_private_dns_fails_closed(self):
        with mock.patch("socket.getaddrinfo", return_value=self._dns("1.1.1.1", "10.1.2.3")):
            self.assertIsNone(security.resolve_safe_callback_target(
                "http://mixed.example.test/callback", localhost_mode=False))

    def test_dns_failure_fails_closed(self):
        with mock.patch("socket.getaddrinfo", side_effect=OSError("resolver unavailable")):
            self.assertIsNone(security.resolve_safe_callback_target(
                "https://public.example.test/callback", localhost_mode=False))

    def test_public_address_pinned_and_no_redirect_followed(self):
        connection = mock.Mock()
        connection.getresponse.return_value.status = 302
        with mock.patch("socket.getaddrinfo", return_value=self._dns("1.1.1.1")), \
             mock.patch("socket.create_connection", return_value=mock.Mock()) as dial, \
             mock.patch("http.client.HTTPConnection", return_value=connection), \
             mock.patch("urllib.request.urlopen") as unpinned:
            self._send("http://public.example.test:80/callback?q=1")
            dial.assert_called_once_with(("1.1.1.1", 80), timeout=10)
            connection.request.assert_called_once()
            args, kwargs = connection.request.call_args
            self.assertEqual(args[0], "POST")
            self.assertEqual(args[1], "/callback?q=1")
            self.assertEqual(kwargs["headers"]["Host"], "public.example.test")
            unpinned.assert_not_called()
            connection.close.assert_called_once()

    def test_public_https_preserves_sni_and_certificate_validation(self):
        connection = mock.Mock()
        connection.getresponse.return_value.status = 204
        tls = mock.Mock()
        with mock.patch("socket.getaddrinfo", return_value=self._dns("1.1.1.1")), \
             mock.patch("socket.create_connection", return_value=mock.Mock()) as dial, \
             mock.patch("ssl.create_default_context", return_value=tls) as context, \
             mock.patch("http.client.HTTPSConnection", return_value=connection), \
             mock.patch("urllib.request.urlopen") as unpinned:
            self._send("https://public.example.test/hook")
            dial.assert_called_once_with(("1.1.1.1", 443), timeout=10)
            context.assert_called_once_with()
            self.assertEqual(tls.wrap_socket.call_args.kwargs["server_hostname"], "public.example.test")
            unpinned.assert_not_called()

    def test_callback_url_credentials_rejected(self):
        self.assertIsNone(security.resolve_safe_callback_target(
            "https://user:secret@public.example.test/hook", localhost_mode=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)

