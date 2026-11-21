"""Resolve reverse-proxy identities without trusting forged X-Forwarded-For hops."""

import unittest
from unittest.mock import patch

from starlette.requests import Request

from app.middleware.client_identity import client_ip


def request(peer: str, forwarded: str) -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/health",
        "scheme": "http", "server": ("localhost", 80), "client": (peer, 2345),
        "headers": [(b"x-forwarded-for", forwarded.encode())],
        "query_string": b"",
    })


class TrustedProxyTests(unittest.TestCase):
    def test_untrusted_peer_cannot_spoof_ip(self):
        with patch.dict("os.environ", {"TRUSTED_PROXIES": "10.0.0.0/8"}):
            self.assertEqual(client_ip(request("192.0.2.5", "203.0.113.1")), "192.0.2.5")

    def test_nearest_untrusted_hop_wins_over_forged_prefix(self):
        with patch.dict("os.environ", {"TRUSTED_PROXIES": "10.0.0.0/8"}):
            self.assertEqual(
                client_ip(request("10.0.0.1", "198.51.100.99, 203.0.113.8")),
                "203.0.113.8",
            )

    def test_valid_chain_of_multiple_proxies(self):
        with patch.dict("os.environ", {"TRUSTED_PROXIES": "10.0.0.0/8"}):
            self.assertEqual(
                client_ip(request("10.0.0.1", "203.0.113.8, 10.0.0.2")),
                "203.0.113.8",
            )

    def test_malformed_chain_is_not_used(self):
        with patch.dict("os.environ", {"TRUSTED_PROXIES": "10.0.0.0/8"}):
            self.assertEqual(client_ip(request("10.0.0.1", "fake,invalid")), "10.0.0.1")
