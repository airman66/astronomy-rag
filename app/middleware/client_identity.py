import hashlib
import hmac
import ipaddress
import os

from fastapi import Request


def client_ip(request: Request) -> str:
    """Use the nearest untrusted hop; do not trust an attacker-supplied XFF prefix."""
    peer = request.client.host if request.client else "unknown"
    try:
        peer_ip = ipaddress.ip_address(peer)
        networks = [
            ipaddress.ip_network(value.strip(), strict=False)
            for value in os.getenv("TRUSTED_PROXIES", "").split(",") if value.strip()
        ]
    except ValueError:
        return peer
    if not any(peer_ip in network for network in networks):
        return str(peer_ip)
    forwarded = request.headers.get("x-forwarded-for", "")
    if not forwarded:
        return str(peer_ip)
    for hop in reversed(forwarded.split(",")):
        try:
            candidate = ipaddress.ip_address(hop.strip())
        except ValueError:
            return str(peer_ip)  # malformed forwarded chain: fail closed
        if not any(candidate in network for network in networks):
            return str(candidate)
    return str(peer_ip)


def client_identity(request: Request) -> tuple[str, str]:
    ip = client_ip(request)
    session = request.cookies.get("glossary_session", "")
    secret = os.getenv("SESSION_SECRET", "")
    if secret and "." in session:
        identifier, signature = session.rsplit(".", 1)
        expected = hmac.new(secret.encode(), identifier.encode(), hashlib.sha256).hexdigest()
        if identifier and len(identifier) <= 128 and hmac.compare_digest(signature, expected):
            return f"session:{identifier}", ip
    return f"ip:{ip}", ip
