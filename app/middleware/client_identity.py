import hashlib
import hmac
import ipaddress
import os

from fastapi import Request


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    trusted = os.getenv("TRUSTED_PROXIES", "").split(",")
    try:
        allowed = any(
            ipaddress.ip_address(peer) in ipaddress.ip_network(value.strip(), strict=False)
            for value in trusted
            if value.strip()
        )
    except ValueError:
        allowed = False
    if allowed:
        candidate = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            pass
    return peer


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
