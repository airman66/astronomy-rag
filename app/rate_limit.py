"""Atomic, shared Redis windows and cooldowns."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os

import redis
from fastapi import Request

_WINDOW = """
local cooldown = redis.call('TTL', KEYS[2])
if cooldown > 0 then return cooldown end
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], 10) end
if count > tonumber(ARGV[1]) then
  redis.call('SET', KEYS[2], '1', 'EX', 10, 'NX')
  return redis.call('TTL', KEYS[2])
end
return 0
"""

# Assistant: one accepted request per 10 s. An early retry gets the remaining time of that
# interval and does not start a new penalty (SPEC: Retry-After = остаток ожидания).
_ONCE = """
if redis.call('SET', KEYS[1], '1', 'EX', ARGV[1], 'NX') then return 0 end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 1 then ttl = 1 end
return ttl
"""

# Shared per-IP ceiling for the assistant: a fixed 10 s window without extra cooldown.
_CEILING = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end
if count > tonumber(ARGV[1]) then
  local ttl = redis.call('TTL', KEYS[1])
  if ttl < 1 then ttl = 1 end
  return ttl
end
return 0
"""

# Lease of a DeepSeek slot; must outlive the assistant deadline (main.ASSISTANT_DEADLINE).
CALL_LEASE_SECONDS = 150


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    trusted = os.getenv("TRUSTED_PROXIES", "").split(",")
    try:
        allowed = any(
            ipaddress.ip_address(peer) in ipaddress.ip_network(value.strip(), strict=False)
            for value in trusted if value.strip()
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


class RedisLimiter:
    def __init__(self, url: str | None = None) -> None:
        self.redis = redis.Redis.from_url(url or os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)

    def check(self, request: Request, category: str) -> int:
        identity, ip = client_identity(request)
        limit = {"assistant": 1, "read": 30, "media": 120}[category]
        key = hashlib.sha256(identity.encode()).hexdigest()
        ip_key = hashlib.sha256(ip.encode()).hexdigest()
        if category == "assistant":
            retry = int(self.redis.eval(_ONCE, 1, f"rl:assistant:{key}", 10))
            if retry:
                return retry
            return int(self.redis.eval(_CEILING, 1, f"rl:ip:assistant:{ip_key}", limit * 4, 10))
        retry = int(self.redis.eval(_WINDOW, 2, f"rl:{category}:{key}", f"rl:cool:{category}:{key}", limit))
        if retry:
            return retry
        # A distinct session cannot evade this shared IP ceiling.
        return int(self.redis.eval(_WINDOW, 2, f"rl:ip:{category}:{ip_key}", f"rl:cool:ip:{category}:{ip_key}", limit * 4))

    def acquire_call(self, token: str, maximum: int = 4) -> bool:
        script = """
        redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
        if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then return 0 end
        redis.call('ZADD', KEYS[1], ARGV[2], ARGV[4])
        redis.call('EXPIRE', KEYS[1], 300)
        return 1
        """
        import time

        now = time.time()
        return bool(self.redis.eval(script, 1, "deepseek:inflight", now, now + CALL_LEASE_SECONDS, maximum, token))

    def release_call(self, token: str) -> None:
        self.redis.zrem("deepseek:inflight", token)
