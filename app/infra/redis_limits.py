"""Atomic, shared Redis windows and cooldowns."""

from __future__ import annotations

import hashlib
import time
from typing import Literal, Protocol

import redis

from app.infra.config import CALL_LEASE_SECONDS, redis_url

RequestCategory = Literal["assistant", "read", "media"]


class RateLimiter(Protocol):
    def check(self, identity: str, ip: str, category: RequestCategory) -> int: ...
    def acquire_call(self, token: str) -> bool: ...
    def release_call(self, token: str) -> None: ...

# Each rate-limit decision updates the client and shared-IP buckets atomically.
# Rejected requests must not consume quota from a different bucket.
_WINDOW_BOTH = """
local client_cooldown = redis.call('TTL', KEYS[2])
if client_cooldown > 0 then return client_cooldown end
local ip_cooldown = redis.call('TTL', KEYS[4])
if ip_cooldown > 0 then return ip_cooldown end
if tonumber(redis.call('GET', KEYS[1]) or '0') >= tonumber(ARGV[1]) then
  redis.call('SET', KEYS[2], '1', 'EX', ARGV[3], 'NX')
  return redis.call('TTL', KEYS[2])
end
if tonumber(redis.call('GET', KEYS[3]) or '0') >= tonumber(ARGV[2]) then
  redis.call('SET', KEYS[4], '1', 'EX', ARGV[3], 'NX')
  return redis.call('TTL', KEYS[4])
end
if redis.call('INCR', KEYS[1]) == 1 then redis.call('EXPIRE', KEYS[1], ARGV[3]) end
if redis.call('INCR', KEYS[3]) == 1 then redis.call('EXPIRE', KEYS[3], ARGV[3]) end
return 0
"""

# Assistant has a per-identity 10-second cooldown and a shared-IP ceiling.
# Check both before changing either key, so IP rejection doesn't burn client quota.
_ASSISTANT_BOTH = """
local ttl = redis.call('TTL', KEYS[1])
if ttl > 0 then return ttl end
if tonumber(redis.call('GET', KEYS[2]) or '0') >= tonumber(ARGV[2]) then
  local ip_ttl = redis.call('TTL', KEYS[2])
  if ip_ttl < 1 then ip_ttl = 1 end
  return ip_ttl
end
redis.call('SET', KEYS[1], '1', 'EX', ARGV[1])
if redis.call('INCR', KEYS[2]) == 1 then
  redis.call('EXPIRE', KEYS[2], ARGV[1])
end
return 0
"""


# Cleanup and capacity checks must run atomically to avoid overbooking slots.
_ACQUIRE_CALL = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then return 0 end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[4])
redis.call('EXPIRE', KEYS[1], 300)
return 1
"""


class RedisLimiter:
    def __init__(self, url: str | None = None) -> None:
        # Bound network failures so requests cannot hang indefinitely on Redis.
        self.redis = redis.Redis.from_url(
            url or redis_url(), decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
        )

    def check(self, identity: str, ip: str, category: RequestCategory) -> int:
        limit = {"assistant": 1, "read": 30, "media": 120}[category]
        key = hashlib.sha256(identity.encode()).hexdigest()
        ip_key = hashlib.sha256(ip.encode()).hexdigest()
        if category == "assistant":
            return int(self.redis.eval(
                _ASSISTANT_BOTH, 2, f"rl:assistant:{key}", f"rl:ip:assistant:{ip_key}",
                10, limit * 4,
            ))
        return int(self.redis.eval(
            _WINDOW_BOTH, 4,
            f"rl:{category}:{key}", f"rl:cool:{category}:{key}",
            f"rl:ip:{category}:{ip_key}", f"rl:cool:ip:{category}:{ip_key}",
            limit, limit * 4, 10,
        ))

    def acquire_call(self, token: str, maximum: int = 4) -> bool:
        now = time.time()
        return bool(self.redis.eval(
            _ACQUIRE_CALL, 1, "deepseek:inflight", now, now + CALL_LEASE_SECONDS, maximum, token,
        ))

    def release_call(self, token: str) -> None:
        self.redis.zrem("deepseek:inflight", token)
