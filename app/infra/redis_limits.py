"""Atomic, shared Redis windows and cooldowns."""

from __future__ import annotations

import hashlib
import time

import redis

from app.infra.config import CALL_LEASE_SECONDS, redis_url

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

class RedisLimiter:
    def __init__(self, url: str | None = None) -> None:
        self.redis = redis.Redis.from_url(url or redis_url(), decode_responses=True)

    def check(self, identity: str, ip: str, category: str) -> int:
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
        now = time.time()
        return bool(self.redis.eval(script, 1, "deepseek:inflight", now, now + CALL_LEASE_SECONDS, maximum, token))

    def release_call(self, token: str) -> None:
        self.redis.zrem("deepseek:inflight", token)
