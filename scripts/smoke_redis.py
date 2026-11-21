"""Integration smoke for atomic Redis rate-limits (run inside Compose)."""

import uuid

from app.infra.redis_limits import RedisLimiter


def expect(value: int, *, allowed: bool) -> None:
    if (value == 0) != allowed:
        raise AssertionError(f"Expected allowed={allowed}, Redis returned retry={value}")


def main() -> None:
    limiter = RedisLimiter()
    prefix = uuid.uuid4().hex
    ip = f"{prefix}:assistant"
    identities = [f"{prefix}:user:{i}" for i in range(5)]
    for identity in identities[:4]:
        expect(limiter.check(identity, ip, "assistant"), allowed=True)
    expect(limiter.check(identities[0], ip, "assistant"), allowed=False)
    expect(limiter.check(identities[4], ip, "assistant"), allowed=False)
    # The fifth identity must not lose its personal token when the shared IP is full.
    expect(limiter.check(identities[4], f"{prefix}:different-ip", "assistant"), allowed=True)

    read_ip = f"{prefix}:read"
    for user_index in range(4):
        identity = f"{prefix}:reader:{user_index}"
        for _ in range(30):
            expect(limiter.check(identity, read_ip, "read"), allowed=True)
    extra = f"{prefix}:reader:extra"
    expect(limiter.check(extra, read_ip, "read"), allowed=False)
    expect(limiter.check(extra, f"{prefix}:read-other-ip", "read"), allowed=True)
    print("Redis real Lua checks OK: per-identity, shared IP ceilings, no quota burn")


if __name__ == "__main__":
    main()
