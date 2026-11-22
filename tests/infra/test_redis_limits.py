"""Contract tests for the Redis-backed atomic client/IP rate limits."""

import unittest
from unittest.mock import Mock, patch

from app.infra import redis_limits


class RedisLimitContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.redis = Mock()
        self.redis.eval.return_value = 0
        self.redis_factory = self.enterContext(
            patch.object(redis_limits.redis.Redis, "from_url", return_value=self.redis)
        )
        self.limiter = redis_limits.RedisLimiter("redis://localhost:6379/0")

    def test_redis_connection_has_finite_timeouts(self):
        self.redis_factory.assert_called_once_with(
            "redis://localhost:6379/0",
            decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
        )

    def test_assistant_checks_identity_and_ip_in_one_eval(self):
        self.redis.eval.return_value = 7
        self.assertEqual(self.limiter.check("session:user", "192.0.2.1", "assistant"), 7)
        self.redis.eval.assert_called_once()
        args = self.redis.eval.call_args.args
        self.assertEqual(args[1], 2)
        self.assertTrue(args[2].startswith("rl:assistant:"))
        self.assertTrue(args[3].startswith("rl:ip:assistant:"))
        self.assertEqual(args[4:], (10, 4))

    def test_read_and_media_use_one_atomic_eval_for_both_buckets(self):
        limits: tuple[tuple[redis_limits.RequestCategory, int], ...] = (("read", 30), ("media", 120))
        for category, limit in limits:
            with self.subTest(category=category):
                self.redis.eval.reset_mock()
                self.assertEqual(self.limiter.check("ip:192.0.2.1", "192.0.2.1", category), 0)
                self.redis.eval.assert_called_once()
                args = self.redis.eval.call_args.args
                self.assertEqual(args[1], 4)
                self.assertEqual(args[-3:], (limit, limit * 4, 10))
                self.assertTrue(args[2].startswith(f"rl:{category}:"))
                self.assertTrue(args[4].startswith(f"rl:ip:{category}:"))

    def test_failed_redis_script_is_not_silently_ignored(self):
        self.redis.eval.side_effect = redis_limits.redis.ConnectionError("connection lost")
        with self.assertRaises(redis_limits.redis.ConnectionError):
            self.limiter.check("user", "192.0.2.1", "assistant")

    def test_call_slots_use_atomic_lease_and_configured_capacity(self):
        self.redis.eval.return_value = 1
        with patch.object(redis_limits.time, "time", return_value=1000.0):
            self.assertTrue(self.limiter.acquire_call("call-token", maximum=3))

        script, *args = self.redis.eval.call_args.args
        self.assertIn("ZREMRANGEBYSCORE", script)
        self.assertEqual(args, [1, "deepseek:inflight", 1000.0, 1150.0, 3, "call-token"])
