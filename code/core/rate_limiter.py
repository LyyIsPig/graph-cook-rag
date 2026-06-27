"""
Redis 令牌桶限流（P4）。按 client(默认 IP) 限流，保护下游 LLM API 不被突发流量冲垮。
- 原子性：用 Lua 脚本在 Redis 内完成"补充令牌→判断→扣减"，避免并发竞态。
- 降级：Redis 不可用时回退到进程内令牌桶（带锁），限流仍生效（仅单进程内）。
令牌桶 vs 漏桶：令牌桶允许突发（桶满时一次消耗多个），更适合"人偶尔连点"的真实流量。
"""

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Redis Lua：原子地 补充令牌 + 判断 + 扣减。返回 1=放行 0=拒绝。
# KEYS[1]=桶key前缀  ARGV: capacity, refill(个/秒), now(秒)
_TOKEN_BUCKET_LUA = """
local prefix = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local tokens = tonumber(redis.call('get', prefix..':t') or capacity)
local last = tonumber(redis.call('get', prefix..':l') or now)
local delta = math.max(0, now - last)
tokens = math.min(capacity, tokens + delta * refill)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('set', prefix..':t', tokens)
redis.call('set', prefix..':l', now)
redis.call('expire', prefix..':t', 120)
redis.call('expire', prefix..':l', 120)
return allowed
"""


class TokenBucketLimiter:
    def __init__(self, redis_client, capacity: int = 20, refill: float = 5.0):
        self.r = redis_client
        self.capacity = capacity
        self.refill = refill
        self._lua = None
        # 进程内降级桶：client -> [tokens, last_ts]
        self._mem = {}
        self._lock = threading.Lock()
        if redis_client is not None and redis_client.available:
            try:
                # register_script 返回一个可调用对象；redis-py 8 兼容
                self._lua = redis_client.client.register_script(_TOKEN_BUCKET_LUA)
                logger.info(f"令牌桶限流就绪(Redis Lua): 容量={capacity} 补充={refill}/s")
            except Exception as e:
                logger.warning(f"注册限流 Lua 失败，降级进程内限流: {e}")
                self._lua = None

    def allow(self, client_id: str) -> bool:
        """消耗 1 个令牌；放行返回 True，超额返回 False。"""
        if self._lua is not None:
            try:
                key = f"grag:rl:{client_id}"
                return bool(self._lua(keys=[key], args=[self.capacity, self.refill, time.time()]))
            except Exception as e:
                logger.warning(f"Redis 限流失败，临时降级进程内: {e}")
        return self._mem_allow(client_id)

    def _mem_allow(self, client_id: str) -> bool:
        """进程内令牌桶降级（单进程有效，Redis 挂时兜底）。"""
        now = time.time()
        with self._lock:
            tokens, last = self._mem.get(client_id, [self.capacity, now])
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1:
                tokens -= 1
                self._mem[client_id] = [tokens, now]
                return True
            self._mem[client_id] = [tokens, now]
            return False


def client_ip_from(request) -> str:
    """取真实客户端 IP（优先代理转发头）。"""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return getattr(request.client, "host", "unknown") or "unknown"
