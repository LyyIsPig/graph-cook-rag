"""
Redis 连接客户端（P1）。
设计要点：
- **优雅降级**：Redis 不可达时 available=False，所有读写 no-op（get 返回 None、set/hset/delete 跳过），
  上层 CacheManager 据此把缓存当 miss 处理，系统行为与"无缓存"完全一致。
- **运行时容错**：即便启动时连上、运行中 Redis 掉线，每次操作仍 try/except 兜底，不抛到业务层。
- 线程安全：redis-py 的 Redis 客户端自带连接池，线程安全。
"""

import logging
from typing import Optional

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisClient:
    """带健康检查与降级的 Redis 包装。decode_responses=True：所有返回为 str（缓存存 JSON）。"""

    def __init__(self, host: str, port: int, db: int = 0, password: str = "",
                 socket_timeout: float = 2.0, socket_connect_timeout: float = 2.0):
        self.host, self.port, self.db = host, port, db
        self.client = None
        self._available = False
        try:
            self.client = redis.Redis(
                host=host, port=port, db=db,
                password=(password or None),
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True,
            )
            self.client.ping()
            self._available = True
            logger.info(f"Redis 连接成功: {host}:{port}/{db}")
        except Exception as e:
            # 启动期不可达不致命：缓存降级为 no-op
            self.client = None
            self._available = False
            logger.warning(f"Redis 不可达，缓存降级为 no-op（系统仍可正常问答）: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def ping(self) -> bool:
        """主动探活；失败则把自身标记为不可用。"""
        if self.client is None:
            return False
        try:
            self.client.ping()
            self._available = True
            return True
        except RedisError:
            self._available = False
            return False

    def get(self, key: str) -> Optional[str]:
        if not self._available or self.client is None:
            return None
        try:
            return self.client.get(key)
        except RedisError as e:
            logger.warning(f"Redis get 失败，降级 miss: {e}")
            return None

    def setex(self, key: str, ttl: int, value: str) -> bool:
        if not self._available or self.client is None:
            return False
        try:
            self.client.setex(key, ttl, value)
            return True
        except RedisError as e:
            logger.warning(f"Redis setex 失败，跳过写入: {e}")
            return False

    def hset(self, name: str, key: str, value: str) -> bool:
        if not self._available or self.client is None:
            return False
        try:
            self.client.hset(name, key, value)
            return True
        except RedisError as e:
            logger.warning(f"Redis hset 失败，跳过: {e}")
            return False

    def hget(self, name: str, key: str) -> Optional[str]:
        if not self._available or self.client is None:
            return None
        try:
            return self.client.hget(name, key)
        except RedisError as e:
            logger.warning(f"Redis hget 失败，降级 None: {e}")
            return None

    def hgetall(self, name: str) -> dict:
        if not self._available or self.client is None:
            return {}
        try:
            return self.client.hgetall(name)
        except RedisError as e:
            logger.warning(f"Redis hgetall 失败，降级空: {e}")
            return {}

    def delete_keys(self, pattern: str) -> int:
        """按 glob 模式批量删除（SCAN，生产安全，不用 KEYS）。返回删除条数。"""
        if not self._available or self.client is None:
            return 0
        try:
            n = 0
            pipe = self.client.pipeline()
            batch = []
            for k in self.client.scan_iter(match=pattern, count=500):
                batch.append(k)
                if len(batch) >= 500:
                    pipe.delete(*batch); n += len(batch); batch = []
            if batch:
                pipe.delete(*batch); n += len(batch)
            pipe.execute()
            return n
        except RedisError as e:
            logger.warning(f"Redis delete_keys 失败: {e}")
            return 0

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
