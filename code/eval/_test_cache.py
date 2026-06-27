"""
cache 包单测（P1-B 验证）。用 FakeRedis 做 in-memory 后端，不依赖真实 Redis / Docker。
覆盖：L1 LRU、L2 精确、L3 语义余弦、L4 路由、embedding 缓存、归一化、降级、失效。
用法：python -m eval._test_cache
"""

import os
import sys
from types import SimpleNamespace

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from cache.cache_manager import CacheManager


class FakeRedis:
    """极简 in-memory Redis：dict + hashes，忽略 TTL。available 可控以测降级。"""
    def __init__(self, available=True):
        self._available = available
        self.kv = {}           # key -> value
        self.hashes = {}       # name -> {field: value}

    @property
    def available(self):
        return self._available

    def ping(self):
        return self._available

    def get(self, k):
        return self.kv.get(k)

    def setex(self, k, ttl, v):
        self.kv[k] = v
        return True

    def hset(self, name, key, value):
        self.hashes.setdefault(name, {})[key] = value
        return True

    def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def delete_keys(self, pattern):
        import fnmatch
        n = 0
        for store in (self.kv, self.hashes):
            for k in list(store.keys()):
                full = k if store is self.kv else k
                if fnmatch.fnmatch(full, pattern):
                    del store[k]; n += 1
        return n

    def close(self):
        pass


def make_cfg(**kw):
    base = dict(cache_enabled=True, cache_l1_size=3, cache_ttl_answer=3600,
                cache_ttl_route=21600, cache_ttl_embedding=86400,
                cache_semantic_threshold=0.92)
    base.update(kw)
    return SimpleNamespace(**base)


def embed(query):
    """确定性 embedding：按 query 里出现的菜名给 one-hot（控制相似度）。"""
    import numpy as np
    dishes = {"宫保鸡丁": 0, "鱼香肉丝": 1, "番茄炒蛋": 2}
    v = np.zeros(4, dtype="float32")
    for d, i in dishes.items():
        if d in query:
            v[i] = 1.0
            return v
    v[3] = 1.0  # 未知菜
    return v


_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        print(f"  ❌ {name}")


def main():
    print("=== 1. 归一化 & 降级（Redis 不可用）===")
    fr = FakeRedis(available=False)
    cm = CacheManager(fr, embed, make_cfg())
    check("归一化去空白", cm._normalize("宫保鸡丁 怎么做") == "宫保鸡丁怎么做")
    check("Redis 不可用时 get_answer 返回 None", cm.get_answer("x") is None)
    cm.set_answer("x", {"answer": "A"})   # no-op
    check("Redis 不可用时 set_answer 不抛错", True)
    check("Redis 不可用时 get_route 返回 None", cm.get_route("x") is None)
    check("redis_on=False", cm.redis_on is False)

    print("\n=== 2. L2 精确 + L1 LRU ===")
    fr = FakeRedis(available=True)
    cm = CacheManager(fr, embed, make_cfg(cache_l1_size=3))
    cm.set_answer("宫保鸡丁怎么做", {"answer": "A1", "strategy": "hybrid"})
    # 清 L1 强制走 L2
    cm._l1.clear()
    hit = cm.get_answer("宫保鸡丁怎么做")
    check("L2 命中（清 L1 后从 Redis 取回）", hit is not None and hit[1] == "L2" and hit[0]["answer"] == "A1")
    # L1 命中（上次 get_answer 已回填 L1）
    hit2 = cm.get_answer("宫保鸡丁怎么做")
    check("L1 命中", hit2 is not None and hit2[1] == "L1")
    # 归一化等价
    cm._l1.clear()
    hit3 = cm.get_answer("宫保鸡丁  怎么做")  # 多空格
    check("归一化等价命中同一 key", hit3 is not None and hit3[0]["answer"] == "A1")
    # LRU 淘汰：容量 3，插 4 条
    cm._l1.clear()
    for i in range(4):
        cm.set_answer(f"q{i}", {"answer": f"a{i}"})
    check("L1 容量上限淘汰到 3", len(cm._l1) == 3)
    check("最旧的 q0 已被淘汰", cm._sha("q0") not in cm._l1)

    print("\n=== 3. L3 语义缓存 ===")
    fr = FakeRedis(available=True)
    cm = CacheManager(fr, embed, make_cfg(cache_semantic_threshold=0.92))
    cm.register_semantic("宫保鸡丁怎么做", {"answer": "A宫保"})
    # 同菜不同问法 → 同 embedding → sim=1.0
    sem = cm.get_semantic_answer("宫保鸡丁如何做")
    check("同菜不同问法语义命中", sem is not None and sem[2] == "L3" and sem[1] >= 0.92)
    # 不同菜 → sim=0
    sem2 = cm.get_semantic_answer("鱼香肉丝怎么做")
    check("不同菜语义不命中", sem2 is None)
    # 语义索引已登记 1 条
    check("语义索引规模=1", len(cm._sem_index) == 1)

    print("\n=== 4. L4 路由缓存 ===")
    fr = FakeRedis(available=True)
    cm = CacheManager(fr, embed, make_cfg())
    check("未写入时 get_route=None", cm.get_route("x") is None)
    cm.set_route("宫保鸡丁怎么做", {"recommended_strategy": "hybrid_traditional", "confidence": 0.9})
    rt = cm.get_route("宫保鸡丁怎么做")
    check("路由命中", rt is not None and rt["recommended_strategy"] == "hybrid_traditional")

    print("\n=== 5. invalidate_all ===")
    fr = FakeRedis(available=True)
    cm = CacheManager(fr, embed, make_cfg())
    cm.set_answer("q", {"answer": "a"})
    cm.register_semantic("宫保鸡丁怎么做", {"answer": "x"})
    cm._ensure_sem_loaded()
    n = cm.invalidate_all()
    check("invalidate 清空命名空间键数>0", n > 0)
    check("invalidate 后 L1 空", len(cm._l1) == 0)
    check("invalidate 后 get_answer miss", cm.get_answer("q") is None)

    print("\n=== 6. stats ===")
    fr = FakeRedis(available=True)
    cm = CacheManager(fr, embed, make_cfg())
    cm.set_answer("q1", {"answer": "a"})
    cm._l1.clear()
    cm.get_answer("q1")  # L2 hit
    cm.get_answer("q2")  # miss
    s = cm.stats()
    check("stats 含 answer_hit_rate", "answer_hit_rate" in s)
    check("stats metrics 含 answer_hit_l2=1", s["metrics"].get("answer_hit_l2") == 1)

    print(f"\n{'='*40}\n通过 {_passed}，失败 {_failed}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
