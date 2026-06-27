"""
P1 真 Redis 烟测：用真实 Redis（docker）验证 RedisClient + CacheManager 的读写/语义/路由/失效。
不连 Neo4j/Milvus（embedding 用内存假函数），快且隔离。
用法（需 redis 容器在跑）：python -m eval._smoke_live_redis
"""
import os, sys, numpy as np
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path: sys.path.insert(0, _CODE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from dotenv import load_dotenv; load_dotenv(os.path.join(_CODE, ".env"))
from types import SimpleNamespace
from cache import RedisClient, CacheManager

def embed(q):
    v = np.zeros(8, dtype="float32")
    for i, k in enumerate(["宫保鸡丁", "鱼香肉丝", "番茄炒蛋", "怎么做", "有哪些"]):
        if k in q: v[i] = 1.0; return v
    v[7] = 1.0; return v

cfg = SimpleNamespace(cache_enabled=True, cache_l1_size=8, cache_ttl_answer=60,
                      cache_ttl_route=60, cache_ttl_embedding=60, cache_semantic_threshold=0.92)
from config import DEFAULT_CONFIG
rc = RedisClient(DEFAULT_CONFIG.redis_host, DEFAULT_CONFIG.redis_port,
                 DEFAULT_CONFIG.redis_db, DEFAULT_CONFIG.redis_password)
if not rc.available:
    print("❌ Redis 未连接，先 docker compose up redis"); sys.exit(1)
cm = CacheManager(rc, embed, cfg)
cm.invalidate_all()  # 干净起跑
ok = fail = 0
def chk(n, c):
    global ok, fail
    ok += bool(c); fail += not bool(c)
    print(("  ✅ " if c else "  ❌ ") + n)

print("=== 真 Redis 烟测 ===")
# 精确
cm.set_answer("宫保鸡丁怎么做", {"answer": "A1"})
cm._l1.clear()  # 强制 L2
chk("L2 精确命中", cm.get_answer("宫保鸡丁怎么做") is not None)
chk("归一化等价(多空格)", cm.get_answer("宫保鸡丁  怎么做") is not None)
# 路由
cm.set_route("宫保鸡丁怎么做", {"recommended_strategy": "hybrid_traditional"})
chk("L4 路由命中", cm.get_route("宫保鸡丁怎么做")["recommended_strategy"] == "hybrid_traditional")
# 语义
cm.register_semantic("宫保鸡丁怎么做", {"answer": "A宫保"})
sem = cm.get_semantic_answer("宫保鸡丁如何做")
chk("L3 语义命中(同菜不同问法)", sem is not None and sem[1] >= 0.92)
chk("L3 语义不命中(不同菜)", cm.get_semantic_answer("鱼香肉丝怎么做") is None)
# embedding 缓存
v1 = cm._embed_cached("宫保鸡丁怎么做")
v2 = cm._embed_cached("宫保鸡丁怎么做")  # 应走 emb 缓存
chk("embedding 缓存返回一致向量", v1 is not None and np.allclose(v1, v2))
# 失效
n = cm.invalidate_all()
chk("invalidate 清除键数>0", n > 0)
chk("invalidate 后精确 miss", cm.get_answer("宫保鸡丁怎么做") is None)
# stats
print("\nstats:", cm.stats())
print(f"\n通过 {ok}，失败 {fail}")
sys.exit(1 if fail else 0)
