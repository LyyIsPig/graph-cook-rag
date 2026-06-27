"""
P4 压测脚本：asyncio + httpx 并发压 /api/ask，报 QPS / p50 / p95 / p99 / 错误率 / 缓存命中分布。
两种场景对比：
  - repeat：小查询池反复问（缓存友好，应高命中、高 QPS）
  - unique ：大查询池少重复（缓存不友好，真压检索+生成链路）

前置：服务已起（uvicorn api.server:app --port 8000）。
用法：
    python -m eval.load_test --users 10 --total 60 --scenario repeat
    python -m eval.load_test --users 10 --total 60 --scenario unique
"""

import os
import sys
import time
import asyncio
import argparse
from collections import Counter

import httpx

# 重复池（缓存友好）
REPEAT_POOL = [
    "番茄炒蛋怎么做", "用鸡蛋做的菜有哪些", "和可乐鸡翅一样用了鸡翅中的菜还有哪些？",
    "用了青辣椒的素菜有哪些", "需要砂锅的菜有哪些", "宫保鸡丁怎么做",
    "红烧肉怎么做", "用花生做的菜有哪些",
]
# 唯一池（缓存不友好；尽量不重复）
UNIQUE_POOL = [
    "糖醋排骨怎么做", "麻婆豆腐怎么做", "鱼香肉丝怎么做", "回锅肉怎么做", "水煮鱼怎么做",
    "宫保鸡丁怎么做", "可乐鸡翅怎么做", "红烧排骨怎么做", "清蒸鲈鱼怎么做", "蒜蓉粉丝蒸虾怎么做",
    "用土豆做的菜有哪些", "用牛肉做的菜有哪些", "用豆腐做的菜有哪些", "用虾仁做的菜有哪些", "用猪肉做的菜有哪些",
    "用了蒜苔的荤菜有哪些", "用了老抽的水产有哪些", "用了鸭肉的荤菜有哪些", "用了香油的主食有哪些", "用了香醋的素菜有哪些",
    "和手抓饼一样用了火腿的菜还有哪些？", "和凉皮一样用了芝麻酱的菜还有哪些？",
    "需要高压锅的菜有哪些", "需要电饭煲的菜有哪些", "需要微波炉的菜有哪些", "需要烤箱的菜有哪些", "需要空气炸锅的菜有哪些",
    "用到炸这种做法的菜有哪些", "用到蒸这种做法的菜有哪些", "用到焖这种做法的菜有哪些",
    "番茄炒蛋和蛋汤哪个难度大", "红烧肉和红烧排骨哪个用的食材更多",
    "西红柿炒鸡蛋怎么做", "蛋炒饭怎么做", "酸辣土豆丝怎么做", "干煸豆角怎么做", "地三鲜怎么做",
    "有哪些川菜", "有哪些粤菜", "有哪些汤类", "有哪些主食", "有哪些甜品",
    "青椒肉丝怎么做", "土豆炖牛肉怎么做", "葱花鸡蛋饼怎么做", "紫菜蛋花汤怎么做", "醋溜白菜怎么做",
]


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


async def run_scenario(url, users, total, scenario):
    pool = REPEAT_POOL if scenario == "repeat" else UNIQUE_POOL
    sem = asyncio.Semaphore(users)
    results = []
    # total 控制总请求数；同时给个时间兜底
    stop_at = time.time() + 600
    async with httpx.AsyncClient() as client:
        # 先确认服务在
        try:
            h = await client.get(url.replace("/api/ask", "/health"), timeout=10)
            if h.status_code != 200:
                print(f"服务未就绪: /health={h.status_code}"); return
        except Exception as e:
            print(f"连不上服务 {url}: {e}"); return
        # 先清缓存，确保冷起跑
        try:
            await client.post(url.replace("/api/ask", "/api/cache/invalidate"), timeout=10)
        except Exception:
            pass

        print(f"\n=== 场景 {scenario}：{users} 并发 × 目标 {total} 请求 ===")
        t0 = time.time()
        # 派 workers；通过共享 results 长度停（达 total 即停）
        tasks = []
        for i in range(users):
            tasks.append(asyncio.create_task(worker_loop(client, url, pool, i, sem, results, total, stop_at)))
        # 监控达 total 提前停
        while len(results) < total and time.time() < stop_at:
            await asyncio.sleep(0.5)
        # 取消还在跑的 worker
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - t0

    report(results, elapsed, scenario)


async def worker_loop(client, url, pool, wid, sem, results, total, stop_at):
    i = wid
    while time.time() < stop_at and len(results) < total:
        async with sem:
            if len(results) >= total:
                break
            q = pool[i % len(pool)]
            i += 1
            t0 = time.time()
            try:
                r = await client.post(url, json={"question": q}, timeout=120)
                ms = (time.time() - t0) * 1000
                try:
                    body = r.json()
                except Exception:
                    body = {}
                results.append({"ms": ms, "status": r.status_code, "cache_hit": body.get("cache_hit")})
            except Exception as e:
                results.append({"ms": (time.time() - t0) * 1000, "status": 0, "error": str(e)[:50]})


def report(results, elapsed, scenario):
    if not results:
        print("无结果"); return
    lat = [r["ms"] for r in results]
    statuses = Counter(r["status"] for r in results)
    hits = Counter()
    for r in results:
        ch = r.get("cache_hit")
        hits["HIT" if ch else "MISS"] += 1
    err = sum(1 for r in results if r["status"] != 200)
    total = len(results)
    qps = total / elapsed if elapsed > 0 else 0
    print("-" * 64)
    print(f"请求总数 : {total}    耗时: {elapsed:.1f}s    QPS: {qps:.2f}")
    print(f"成功/错误: {statuses.get(200,0)} / {err}  (状态分布 {dict(statuses)})")
    print(f"延迟(ms) : p50={percentile(lat,50):.0f}  p95={percentile(lat,95):.0f}  p99={percentile(lat,99):.0f}  max={max(lat):.0f}")
    print(f"缓存命中 : {hits}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/api/ask")
    ap.add_argument("--users", type=int, default=10, help="并发用户数")
    ap.add_argument("--total", type=int, default=60, help="总请求数")
    ap.add_argument("--scenario", default="repeat", choices=["repeat", "unique"])
    args = ap.parse_args()
    asyncio.run(run_scenario(args.url, args.users, args.total, args.scenario))


if __name__ == "__main__":
    main()
