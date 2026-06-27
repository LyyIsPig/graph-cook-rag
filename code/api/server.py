"""
FastAPI 服务层 - 把 AdvancedGraphRAGSystem 暴露为 HTTP API
P0 阶段：RESTful 接口 + SSE 流式 + 结构化日志 + request_id 全链路追踪

启动（在 code/ 目录下）：
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
或：
    python -m api.server

接口：
    GET  /health            健康检查 + 系统状态（配置脱敏）
    POST /api/ask           非流式问答，返回答案 + 路由 + 检索溯源
    POST /api/ask/stream    SSE 流式问答：先推 metadata 帧，再逐 token 推，最后 done
"""

import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

# Windows 控制台默认 GBK，遇 emoji(✅/❓) 会 UnicodeEncodeError；强制 UTF-8 避免崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 让本文件无论从哪启动都能找到 code/ 下的包（config / main / rag_modules）
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE_DIR, ".env"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.logging_config import setup_logging, RequestContext, get_request_id, request_id_var
from core.metrics import HTTP_REQUESTS, HTTP_LATENCY, metrics_body
from core.rate_limiter import TokenBucketLimiter, client_ip_from
from core.stats import RequestStats
from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem, REFUSAL_MSG

logger = setup_logging()

# 单例系统实例（在 lifespan 中初始化一次，避免每个请求重建知识库）
_system: Optional[AdvancedGraphRAGSystem] = None
_limiter: Optional[TokenBucketLimiter] = None
_stats: Optional[RequestStats] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时构建系统，关闭时释放资源。"""
    global _system, _limiter, _stats
    with RequestContext("startup"):
        logger.info("Graph RAG 系统启动中（首次较慢：连接 Neo4j/Milvus + 构建知识库）...")
        _system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
        _system.initialize_system()
        _system.build_knowledge_base()
        # P4 限流器（复用系统的 Redis 连接）
        if DEFAULT_CONFIG.rate_limit_enabled:
            _limiter = TokenBucketLimiter(
                _system.cache.r if _system.cache else None,
                capacity=DEFAULT_CONFIG.rate_limit_capacity,
                refill=DEFAULT_CONFIG.rate_limit_refill,
            )
        # P3/P5 轻量请求统计（监控条 / /api/stats）
        _stats = RequestStats()
        logger.info("Graph RAG 系统就绪")
        yield
        if _system is not None:
            _system._cleanup()
            logger.info("Graph RAG 系统已关闭")


app = FastAPI(title="Graph RAG 烹饪助手 API", version="0.1.0", lifespan=lifespan)

# 允许前端（P3 的 Gradio/React）跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    为每个请求注入 request_id（写进请求级异步上下文）。
    - StreamingResponse 的同步生成器每次 next() 在不同 worker 线程跑，会复制本上下文 → 能继承到 rid；
    - 请求结束后该异步上下文销毁，不会泄漏到下一个请求，因此无需 reset（reset 反而会跨 Context 报错）。
    """
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """P4 令牌桶限流：对 /api/* 按客户端 IP 限流，超额返回 429。健康/指标端点不限。"""
    path = request.url.path
    if (_limiter is not None) and path.startswith("/api/"):
        ip = client_ip_from(request)
        if not _limiter.allow(ip):
            logger.info(f"rate limited ip={ip} path={path}")
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试。", "retry_after": "1s"},
            )
    return await call_next(request)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """P5 HTTP 指标：记录每个请求的计数 + 延迟（/metrics 自身不计，避免自激）。"""
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    t0 = time.time()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        HTTP_REQUESTS.labels(endpoint=path, method=request.method, status="500").inc()
        raise
    elapsed = time.time() - t0
    HTTP_REQUESTS.labels(endpoint=path, method=request.method, status=str(status)).inc()
    HTTP_LATENCY.labels(endpoint=path).observe(elapsed)
    if _stats is not None and path.startswith("/api/ask"):
        _stats.record_request(elapsed * 1000, status)
    return response


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    top_k: Optional[int] = Field(None, ge=1, le=20)
    stream: bool = False
    explain_routing: bool = False


def _need_system() -> AdvancedGraphRAGSystem:
    if _system is None or not _system.system_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪，请稍后再试")
    return _system


def _strategy_value(analysis: Any) -> Optional[str]:
    if analysis is None:
        return None
    strat = getattr(analysis, "recommended_strategy", None)
    return getattr(strat, "value", str(strat)) if strat is not None else None


def _doc_summary(doc: Any) -> dict:
    md = getattr(doc, "metadata", {}) or {}
    return {
        "recipe_name": md.get("recipe_name") or md.get("name") or "未知菜品",
        "search_type": md.get("search_type") or md.get("search_method") or md.get("route_strategy"),
        "score": md.get("final_score", md.get("relevance_score", md.get("score", 0.0))),
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    """健康检查 + 系统状态（配置脱敏，不泄露 API Key）。"""
    return {
        "status": "ok" if (_system and _system.system_ready) else "loading",
        "system_ready": bool(_system and _system.system_ready),
        "config": DEFAULT_CONFIG.to_dict(),
        "cache": _system.cache_stats() if _system else None,
        "rate_limit_enabled": _limiter is not None,
        "request_id": get_request_id(),
    }


@app.get("/metrics")
def metrics():
    """P5 Prometheus 指标端点（供 Prometheus 抓取 / curl 裸文本查看）。"""
    body, ctype = metrics_body()
    return Response(content=body, media_type=ctype)


@app.get("/api/stats")
def api_stats():
    """P3 监控条 / P5 指标的 JSON 源：QPS、p50/p95/p99、缓存命中分层、路由分布、拒答。"""
    return _stats.snapshot() if _stats else {}


class GraphReq(BaseModel):
    recipes: list[str] = Field(default_factory=list)


@app.post("/api/graph")
def api_graph(req: GraphReq):
    """P3 图谱面板：给一组菜谱名，返回 1 跳(食材+分类)的 nodes/edges。"""
    system = _need_system()
    return system.graph_rag_retrieval.provenance_graph(req.recipes)


@app.get("/api/cache/stats")
def cache_stats():
    """P1 缓存命中率与各层埋点（喂给 P5 监控）。"""
    system = _need_system()
    return system.cache_stats()


@app.post("/api/cache/invalidate")
def cache_invalidate():
    """手动清空缓存命名空间（知识库 rebuild 后调用）。"""
    system = _need_system()
    cleared = system.invalidate_cache()
    logger.info(f"cache invalidated, cleared={cleared}")
    return {"cleared": cleared}


@app.post("/api/ask")
async def ask(req: AskRequest):
    """非流式问答（P4 异步）：一次返回答案 + 路由 + 溯源 + 缓存命中层。
    全程 async：闸门/检索/生成均让出事件循环，多请求并发。"""
    rid = get_request_id()
    logger.info(f"ask question={req.question!r}")
    system = _need_system()
    try:
        result = await system.ask_async(req.question, req.top_k or system.config.top_k)
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {e}")
    resp = {"request_id": rid, **result}
    if _stats is not None:
        _stats.record_route(resp.get("strategy"))
        _stats.record_cache(resp.get("cache_hit"))
        if resp.get("refused"):
            _stats.record_refusal(resp.get("reason"))
    logger.info(f"ask done strategy={resp.get('strategy')} cache_hit={resp.get('cache_hit')} "
                f"latency_ms={resp.get('latency_ms')}")
    return resp


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE 流式问答：先推 metadata 帧（检索溯源），再逐 token 推生成内容，最后推 done。
    生成器为同步实现，StreamingResponse 会用线程池迭代，不阻塞事件循环。"""
    system = _need_system()
    top_k = req.top_k or system.config.top_k

    def event_stream():
        rid = get_request_id()
        logger.info(f"stream question={req.question!r}")
        # P1 精确缓存(L1/L2)：命中前置，把缓存答案一次性吐出（不再检索/生成）
        if system.cache is not None:
            cached = system.cache.get_answer(req.question)
            if cached is not None:
                payload, layer = cached
                yield _sse({"type": "metadata", "request_id": rid, "cache_hit": layer,
                            "strategy": payload.get("strategy"),
                            "sources": payload.get("sources", [])})
                yield _sse({"type": "token", "content": payload.get("answer", "")})
                yield _sse({"type": "done", "cache_hit": layer})
                return

        # 拒答闸门（P2-3）：明显无关/不存在 → 推一帧拒答后结束
        answerable, reason, conf = system.check_answerable(req.question)
        if not answerable:
            yield _sse({"type": "metadata", "request_id": rid, "refused": True,
                        "reason": reason, "sources": []})
            yield _sse({"type": "done", "refused": True})
            return

        # P1 语义缓存(L3)：闸门通过后，复用相似历史 query
        if system.cache is not None:
            sem = system.cache.get_semantic_answer(req.question)
            if sem:
                payload, layer = sem[0], f"L3({sem[1]:.2f})"
                yield _sse({"type": "metadata", "request_id": rid, "cache_hit": layer,
                            "strategy": payload.get("strategy"),
                            "sources": payload.get("sources", [])})
                yield _sse({"type": "token", "content": payload.get("answer", "")})
                yield _sse({"type": "done", "cache_hit": layer})
                return

        t0 = time.time()

        # 1) 路由 + 检索（阻塞），先把检索溯源推给前端
        try:
            relevant_docs, analysis = system.query_router.route_query(req.question, top_k)
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            yield _sse({"type": "error", "message": f"检索失败: {e}"})
            return

        yield _sse({
            "type": "metadata",
            "request_id": rid,
            "strategy": _strategy_value(analysis),
            "query_complexity": getattr(analysis, "query_complexity", None),
            "relationship_intensity": getattr(analysis, "relationship_intensity", None),
            "sources": [_doc_summary(d) for d in relevant_docs],
        })

        # 2) 流式生成
        full_text = ""
        try:
            for chunk in system.generation_module.generate_adaptive_answer_stream(
                req.question, relevant_docs
            ):
                full_text += chunk
                yield _sse({"type": "token", "content": chunk})
        except Exception as e:
            logger.error(f"生成失败: {e}", exc_info=True)
            yield _sse({"type": "error", "message": f"生成失败: {e}"})
            return

        latency_ms = round((time.time() - t0) * 1000, 1)
        # P1 落库（累积完整文本 + 溯源，供后续精确/语义命中）
        if system.cache is not None:
            _payload = {"answer": full_text, "strategy": _strategy_value(analysis),
                        "sources": [_doc_summary(d) for d in relevant_docs], "latency_ms": latency_ms}
            system.cache.set_answer(req.question, _payload)
            system.cache.register_semantic(req.question, _payload)
        logger.info(f"stream done latency_ms={latency_ms}")
        yield _sse({"type": "done", "latency_ms": latency_ms})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证流式实时
        },
    )


# ---------- P3 前端静态托管（工坊 SPA：3 栏 X 光机）----------
_STATIC_DIR = os.path.join(_CODE_DIR, "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def _index():
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
