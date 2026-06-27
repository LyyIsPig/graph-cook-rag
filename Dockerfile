# Graph RAG 烹饪助手 · 应用镜像
# python:3.12-slim + 依赖 + BGE 嵌入模型烤进镜像（运行时离线）
FROM python:3.12-slim

# 编译依赖：pymilvus / numpy / scipy 等部分 wheel 需要 gcc/g++
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（独立层，利用 docker 构建缓存）
# torch 用 CPU 专用索引：默认 PyPI 的 torch==2.6.0 是 CUDA 版（带 ~3.5GB 的 nvidia 显卡库），
# 本项目只用 CPU 跑 BGE，装 CPU 版（~200MB）省一个数量级。
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 \
        --index-url https://download.pytorch.org/whl/cpu \
        --trusted-host download.pytorch.org \
        torch==2.6.0
# 其余依赖用清华 PyPI 镜像（国内 CDN 快且稳，避免 files.pythonhosted.org 超时）
RUN pip install --no-cache-dir --timeout 120 \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        -r requirements.txt

# 把 BGE 嵌入模型烤进镜像：构建时用国内镜像源下载一次，运行时 HF_HUB_OFFLINE=1 直接用缓存
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-zh-v1.5')"

# 拷贝应用代码
COPY code/ ./code/

# 运行时：离线（模型已烤进）、不缓冲（日志实时）
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/code
EXPOSE 8000

# 健康检查：/health 返回 200 即健康
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
