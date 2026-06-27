"""
基于图数据库的RAG系统配置文件
（P0 工程化重构：配置统一收口，LLM 提供商默认智谱 GLM，从环境变量加载）
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "glm-4-flash"
    # 统一 LLM 提供商：智谱 GLM（from_env 从环境变量读取密钥与 endpoint）
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # 检索配置（LightRAG Round-robin策略）
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度

    # Redis 多级缓存配置（P1）
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_enabled: bool = True            # 总开关；False 则完全旁路缓存
    cache_l1_size: int = 512              # L1 进程内 LRU 容量（条）
    cache_ttl_answer: int = 3600          # L1/L2 答案缓存 TTL（秒，1h）
    cache_ttl_route: int = 21600          # L4 路由决策 TTL（秒，6h）
    cache_ttl_embedding: int = 86400      # embedding 缓存 TTL（秒，24h）
    cache_semantic_threshold: float = 0.92  # L3 语义缓存余弦阈值（越高越严，误命中越少）

    # 限流配置（P4，Redis 令牌桶，按 IP）
    rate_limit_enabled: bool = True
    rate_limit_capacity: int = 20       # 桶容量（突发上限，单 IP 最多累积令牌数）
    rate_limit_refill: float = 5.0      # 令牌补充速率（个/秒，即单 IP 持续 QPS 上限）

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass

    @classmethod
    def from_env(cls) -> "GraphRAGConfig":
        """
        从环境变量构建配置（P0 配置收口入口）。
        密钥读取顺序：LLM_API_KEY → ZHIPU_API_KEY → MOONSHOT_API_KEY（兼容历史 .env）
        """
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", cls.neo4j_uri),
            neo4j_user=os.getenv("NEO4J_USER", cls.neo4j_user),
            neo4j_password=os.getenv("NEO4J_PASSWORD", cls.neo4j_password),
            neo4j_database=os.getenv("NEO4J_DATABASE", cls.neo4j_database),
            milvus_host=os.getenv("MILVUS_HOST", cls.milvus_host),
            milvus_port=int(os.getenv("MILVUS_PORT", cls.milvus_port)),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_api_key=(
                os.getenv("LLM_API_KEY")
                or os.getenv("ZHIPU_API_KEY")
                or os.getenv("MOONSHOT_API_KEY", "")
            ),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
            redis_host=os.getenv("REDIS_HOST", cls.redis_host),
            redis_port=int(os.getenv("REDIS_PORT", cls.redis_port)),
            redis_db=int(os.getenv("REDIS_DB", cls.redis_db)),
            redis_password=os.getenv("REDIS_PASSWORD", cls.redis_password),
            cache_enabled=os.getenv("CACHE_ENABLED", "1").lower() not in ("0", "false", "no"),
            cache_l1_size=int(os.getenv("CACHE_L1_SIZE", cls.cache_l1_size)),
            cache_ttl_answer=int(os.getenv("CACHE_TTL_ANSWER", cls.cache_ttl_answer)),
            cache_ttl_route=int(os.getenv("CACHE_TTL_ROUTE", cls.cache_ttl_route)),
            cache_ttl_embedding=int(os.getenv("CACHE_TTL_EMBEDDING", cls.cache_ttl_embedding)),
            cache_semantic_threshold=float(os.getenv("CACHE_SEMANTIC_THRESHOLD", cls.cache_semantic_threshold)),
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "1").lower() not in ("0", "false", "no"),
            rate_limit_capacity=int(os.getenv("RATE_LIMIT_CAPACITY", cls.rate_limit_capacity)),
            rate_limit_refill=float(os.getenv("RATE_LIMIT_REFILL", cls.rate_limit_refill)),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'llm_api_key': '***' if self.llm_api_key else '',  # 脱敏，避免日志泄露
            'llm_base_url': self.llm_base_url,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'redis_db': self.redis_db,
            'redis_password': '***' if self.redis_password else '',
            'cache_enabled': self.cache_enabled,
            'cache_semantic_threshold': self.cache_semantic_threshold,
        }

# 默认配置实例：启动时从环境变量加载
DEFAULT_CONFIG = GraphRAGConfig.from_env()
