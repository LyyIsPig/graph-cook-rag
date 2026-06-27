"""
智能查询路由器
根据查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的信息查找
- 图RAG检索：适合复杂的关系推理和知识发现
"""

import json
import logging
import asyncio
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

from rag_modules.graph_rag_retrieval import detect_relation_pattern

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略
    
@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    reasoning_required: bool  # 是否需要推理
    entity_count: int  # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float  # 推荐置信度
    reasoning: str  # 推荐理由

class IntelligentQueryRouter:
    """
    智能查询路由器
    
    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 结果质量监控：基于反馈优化路由决策
    """
    
    def __init__(self,
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,    # 图RAG检索模块
                 llm_client,
                 config,
                 cache=None):            # P1 L4 路由决策缓存（CacheManager 或 None）
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        self.cache = cache
        
        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }
        
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        深度分析查询特征，决定最佳检索策略
        """
        logger.info(f"分析查询特征: {query}")
        
        # 使用LLM进行智能分析
        analysis_prompt = f"""
分析以下查询并返回JSON结果。不要输出任何代码、解释或额外文字，只输出JSON对象。

查询：{query}

JSON格式（直接输出，不要用代码块包裹）：
{{"query_complexity": 0.6, "relationship_intensity": 0.8, "reasoning_required": true, "entity_count": 3, "recommended_strategy": "graph_rag", "confidence": 0.85, "reasoning": "分析理由"}}

字段说明：
- query_complexity (0-1): 0-0.3简单查找, 0.4-0.7中等, 0.8-1.0复杂推理
- relationship_intensity (0-1): 0-0.3单一实体, 0.4-0.7实体关系, 0.8-1.0复杂网络
- reasoning_required: 是否需要推理
- entity_count: 实体数量
- recommended_strategy: "hybrid_traditional" | "graph_rag" | "combined"
- confidence (0-1): 置信度
- reasoning: 推荐理由
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            if content is None:
                content = ""
            content = content.strip()
            if not content:
                raise ValueError("LLM返回为空")
            import re
            json_pattern = r'\{[^{}]*"query_complexity"[^{}]*\}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            else:
                raise ValueError(f"无法从返回中提取JSON: {content[:100]}")
            result = json.loads(content)
            
            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )
            
            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            # 降级方案：基于规则的简单分析
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于规则的降级分析"""
        # 简单的规则判断
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别"]
        relation_keywords = ["配", "搭配", "组合", "相关", "联系", "连接"]
        
        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)
        
        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL
            
        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的简单分析"
        )
    
    def _routing_policy(self, analysis: QueryAnalysis, query: str) -> SearchStrategy:
        """
        数据支撑的最终路由策略（覆盖 LLM 原始推荐）。
        依据 P2 评测（详见 问题与解决方案记录.md P2-1/P2-2/[B]）：
          - hybrid 在 lookup/list/reasoning 全部最强或并列；
          - graph_rag 通用子图在 relation 因无分类/工具过滤而召回 0；
          - 已实现改进待办 [B]：把关系查询编译成带过滤的精确 Cypher，relation 命中真值。
        故策略：
          ① 关系查询（共用食材/食材×分类/按工具/按做法）→ graph_rag（走 [B] 目标 Cypher）；
          ② 其余一律 hybrid（仍是最稳的兜底）。
        """
        if detect_relation_pattern(query):
            return SearchStrategy.GRAPH_RAG
        return SearchStrategy.HYBRID_TRADITIONAL

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最适合的检索引擎。
        P1 L4：路由决策缓存命中时跳过 analyze_query 的 LLM 调用。
        """
        logger.info(f"开始智能路由: {query}")

        # 0. P1 L4 路由决策缓存（命中 → 直接用缓存策略，省一次 LLM 分析）
        if self.cache is not None:
            cached_route = self.cache.get_route(query)
            if cached_route:
                analysis = self._analysis_from_dict(cached_route)
                logger.info(f"L4 路由缓存命中: {analysis.recommended_strategy.value}")
                return self._execute(analysis, query, top_k)

        # 1. 分析查询特征（LLM）
        analysis = self.analyze_query(query)
        llm_strategy = analysis.recommended_strategy

        # 1b. 用评测数据支撑的策略覆盖 LLM 的原始推荐
        analysis.recommended_strategy = self._routing_policy(analysis, query)
        if analysis.recommended_strategy != llm_strategy:
            logger.info(f"路由策略覆盖: LLM推荐={llm_strategy.value} "
                        f"→ 实际={analysis.recommended_strategy.value}")

        # 1c. 落 L4 路由缓存（Redis 不可用时 no-op）
        if self.cache is not None:
            self.cache.set_route(query, self._analysis_to_dict(analysis))

        return self._execute(analysis, query, top_k)

    def _execute(self, analysis: QueryAnalysis, query: str, top_k: int) -> Tuple[List[Document], QueryAnalysis]:
        """按既定策略执行检索 + 后处理。路由命中与未命中共用，避免逻辑分叉。"""
        self._update_route_stats(analysis.recommended_strategy)
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)
                # [B] 关系查询若 graph_rag 未命中（编译失败/0 结果），降级 hybrid 保召回
                if not documents and detect_relation_pattern(query):
                    logger.info("[B] graph_rag 关系未命中，降级 hybrid")
                    documents = self.traditional_retrieval.hybrid_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(query, top_k)

            documents = self._post_process_results(documents, analysis)
            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis

        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis

    async def route_query_async(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """P4 异步路由：analyze_query(LLM) 走 to_thread；dispatch 走各检索模块的 async 版（hybrid 三路并发）。
        L4 路由缓存命中则跳过 analyze_query。"""
        logger.info(f"开始智能路由(async): {query}")

        # L4 路由缓存（命中跳过 LLM analyze）
        if self.cache is not None:
            cached_route = self.cache.get_route(query)
            if cached_route:
                analysis = self._analysis_from_dict(cached_route)
                logger.info(f"L4 路由缓存命中: {analysis.recommended_strategy.value}")
                return await self._execute_async(analysis, query, top_k)

        # analyze_query 是阻塞 LLM 调用 → 卸到线程池
        analysis = await asyncio.to_thread(self.analyze_query, query)
        llm_strategy = analysis.recommended_strategy
        analysis.recommended_strategy = self._routing_policy(analysis, query)
        if analysis.recommended_strategy != llm_strategy:
            logger.info(f"路由策略覆盖: LLM推荐={llm_strategy.value} "
                        f"→ 实际={analysis.recommended_strategy.value}")
        if self.cache is not None:
            self.cache.set_route(query, self._analysis_to_dict(analysis))
        return await self._execute_async(analysis, query, top_k)

    async def _execute_async(self, analysis: QueryAnalysis, query: str, top_k: int) -> Tuple[List[Document], QueryAnalysis]:
        """异步执行检索：hybrid 走三路并发的 hybrid_search_async；graph_rag 走 async 包装。"""
        self._update_route_stats(analysis.recommended_strategy)
        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索(async, 三路并发)")
                documents = await self.traditional_retrieval.hybrid_search_async(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索(async)")
                documents = await self.graph_rag_retrieval.graph_rag_search_async(query, top_k)
                if not documents and detect_relation_pattern(query):
                    logger.info("[B] graph_rag 关系未命中，降级 hybrid")
                    documents = await self.traditional_retrieval.hybrid_search_async(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略(async)")
                documents = await asyncio.to_thread(self._combined_search, query, top_k)
            else:
                documents = await self.traditional_retrieval.hybrid_search_async(query, top_k)

            documents = self._post_process_results(documents, analysis)
            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis

        except Exception as e:
            logger.error(f"异步查询路由失败: {e}")
            documents = await asyncio.to_thread(self.traditional_retrieval.hybrid_search, query, top_k)
            return documents, analysis

    @staticmethod
    def _analysis_to_dict(analysis: QueryAnalysis) -> dict:
        """QueryAnalysis → 可 JSON 序列化的 dict（策略存 value 字符串）。"""
        return {
            "query_complexity": analysis.query_complexity,
            "relationship_intensity": analysis.relationship_intensity,
            "reasoning_required": analysis.reasoning_required,
            "entity_count": analysis.entity_count,
            "recommended_strategy": analysis.recommended_strategy.value,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
        }

    @staticmethod
    def _analysis_from_dict(d: dict) -> QueryAnalysis:
        """dict → QueryAnalysis（缓存命中时重建，省 analyze_query LLM 调用）。"""
        return QueryAnalysis(
            query_complexity=d.get("query_complexity", 0.5),
            relationship_intensity=d.get("relationship_intensity", 0.5),
            reasoning_required=d.get("reasoning_required", False),
            entity_count=d.get("entity_count", 1),
            recommended_strategy=SearchStrategy(d.get("recommended_strategy", "hybrid_traditional")),
            confidence=d.get("confidence", 0.5),
            reasoning=d.get("reasoning", "L4缓存命中"),
        )
    
    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """
        组合搜索策略：结合传统检索和图RAG的优势
        """
        # 分配结果数量
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k
        
        # 执行两种检索
        traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)
        
        # 合并和去重
        combined_docs = []
        seen_contents = set()
        
        # 交替添加结果（Round-robin）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图RAG结果（通常质量更高）
            if i < len(graph_docs):
                doc = graph_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"
                    combined_docs.append(doc)
            
            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"
                    combined_docs.append(doc)
        
        return combined_docs[:top_k]
    
    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """
        结果后处理：根据查询分析优化结果
        """
        for doc in documents:
            # 添加路由信息到元数据
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })
        
        return documents
    
    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计"""
        self.route_stats["total_queries"] += 1
        
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1
    
    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }
    
    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)
        
        explanation = f"""
        查询路由分析报告
        
        查询：{query}
        
        特征分析：
        - 复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体关系' if analysis.relationship_intensity < 0.8 else '复杂关系网络'})
        - 推理需求：{'是' if analysis.reasoning_required else '否'}
        - 实体数量：{analysis.entity_count}
        
        推荐策略：{analysis.recommended_strategy.value}
        置信度：{analysis.confidence:.2f}
        
        决策理由：{analysis.reasoning}
        """
        
        return explanation

 