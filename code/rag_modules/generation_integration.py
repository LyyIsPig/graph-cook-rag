"""
生成集成模块
"""

import logging
import os
import time
from typing import List

from openai import OpenAI, AsyncOpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""

    def __init__(self, model_name: str = "glm-4-flash", temperature: float = 0.1, max_tokens: int = 2048,
                 api_key: "str | None" = None, base_url: "str | None" = None,
                 config: "GraphRAGConfig | None" = None):
        """
        初始化生成集成模块（P0：LLM 提供商统一为智谱 GLM，配置从 config/.env 注入）。
        密钥解析优先级：显式 api_key → config.llm_api_key → 环境变量 LLM_API_KEY/ZHIPU_API_KEY/MOONSHOT_API_KEY。
        """
        # 传入的 config 优先（来自 main.py 注入）
        if config is not None:
            model_name = model_name or config.llm_model
            api_key = api_key or config.llm_api_key
            base_url = base_url or config.llm_base_url

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("ZHIPU_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("未找到 LLM API Key，请在 .env 设置 LLM_API_KEY / ZHIPU_API_KEY（智谱 GLM）")

        base_url = base_url or os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # P4 异步：AsyncOpenAI 与同步 client 共享同一密钥/endpoint，给 async 管道用
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"生成模块初始化完成，模型: {model_name}, endpoint: {base_url}")

    def _build_prompt(self, question: str, documents: List[Document]) -> str:
        """构造生成提示词（同步/异步共用，含拒答指令 P2-3）。"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)
        context = "\n\n".join(context_parts)
        return f"""
作为一位专业的烹饪助手，请严格【只基于以下检索信息】回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

请提供准确、实用的回答。根据问题的性质：
- 如果是询问多个菜品，请提供清晰的列表
- 如果是询问具体制作方法，请提供详细步骤
- 如果是一般性咨询，请提供综合性回答

【拒答规则（仅在以下情况触发，避免误伤正常查询）】
- 用户问【某道具体菜谱】（如"X怎么做"），但检索到的【明显是另一道菜名对不上的菜】→ 拒答；
- 检索内容与问题【完全无关】（如问菜谱却检索到天气/编程/无关内容）→ 拒答。
【以下情况不要拒答，必须尽力据已有信息作答】
- "有哪些 / 用了X的 / 和X共用…的菜"这类【列表或关系】查询：即使检索结果不全，
  也要根据已检索到的相关菜作答，绝不要因为"没有完整列表"就拒答。
触发拒答时回复："抱歉，我的菜谱库里没有这道菜的相关信息，无法回答。"
绝不要根据菜名编造不存在的菜谱、食材或烹饪步骤。

回答：
"""

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """
        智能统一答案生成（同步）。
        自动适应不同类型的查询，无需预先分类。
        """
        prompt = self._build_prompt(question, documents)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LightRAG答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    async def generate_adaptive_answer_async(self, question: str, documents: List[Document]) -> str:
        """P4 异步生成：用 AsyncOpenAI，让事件循环在等 LLM 时让出（吞吐关键）。"""
        prompt = self._build_prompt(question, documents)
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"异步答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        """
        LightRAG风格的流式答案生成（带重试机制）
        """
        # 构建上下文
        context_parts = []
        
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)
        
        context = "\n\n".join(context_parts)
        
        # LightRAG风格的统一提示词（含拒答指令，防幻觉——P2-3）
        prompt = f"""
        作为一位专业的烹饪助手，请严格【只基于以下检索信息】回答用户的问题。

        检索到的相关信息：
        {context}

        用户问题：{question}

        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个菜品，请提供清晰的列表
        - 如果是询问具体制作方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答

        【拒答规则（仅在以下情况触发，避免误伤正常查询）】
        - 用户问【某道具体菜谱】（如"X怎么做"），但检索到的【明显是另一道菜名对不上的菜】→ 拒答；
        - 检索内容与问题【完全无关】（如问菜谱却检索到天气/编程/无关内容）→ 拒答。
        【以下情况不要拒答，必须尽力据已有信息作答】
        - "有哪些 / 用了X的 / 和X共用…的菜"这类【列表或关系】查询：即使检索结果不全，
          也要根据已检索到的相关菜作答，绝不要因为"没有完整列表"就拒答。
        触发拒答时回复："抱歉，我的菜谱库里没有这道菜的相关信息，无法回答。"
        绝不要根据菜名编造不存在的菜谱、食材或烹饪步骤。

        回答：
        """
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60  # 增加超时设置
                )
                
                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")
                
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content  # 使用yield返回流式内容
                
                # 如果成功完成，退出重试循环
                return
                
            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败，使用非流式作为后备
                    logger.error(f"流式生成完全失败，尝试非流式后备方案")
                    print("⚠️ 流式生成失败，切换到标准模式...")
                    
                    try:
                        fallback_response = self.generate_adaptive_answer(question, documents)
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        logger.error(f"后备生成也失败: {fallback_error}")
                        error_msg = f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                        yield error_msg
                        return 