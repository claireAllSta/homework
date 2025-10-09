"""
基础对话链实现
使用 LangChain 构建 Prompt → LLM → OutputParser 链
"""
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import settings
from output_parser import TimeInferenceOutputParser


class BasicConversationChain:
    """基础对话链，实现时间推断功能"""
    
    def __init__(self, current_date: Optional[datetime] = None):
        """
        初始化对话链
        
        Args:
            current_date: 当前日期，默认为今天
        """
        self.current_date = current_date or datetime.now()
        self.llm = self._create_llm()
        self.prompt = self._create_prompt()
        self.output_parser = TimeInferenceOutputParser(self.current_date)
        self.chain = self._create_chain()
    
    def _create_llm(self) -> ChatOpenAI:
        """创建LLM实例"""
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.1,
        }
        
        if settings.openai_api_key:
            llm_kwargs["openai_api_key"] = settings.openai_api_key
        
        if settings.openai_api_base:
            llm_kwargs["openai_api_base"] = settings.openai_api_base
        
        return ChatOpenAI(**llm_kwargs)
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """创建提示模板"""
        template = """你是一个智能时间推断助手。用户会提到一些相对时间概念，你需要根据当前时间推断出具体的日期。

当前时间信息：
- 当前日期：{current_date}
- 当前是星期：{current_weekday}

用户查询：{user_query}

请分析用户查询中的时间概念，并推断出具体的日期。请按以下格式输出：

```json
{{
    "original_query": "用户的原始查询",
    "inferred_date": "推断出的具体日期(YYYY-MM-DD格式)",
    "explanation": "推断过程的详细说明",
    "current_date": "当前日期"
}}
```

推断规则：
1. "昨天" = 当前日期 - 1天
2. "前天" = 当前日期 - 2天  
3. "明天" = 当前日期 + 1天
4. "后天" = 当前日期 + 2天
5. "上周" = 当前日期 - 7天
6. "下周" = 当前日期 + 7天
7. "上个月" = 当前日期 - 30天（大概）
8. "下个月" = 当前日期 + 30天（大概）

请确保推断准确，并提供清晰的解释。"""

        return ChatPromptTemplate.from_template(template)
    
    def _create_chain(self):
        """创建完整的处理链"""
        # 构建链：输入处理 → Prompt → LLM → OutputParser
        chain = (
            RunnablePassthrough.assign(
                current_date=lambda x: self.current_date.strftime('%Y-%m-%d'),
                current_weekday=lambda x: self._get_weekday_chinese()
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
            | self.output_parser
        )
        
        return chain
    
    def _get_weekday_chinese(self) -> str:
        """获取中文星期"""
        weekdays = {
            0: "星期一",
            1: "星期二", 
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日"
        }
        return weekdays[self.current_date.weekday()]
    
    def invoke(self, user_query: str) -> Dict[str, Any]:
        """
        调用对话链处理用户查询
        
        Args:
            user_query: 用户查询文本
            
        Returns:
            包含推断结果的字典
        """
        try:
            result = self.chain.invoke({"user_query": user_query})
            return result
        except Exception as e:
            return {
                "original_query": user_query,
                "inferred_date": self.current_date.strftime('%Y-%m-%d'),
                "explanation": f"处理查询时出错: {str(e)}",
                "current_date": self.current_date.strftime('%Y-%m-%d'),
                "error": True
            }
    
