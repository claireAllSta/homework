"""
配置模块
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    
    # OpenAI API 配置
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_model: str = "deepseek-ai/DeepSeek-V3.1"
    
    # 应用配置
    app_name: str = "BasicConv"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()