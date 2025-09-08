"""
配置管理模块
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """系统配置"""
    
    # OpenAI配置
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field("https://api.openai.com/v1", env="OPENAI_BASE_URL")
    
    # Neo4j配置
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    
    # 系统配置
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_hops: int = Field(3, env="MAX_HOPS")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    
    # 模型配置
    embedding_model: str = "BAAI/bge-large-zh-v1.5"  # SiliconFlow支持的中文embedding模型
    llm_model: str = "deepseek-ai/DeepSeek-V3.1"
    temperature: float = 0.1
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 全局配置实例
settings = Settings()