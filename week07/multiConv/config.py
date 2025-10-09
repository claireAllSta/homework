"""
多轮对话配置模块
"""
import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolConfig:
    """工具配置"""
    name: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool = True
    hot_reload: bool = True


class Settings(BaseSettings):
    """应用配置"""
    
    # OpenAI API 配置
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_model: str = "deepseek-ai/DeepSeek-V3.1"
    
    # 应用配置
    app_name: str = "MultiConv"
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"
    
    # 对话配置
    max_conversation_history: int = 20
    session_timeout_minutes: int = 30
    
    # 工具配置
    tools_directory: str = "tools"
    enable_hot_reload: bool = True
    hot_reload_interval: int = 5  # 秒
    
    # 健康检查配置
    health_check_timeout: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 工具注册表
AVAILABLE_TOOLS = {
    "query_order": ToolConfig(
        name="query_order",
        description="查询订单状态和物流信息",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号"
                }
            },
            "required": ["order_id"]
        }
    ),
    "apply_refund": ToolConfig(
        name="apply_refund",
        description="申请退款",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号"
                },
                "reason": {
                    "type": "string",
                    "description": "退款原因"
                },
                "amount": {
                    "type": "number",
                    "description": "退款金额，不填则全额退款",
                    "default": None
                }
            },
            "required": ["order_id", "reason"]
        }
    ),
    "issue_invoice": ToolConfig(
        name="issue_invoice",
        description="开具发票",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号"
                },
                "invoice_type": {
                    "type": "string",
                    "enum": ["个人", "企业"],
                    "description": "发票类型"
                },
                "invoice_title": {
                    "type": "string",
                    "description": "发票抬头"
                },
                "tax_number": {
                    "type": "string",
                    "description": "税号（企业发票必填）",
                    "default": None
                }
            },
            "required": ["order_id", "invoice_type", "invoice_title"]
        }
    )
}

# 全局配置实例
settings = Settings()

# 对话状态枚举
class ConversationState:
    """对话状态定义"""
    START = "start"
    WAITING_ORDER_ID = "waiting_order_id"
    WAITING_REFUND_REASON = "waiting_refund_reason"
    WAITING_INVOICE_INFO = "waiting_invoice_info"
    TOOL_CALLING = "tool_calling"
    COMPLETED = "completed"
    ERROR = "error"

# 消息类型
class MessageType:
    """消息类型定义"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"