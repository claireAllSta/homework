"""
多轮对话与工具调用模块
基于 LangGraph 实现的智能对话系统，支持：
- 多轮对话状态管理
- 工具自动调用
- 热更新功能
- 生产部署支持
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .config import settings, AVAILABLE_TOOLS, ConversationState, MessageType
from .tools import ToolManager
from .graph import ConversationGraph
from .service import MultiConversationService

__all__ = [
    "settings",
    "AVAILABLE_TOOLS", 
    "ConversationState",
    "MessageType",
    "ToolManager",
    "ConversationGraph",
    "MultiConversationService"
]