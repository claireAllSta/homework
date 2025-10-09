"""
多轮对话服务和会话管理
"""
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from loguru import logger

from .config import settings, ConversationState, MessageType
from .graph import ConversationGraph


@dataclass
class ConversationSession:
    """对话会话"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages: List[BaseMessage] = field(default_factory=list)
    current_state: str = ConversationState.START
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        timeout = timedelta(minutes=settings.session_timeout_minutes)
        return datetime.now() - self.last_activity > timeout
    
    def update_activity(self):
        """更新最后活动时间"""
        self.last_activity = datetime.now()
    
    def add_message(self, message: BaseMessage):
        """添加消息"""
        self.messages.append(message)
        self.update_activity()
        
        # 限制消息历史长度
        if len(self.messages) > settings.max_conversation_history:
            self.messages = self.messages[-settings.max_conversation_history:]
    
    def get_recent_messages(self, count: int = 10) -> List[BaseMessage]:
        """获取最近的消息"""
        return self.messages[-count:] if count > 0 else self.messages


class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.cleanup_interval = 300  # 5分钟清理一次
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        def cleanup_expired_sessions():
            while True:
                try:
                    expired_sessions = [
                        session_id for session_id, session in self.sessions.items()
                        if session.is_expired()
                    ]
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        logger.info(f"清理过期会话: {session_id}")
                    
                    if expired_sessions:
                        logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
                    
                except Exception as e:
                    logger.error(f"会话清理失败: {str(e)}")
                
                # 等待下次清理
                import time
                time.sleep(self.cleanup_interval)
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
        cleanup_thread.start()
    
    def create_session(self, user_id: Optional[str] = None) -> ConversationSession:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id
        )
        self.sessions[session_id] = session
        logger.info(f"创建新会话: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        session = self.sessions.get(session_id)
        if session and not session.is_expired():
            session.update_activity()
            return session
        elif session and session.is_expired():
            # 删除过期会话
            del self.sessions[session_id]
            logger.info(f"会话已过期并删除: {session_id}")
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False
    
    def get_active_sessions_count(self) -> int:
        """获取活跃会话数量"""
        return len([s for s in self.sessions.values() if not s.is_expired()])
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        active_sessions = [s for s in self.sessions.values() if not s.is_expired()]
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(active_sessions),
            "average_messages": sum(len(s.messages) for s in active_sessions) / len(active_sessions) if active_sessions else 0,
            "oldest_session": min((s.created_at for s in active_sessions), default=None),
            "newest_session": max((s.created_at for s in active_sessions), default=None)
        }


class MultiConversationService:
    """多轮对话服务"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.conversation_graph = ConversationGraph()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._stats = {
            "total_conversations": 0,
            "successful_conversations": 0,
            "failed_conversations": 0,
            "tool_calls": 0,
            "start_time": datetime.now()
        }
    
    async def start_conversation(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """开始新对话"""
        try:
            session = self.session_manager.create_session(user_id)
            
            # 添加欢迎消息
            welcome_message = AIMessage(content="您好！我是智能客服助手，可以帮您查询订单、申请退款、开具发票等。请告诉我您需要什么帮助？")
            session.add_message(welcome_message)
            
            self._stats["total_conversations"] += 1
            
            return {
                "success": True,
                "session_id": session.session_id,
                "message": welcome_message.content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"开始对话失败: {str(e)}")
            self._stats["failed_conversations"] += 1
            return {
                "success": False,
                "error": f"开始对话失败: {str(e)}"
            }
    
    async def send_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """发送消息"""
        try:
            # 获取会话
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": "会话不存在或已过期，请重新开始对话"
                }
            
            # 处理消息
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_message_sync,
                session, user_message
            )
            
            # 更新统计
            if result["success"]:
                self._stats["successful_conversations"] += 1
                if "tool_result" in result.get("extracted_info", {}):
                    self._stats["tool_calls"] += 1
            else:
                self._stats["failed_conversations"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")
            self._stats["failed_conversations"] += 1
            return {
                "success": False,
                "error": f"消息处理失败: {str(e)}"
            }
    
    def _process_message_sync(self, session: ConversationSession, user_message: str) -> Dict[str, Any]:
        """同步处理消息"""
        try:
            # 添加用户消息到会话
            session.add_message(HumanMessage(content=user_message))
            
            # 使用对话图处理消息
            result = self.conversation_graph.process_message(
                session.session_id,
                user_message,
                session.get_recent_messages(10)  # 传递最近10条消息作为上下文
            )
            
            if result["success"]:
                # 更新会话状态
                session.current_state = result.get("state", ConversationState.START)
                session.context.update(result.get("extracted_info", {}))
                
                # 添加AI响应到会话
                if result["response"]:
                    session.add_message(AIMessage(content=result["response"]))
                
                return {
                    "success": True,
                    "response": result["response"],
                    "session_id": session.session_id,
                    "state": session.current_state,
                    "intent": result.get("intent"),
                    "extracted_info": result.get("extracted_info", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"对话图处理失败: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "消息处理失败"),
                    "session_id": session.session_id
                }
                
        except Exception as e:
            logger.error(f"同步消息处理失败: {str(e)}")
            return {
                "success": False,
                "error": f"消息处理失败: {str(e)}"
            }
    
    async def get_conversation_history(self, session_id: str, limit: int = 20) -> Dict[str, Any]:
        """获取对话历史"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": "会话不存在或已过期"
                }
            
            messages = session.get_recent_messages(limit)
            
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({
                        "role": "user",
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    })
                elif isinstance(msg, AIMessage):
                    history.append({
                        "role": "assistant", 
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    })
            
            return {
                "success": True,
                "session_id": session_id,
                "history": history,
                "total_messages": len(messages),
                "session_created": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取对话历史失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取对话历史失败: {str(e)}"
            }
    
    async def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """结束对话"""
        try:
            success = self.session_manager.delete_session(session_id)
            
            if success:
                return {
                    "success": True,
                    "message": "对话已结束，感谢使用！"
                }
            else:
                return {
                    "success": False,
                    "error": "会话不存在"
                }
                
        except Exception as e:
            logger.error(f"结束对话失败: {str(e)}")
            return {
                "success": False,
                "error": f"结束对话失败: {str(e)}"
            }
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        try:
            session_stats = self.session_manager.get_session_stats()
            uptime = datetime.now() - self._stats["start_time"]
            
            return {
                "success": True,
                "stats": {
                    "service_uptime": str(uptime),
                    "total_conversations": self._stats["total_conversations"],
                    "successful_conversations": self._stats["successful_conversations"],
                    "failed_conversations": self._stats["failed_conversations"],
                    "tool_calls": self._stats["tool_calls"],
                    "success_rate": (self._stats["successful_conversations"] / max(self._stats["total_conversations"], 1)) * 100,
                    "session_stats": session_stats
                }
            }
            
        except Exception as e:
            logger.error(f"获取服务统计失败: {str(e)}")
            return {
                "success": False,
                "error": f"获取统计信息失败: {str(e)}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查组件状态
            checks = {
                "session_manager": self.session_manager is not None,
                "conversation_graph": self.conversation_graph is not None,
                "executor": self.executor is not None,
                "active_sessions": self.session_manager.get_active_sessions_count()
            }
            
            # 简单的功能测试
            test_session = self.session_manager.create_session("health_check")
            test_result = self.conversation_graph.process_message(
                test_session.session_id,
                "你好",
                []
            )
            self.session_manager.delete_session(test_session.session_id)
            
            checks["conversation_test"] = test_result.get("success", False)
            
            all_healthy = all(checks.values())
            
            return {
                "healthy": all_healthy,
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def shutdown(self):
        """关闭服务"""
        try:
            logger.info("正在关闭多轮对话服务...")
            self.executor.shutdown(wait=True)
            logger.info("多轮对话服务已关闭")
        except Exception as e:
            logger.error(f"关闭服务失败: {str(e)}")


# 全局服务实例
conversation_service = MultiConversationService()
