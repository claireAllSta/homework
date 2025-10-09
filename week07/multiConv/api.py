"""
FastAPI接口实现
提供多轮对话、健康检查、热更新等API接口
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
from loguru import logger

from .config import settings
from .service import conversation_service
from .hot_update import hot_updater
from .tools import tool_manager


# Pydantic模型定义
class ConversationStartRequest(BaseModel):
    """开始对话请求"""
    user_id: Optional[str] = Field(None, description="用户ID")


class MessageRequest(BaseModel):
    """发送消息请求"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息")


class ConversationHistoryRequest(BaseModel):
    """获取对话历史请求"""
    session_id: str = Field(..., description="会话ID")
    limit: int = Field(20, description="消息数量限制", ge=1, le=100)


class EndConversationRequest(BaseModel):
    """结束对话请求"""
    session_id: str = Field(..., description="会话ID")


class HotUpdateRequest(BaseModel):
    """热更新请求"""
    action: str = Field(..., description="操作类型: reload_all, reload_plugin, reload_model")
    target: Optional[str] = Field(None, description="目标文件路径或插件名")


class HealthResponse(BaseModel):
    """健康检查响应"""
    healthy: bool
    timestamp: str
    checks: Dict[str, Any]
    version: str


class ConversationResponse(BaseModel):
    """对话响应"""
    success: bool
    session_id: Optional[str] = None
    response: Optional[str] = None
    state: Optional[str] = None
    intent: Optional[str] = None
    timestamp: str
    error: Optional[str] = None


# 创建FastAPI应用
app = FastAPI(
    title="多轮对话系统",
    description="基于LangGraph的智能多轮对话与工具调用系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 依赖项
async def get_conversation_service():
    """获取对话服务依赖"""
    return conversation_service


async def get_hot_updater():
    """获取热更新器依赖"""
    return hot_updater


# 健康检查接口
@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """
    系统健康检查
    
    检查各个组件的运行状态，包括：
    - 会话管理器状态
    - 对话图状态
    - 热更新状态
    - 工具管理器状态
    """
    try:
        # 获取各组件健康状态
        service_health = await conversation_service.health_check()
        hot_update_status = hot_updater.get_status()
        
        # 合并检查结果
        all_checks = {
            **service_health.get("checks", {}),
            "hot_updater": hot_updater.is_running,
            "loaded_plugins": len(hot_updater.plugin_manager.list_plugins()),
            "tool_manager": len(tool_manager.list_available_tools())
        }
        
        # 判断整体健康状态
        overall_healthy = (
            service_health.get("healthy", False) and
            hot_updater.is_running and
            len(tool_manager.list_available_tools()) > 0
        )
        
        return HealthResponse(
            healthy=overall_healthy,
            timestamp=datetime.now().isoformat(),
            checks=all_checks,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return HealthResponse(
            healthy=False,
            timestamp=datetime.now().isoformat(),
            checks={"error": str(e)},
            version="1.0.0"
        )


# 对话相关接口
@app.post("/conversation/start", response_model=ConversationResponse, tags=["对话管理"])
async def start_conversation(
    request: ConversationStartRequest,
    service=Depends(get_conversation_service)
):
    """
    开始新的对话会话
    
    创建一个新的对话会话，返回会话ID和欢迎消息
    """
    try:
        result = await service.start_conversation(request.user_id)
        
        if result["success"]:
            return ConversationResponse(
                success=True,
                session_id=result["session_id"],
                response=result["message"],
                state="start",
                timestamp=result["timestamp"]
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "启动对话失败"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动对话失败: {str(e)}")


@app.post("/conversation/message", response_model=ConversationResponse, tags=["对话管理"])
async def send_message(
    request: MessageRequest,
    service=Depends(get_conversation_service)
):
    """
    发送消息到指定会话
    
    处理用户消息，返回AI响应和对话状态
    """
    try:
        result = await service.send_message(request.session_id, request.message)
        
        if result["success"]:
            return ConversationResponse(
                success=True,
                session_id=result["session_id"],
                response=result["response"],
                state=result.get("state"),
                intent=result.get("intent"),
                timestamp=result["timestamp"]
            )
        else:
            return ConversationResponse(
                success=False,
                session_id=request.session_id,
                error=result.get("error", "消息处理失败"),
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"发送消息失败: {str(e)}")
        return ConversationResponse(
            success=False,
            session_id=request.session_id,
            error=f"消息处理失败: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


@app.post("/conversation/history", tags=["对话管理"])
async def get_conversation_history(
    request: ConversationHistoryRequest,
    service=Depends(get_conversation_service)
):
    """
    获取对话历史记录
    
    返回指定会话的历史消息记录
    """
    try:
        result = await service.get_conversation_history(request.session_id, request.limit)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=404, detail=result.get("error", "获取历史记录失败"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


@app.post("/conversation/end", tags=["对话管理"])
async def end_conversation(
    request: EndConversationRequest,
    service=Depends(get_conversation_service)
):
    """
    结束对话会话
    
    删除指定的对话会话
    """
    try:
        result = await service.end_conversation(request.session_id)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=404, detail=result.get("error", "结束对话失败"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"结束对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"结束对话失败: {str(e)}")


# 统计接口
@app.get("/stats", tags=["系统统计"])
async def get_service_stats(service=Depends(get_conversation_service)):
    """
    获取服务统计信息
    
    返回对话服务的运行统计数据
    """
    try:
        result = await service.get_service_stats()
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "获取统计信息失败"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# 热更新接口
@app.post("/admin/hot-update", tags=["系统管理"])
async def hot_update(
    request: HotUpdateRequest,
    background_tasks: BackgroundTasks,
    updater=Depends(get_hot_updater)
):
    """
    执行热更新操作
    
    支持的操作：
    - reload_all: 重新加载所有插件和工具
    - reload_plugin: 重新加载指定插件
    - reload_model: 重新加载模型配置
    """
    try:
        if request.action == "reload_all":
            # 在后台执行重载操作
            background_tasks.add_task(updater.manual_reload_all)
            return {"success": True, "message": "开始重新加载所有组件"}
            
        elif request.action == "reload_plugin" and request.target:
            success = updater.reload_plugin(request.target)
            return {
                "success": success,
                "message": f"重新加载插件 {request.target} {'成功' if success else '失败'}"
            }
            
        elif request.action == "reload_model" and request.target:
            success = updater.reload_model_config(request.target)
            return {
                "success": success,
                "message": f"重新加载模型配置 {request.target} {'成功' if success else '失败'}"
            }
            
        else:
            raise HTTPException(status_code=400, detail="无效的热更新操作")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"热更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"热更新失败: {str(e)}")


@app.get("/admin/hot-update/status", tags=["系统管理"])
async def get_hot_update_status(updater=Depends(get_hot_updater)):
    """
    获取热更新状态
    
    返回热更新器的运行状态和历史记录
    """
    try:
        status = updater.get_status()
        history = updater.get_update_history(10)
        
        return {
            "status": status,
            "recent_updates": history
        }
        
    except Exception as e:
        logger.error(f"获取热更新状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@app.get("/admin/hot-update/history", tags=["系统管理"])
async def get_hot_update_history(
    limit: int = 20,
    updater=Depends(get_hot_updater)
):
    """
    获取热更新历史记录
    
    返回指定数量的热更新历史记录
    """
    try:
        history = updater.get_update_history(limit)
        return {"history": history}
        
    except Exception as e:
        logger.error(f"获取热更新历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 工具管理接口
@app.get("/admin/tools", tags=["工具管理"])
async def list_tools():
    """
    列出所有可用工具
    
    返回系统中注册的所有工具列表和配置
    """
    try:
        tools = tool_manager.list_available_tools()
        tool_info = {}
        
        for tool_name in tools:
            config = tool_manager.get_tool_config(tool_name)
            if config:
                tool_info[tool_name] = {
                    "name": config.name,
                    "description": config.description,
                    "enabled": config.enabled,
                    "hot_reload": config.hot_reload,
                    "parameters": config.parameters
                }
        
        return {"tools": tool_info}
        
    except Exception as e:
        logger.error(f"列出工具失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取工具列表失败: {str(e)}")


@app.post("/admin/tools/reload", tags=["工具管理"])
async def reload_tools():
    """
    重新加载所有工具
    
    手动触发工具管理器的重新加载
    """
    try:
        success = tool_manager.reload_tools()
        return {
            "success": success,
            "message": f"工具重新加载{'成功' if success else '失败'}",
            "available_tools": tool_manager.list_available_tools()
        }
        
    except Exception as e:
        logger.error(f"重新加载工具失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重新加载工具失败: {str(e)}")


# 应用启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("多轮对话系统启动中...")
    
    # 启动热更新监视（如果尚未启动）
    if settings.enable_hot_reload and not hot_updater.is_running:
        hot_updater.start_watching()
    
    logger.info("多轮对话系统启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("多轮对话系统关闭中...")
    
    # 关闭各种服务
    conversation_service.shutdown()
    hot_updater.shutdown()
    
    logger.info("多轮对话系统已关闭")


# 根路径
@app.get("/", tags=["基础"])
async def root():
    """
    根路径，返回系统信息
    """
    return {
        "name": "多轮对话与工具调用系统",
        "version": "1.0.0",
        "description": "基于LangGraph的智能多轮对话系统，支持工具调用和热更新",
        "features": [
            "多轮对话状态管理",
            "订单查询、退款申请、发票开具",
            "模型与插件热更新",
            "会话管理",
            "健康检查",
            "生产部署支持"
        ],
        "docs": "/docs",
        "health": "/health"
    }


def main():
    """启动应用"""
    try:
        logger.info(f"启动多轮对话系统，监听 {settings.host}:{settings.port}")
        
        uvicorn.run(
            "api:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="info" if not settings.debug else "debug"
        )
        
    except Exception as e:
        logger.error(f"启动应用失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()