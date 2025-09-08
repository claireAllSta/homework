from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys

from faq_service import faq_service
from api_models import (
    SearchRequest, SearchResponse, SearchResult,
    SimilarQuestionsRequest, SimilarQuestionsResponse,
    HealthResponse, StatsResponse,
    RebuildRequest, RebuildResponse,
    ErrorResponse
)
from config import settings

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/faq_service.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO"
)

# 新的 lifespan 上下文管理器方式（推荐）
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    logger.info("正在启动 FAQ 检索服务...")
    success = await faq_service.initialize()
    if not success:
        logger.error("FAQ 服务初始化失败")
        raise RuntimeError("服务初始化失败")
    
    logger.info("FAQ 检索服务启动成功")
    yield
    
    # 关闭时清理资源
    logger.info("正在关闭 FAQ 检索服务...")
    await faq_service.shutdown()
    logger.info("FAQ 检索服务已关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title="FAQ 检索系统",
    description="基于 Milvus 和 LlamaIndex 的智能 FAQ 检索系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="服务器内部错误",
            error_code="INTERNAL_ERROR",
            details={"error": str(exc)}
        ).dict()
    )

@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "message": "FAQ 检索系统",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/search", response_model=SearchResponse)
async def search_faq(request: SearchRequest):
    """
    搜索 FAQ
    
    根据用户查询返回最相关的 FAQ 条目
    """
    try:
        logger.info(f"收到搜索请求: {request.query}")
        
        # 执行搜索
        results = await faq_service.search(
            query=request.query,
            top_k=request.top_k,
            category=request.category
        )
        
        # 转换结果格式
        search_results = [
            SearchResult(
                question=result["question"],
                answer=result["answer"],
                category=result["category"],
                score=result["score"],
                confidence=result["confidence"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        response = SearchResponse(
            success=True,
            message="搜索完成",
            results=search_results,
            total=len(search_results),
            query=request.query
        )
        
        logger.info(f"搜索完成，返回 {len(search_results)} 条结果")
        return response
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"搜索失败: {str(e)}"
        )

@app.post("/similar-questions", response_model=SimilarQuestionsResponse)
async def get_similar_questions(request: SimilarQuestionsRequest):
    """
    获取相似问题
    
    根据输入问题返回相似的问题列表
    """
    try:
        logger.info(f"获取相似问题: {request.question}")
        
        results = await faq_service.get_similar_questions(
            question=request.question,
            top_k=request.top_k
        )
        
        response = SimilarQuestionsResponse(
            success=True,
            message="获取相似问题完成",
            similar_questions=results,
            total=len(results)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"获取相似问题失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取相似问题失败: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查
    
    返回服务的健康状态和统计信息
    """
    try:
        health_info = await faq_service.get_health_status()
        return HealthResponse(**health_info)
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthResponse(
            status="error",
            message=f"健康检查失败: {str(e)}"
        )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    获取统计信息
    
    返回索引和服务的统计信息
    """
    try:
        stats = await faq_service.get_stats()
        
        return StatsResponse(
            success=True,
            message="获取统计信息成功",
            index_stats=stats.get("index_stats", {}),
            category_stats=stats.get("category_stats", {}),
            service_status=stats.get("service_status", "unknown")
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )

@app.post("/rebuild-index", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest, background_tasks: BackgroundTasks):
    """
    重建索引
    
    重新构建 FAQ 索引（后台任务）
    """
    try:
        file_path = request.file_path or settings.FAQ_DATA_PATH
        
        logger.info(f"开始重建索引: {file_path}")
        
        # 在后台执行重建任务
        background_tasks.add_task(faq_service.rebuild_index, file_path)
        
        return RebuildResponse(
            success=True,
            message="索引重建任务已启动",
            file_path=file_path
        )
        
    except Exception as e:
        logger.error(f"启动重建索引任务失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"启动重建索引任务失败: {str(e)}"
        )

@app.get("/categories")
async def get_categories():
    """
    获取所有类别
    
    返回系统中所有的 FAQ 类别
    """
    try:
        stats = await faq_service.get_stats()
        categories = list(stats.get("category_stats", {}).keys())
        
        return {
            "success": True,
            "message": "获取类别成功",
            "categories": categories,
            "total": len(categories)
        }
        
    except Exception as e:
        logger.error(f"获取类别失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取类别失败: {str(e)}"
        )

if __name__ == "__main__":
    # 创建日志目录
    import os
    os.makedirs("logs", exist_ok=True)
    
    # 启动服务
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )