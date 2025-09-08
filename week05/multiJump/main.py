"""
简化的多跳查询系统主入口
避免typer版本兼容性问题
"""
import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config import settings
from rag_manager import SimpleRAGManager
from multihop_coordinator import MultihopCoordinator, QueryRequest, QueryType

# 初始化
console = Console()

# 全局组件
rag_manager: Optional[SimpleRAGManager] = None

# Pydantic模型
class QueryRequestModel(BaseModel):
    query: str
    entity: str = ""
    max_results: int = 5

class QueryResponseModel(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str

# FastAPI生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_manager
    
    # 启动
    try:
        logger.info("启动简化RAG系统...")
        rag_manager = SimpleRAGManager()
        logger.info("系统启动完成")
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise
    
    yield
    
    # 关闭
    try:
        if rag_manager:
            rag_manager.close()
        logger.info("系统关闭完成")
    except Exception as e:
        logger.error(f"系统关闭失败: {e}")

# 创建FastAPI应用
app = FastAPI(title="简化多跳查询系统", version="1.0.0", lifespan=lifespan)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "简化多跳查询系统API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "init_data": "/init_data",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "components": {
            "rag": rag_manager is not None
        }
    }


@app.post("/query", response_model=QueryResponseModel)
async def api_query(request: QueryRequestModel):
    """API查询接口"""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        # 执行混合搜索
        result = rag_manager.hybrid_search(
            query=request.query,
            entity_filter=request.entity if request.entity else None
        )
        
        # 返回结果
        return QueryResponseModel(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            reasoning=result["reasoning"]
        )
        
    except Exception as e:
        logger.error(f"API查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="系统未初始化")
    
    try:
        stats = rag_manager.get_document_stats()
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

# 命令行功能
def init_sample_data():
    """初始化示例数据"""
    console.print("[bold blue]正在初始化示例数据...[/bold blue]")
    
    try:
        rag = SimpleRAGManager()
        success = rag.load_sample_documents()
        
        if success:
            console.print("[bold green]✓ 示例数据初始化完成[/bold green]")
            
            # 测试查询
            console.print("[bold blue]测试查询功能...[/bold blue]")
            result = rag.hybrid_search("字节跳动的最大股东是谁？", entity_filter="字节跳动")
            
            # 显示结果
            table = Table(title="查询结果")
            table.add_column("字段", style="cyan")
            table.add_column("值", style="green")
            
            table.add_row("查询", result["query"])
            table.add_row("答案", result["answer"])
            table.add_row("置信度", f"{result['confidence']:.2f}")
            table.add_row("推理", result["reasoning"])
            
            console.print(table)
            
        else:
            console.print("[bold red]✗ 示例数据初始化失败[/bold red]")
            
        rag.close()
        
    except Exception as e:
        console.print(f"[bold red]✗ 初始化失败: {e}[/bold red]")
        logger.error(f"初始化失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化多跳查询系统")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    
    args = parser.parse_args()
    
    init_sample_data()

if __name__ == "__main__":
    main()