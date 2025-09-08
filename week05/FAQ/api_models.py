from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="用户查询问题", min_length=1, max_length=500)
    top_k: Optional[int] = Field(5, description="返回结果数量", ge=1, le=20)
    category: Optional[str] = Field(None, description="类别过滤")

class SearchResult(BaseModel):
    """搜索结果模型"""
    question: str = Field(..., description="FAQ 问题")
    answer: str = Field(..., description="FAQ 答案")
    category: str = Field(..., description="问题类别")
    score: float = Field(..., description="相似度分数")
    confidence: str = Field(..., description="置信度等级")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class SearchResponse(BaseModel):
    """搜索响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    results: List[SearchResult] = Field(default_factory=list, description="搜索结果")
    total: int = Field(..., description="结果总数")
    query: str = Field(..., description="原始查询")

class SimilarQuestion(BaseModel):
    """相似问题模型"""
    question: str = Field(..., description="相似问题")
    similarity: float = Field(..., description="相似度")
    category: str = Field(..., description="问题类别")

class SimilarQuestionsRequest(BaseModel):
    """相似问题请求模型"""
    question: str = Field(..., description="输入问题", min_length=1, max_length=500)
    top_k: Optional[int] = Field(5, description="返回数量", ge=1, le=10)

class SimilarQuestionsResponse(BaseModel):
    """相似问题响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    similar_questions: List[SimilarQuestion] = Field(default_factory=list, description="相似问题列表")
    total: int = Field(..., description="结果总数")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    message: Optional[str] = Field(None, description="状态消息")
    collection_stats: Optional[Dict[str, Any]] = Field(None, description="集合统计信息")
    embedding_model: Optional[str] = Field(None, description="嵌入模型名称")
    vector_dimension: Optional[int] = Field(None, description="向量维度")
    service_initialized: Optional[bool] = Field(None, description="服务是否初始化")
    file_watching: Optional[bool] = Field(None, description="是否启用文件监控")
    watch_directory: Optional[str] = Field(None, description="监控目录")

class StatsResponse(BaseModel):
    """统计信息响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    index_stats: Dict[str, Any] = Field(default_factory=dict, description="索引统计")
    category_stats: Dict[str, Any] = Field(default_factory=dict, description="类别统计")
    service_status: str = Field(..., description="服务状态")

class RebuildRequest(BaseModel):
    """重建索引请求模型"""
    file_path: Optional[str] = Field(None, description="FAQ 数据文件路径")

class RebuildResponse(BaseModel):
    """重建索引响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    file_path: str = Field(..., description="处理的文件路径")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(None, description="错误代码")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")