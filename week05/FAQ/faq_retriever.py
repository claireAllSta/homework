from typing import List, Dict, Any, Optional
from loguru import logger
from embedding_model import EmbeddingModel
from milvus_client import MilvusClient
from config import settings

class FAQRetriever:
    """FAQ 检索器，负责处理用户查询并返回相关结果"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.milvus_client = MilvusClient()
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化检索器"""
        try:
            # 连接 Milvus
            if not self.milvus_client.connect():
                return False
            
            # 检查集合是否存在
            from pymilvus import utility
            if not utility.has_collection(settings.COLLECTION_NAME):
                logger.error(f"集合 {settings.COLLECTION_NAME} 不存在，请先构建索引")
                return False
            
            # 获取集合
            from pymilvus import Collection
            self.milvus_client.collection = Collection(settings.COLLECTION_NAME)
            
            self._initialized = True
            logger.info("FAQ 检索器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化检索器失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = None, category: str = None) -> List[Dict[str, Any]]:
        """
        搜索相关的 FAQ
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            category: 指定类别过滤
            
        Returns:
            搜索结果列表
        """
        try:
            if not self._initialized:
                logger.error("检索器未初始化")
                return []
            
            if not query.strip():
                logger.warning("查询为空")
                return []
            
            # 第一步： 生成查询向量  
            query_embedding = self.embedding_model.encode(query)
            
            # 执行向量搜索
            search_results = self.milvus_client.search(
                query_embedding=query_embedding,
                top_k=top_k or settings.TOP_K
            )
            
            # 过滤和后处理结果
            filtered_results = self._post_process_results(search_results, category)
            
            logger.info(f"查询: '{query}' 返回 {len(filtered_results)} 条结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    # 查询后处理，可以做权限控制
    def _post_process_results(self, results: List[Dict[str, Any]], category: str = None) -> List[Dict[str, Any]]:
        """
        后处理搜索结果
        
        Args:
            results: 原始搜索结果
            category: 类别过滤
            
        Returns:
            处理后的结果
        """
        processed_results = []
        seen_questions = set()
        
        for result in results:
            # 类别过滤
            if category and result.get("category") != category:
                continue
            
            # 去重（基于问题）
            question = result.get("question", "")
            if question in seen_questions:
                continue
            seen_questions.add(question)
            
            # 格式化结果
            processed_result = {
                "question": question,
                "answer": result.get("answer", ""),
                "category": result.get("category", "general"),
                "score": result.get("score", 0.0),
                "confidence": self._calculate_confidence(result.get("score", 0.0)),
                "metadata": result.get("metadata", {})
            }
            
            processed_results.append(processed_result)
        
        # 按相似度分数排序
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        
        return processed_results
    
    def _calculate_confidence(self, score: float) -> str:
        """
        根据相似度分数计算置信度等级
        
        Args:
            score: 相似度分数
            
        Returns:
            置信度等级
        """
        if score >= 0.9:
            return "very_high"
        elif score >= 0.8:
            return "high"
        elif score >= 0.7:
            return "medium"
        elif score >= 0.6:
            return "low"
        else:
            return "very_low"
    
    def get_similar_questions(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取相似问题
        
        Args:
            question: 输入问题
            top_k: 返回数量
            
        Returns:
            相似问题列表
        """
        results = self.search(question, top_k=top_k)
        
        similar_questions = []
        for result in results:
            similar_questions.append({
                "question": result["question"],
                "similarity": result["score"],
                "category": result["category"]
            })
        
        return similar_questions
    
    def get_category_stats(self) -> Dict[str, int]:
        """获取各类别的统计信息"""
        try:
            # 这里简化实现，实际可以通过 Milvus 查询获取
            # 由于 Milvus 不直接支持聚合查询，这里返回空字典
            # 在实际应用中可以维护一个单独的统计表
            return {}
        except Exception as e:
            logger.error(f"获取类别统计失败: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self._initialized:
                return {"status": "error", "message": "检索器未初始化"}
            
            # 获取集合统计信息
            stats = self.milvus_client.get_collection_stats()
            
            return {
                "status": "healthy",
                "collection_stats": stats,
                "embedding_model": self.embedding_model.model_name,
                "vector_dimension": self.embedding_model.get_dimension()
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """关闭检索器"""
        self.milvus_client.disconnect()
        self._initialized = False