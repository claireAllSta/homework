from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings

class EmbeddingModel:
    """文本嵌入模型"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
            
        Returns:
            单个向量或向量列表
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                single_text = True
            else:
                single_text = False
            
            # 编码文本
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 10
            )
            
            # 转换为列表格式
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # 返回单个向量或向量列表
            if single_text:
                return embeddings[0]
            else:
                return embeddings
                
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0