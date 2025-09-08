import json
import os
from typing import List, Dict, Any, Optional
from loguru import logger
from text_splitter import SemanticTextSplitter
from embedding_model import EmbeddingModel
from milvus_client import MilvusClient

class FAQIndexer:
    """FAQ 索引器，负责构建和管理 FAQ 知识库索引"""
    
    def __init__(self):
        self.text_splitter = SemanticTextSplitter()
        self.embedding_model = EmbeddingModel()
        self.milvus_client = MilvusClient()
        
    def initialize(self) -> bool:
        """初始化索引器"""
        try:
            # 连接 Milvus
            if not self.milvus_client.connect():
                return False
            
            # 创建集合
            if not self.milvus_client.create_collection():
                return False
            
            # 创建索引
            if not self.milvus_client.create_index():
                return False
            
            logger.info("FAQ 索引器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化索引器失败: {e}")
            return False
    
    def load_faq_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从文件加载 FAQ 数据
        
        Args:
            file_path: FAQ 数据文件路径
            
        Returns:
            FAQ 数据列表
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"FAQ 数据文件不存在: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if not isinstance(data, list):
                logger.error("FAQ 数据格式错误，应为列表格式")
                return []
            
            # 验证每个条目的必要字段
            valid_data = []
            for item in data:
                if self._validate_faq_item(item):
                    valid_data.append(item)
                else:
                    logger.warning(f"跳过无效的 FAQ 条目: {item}")
            
            logger.info(f"成功加载 {len(valid_data)} 条 FAQ 数据")
            return valid_data
            
        except Exception as e:
            logger.error(f"加载 FAQ 数据失败: {e}")
            return []
    
    def _validate_faq_item(self, item: Dict[str, Any]) -> bool:
        """验证 FAQ 条目格式"""
        required_fields = ["question", "answer"]
        return all(field in item and item[field] for field in required_fields)
    
    def process_faq_data(self, faq_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理 FAQ 数据，进行文本切分和向量化
        
        Args:
            faq_data: 原始 FAQ 数据
            
        Returns:
            处理后的数据，包含向量嵌入
        """
        processed_data = []
        
        for faq_item in faq_data:
            try:
                question = faq_item["question"]
                answer = faq_item["answer"]
                category = faq_item.get("category", "general")
                
                # 组合问题和答案作为完整文本
                full_text = f"问题: {question}\n答案: {answer}"
                
                # 文本切分
                metadata = {
                    "original_question": question,
                    "original_answer": answer,
                    "category": category,
                    "faq_id": faq_item.get("id", len(processed_data))
                }
                
                chunks = self.text_splitter.split_text(full_text, metadata)
                
                # 为每个文本块生成嵌入向量
                for chunk in chunks:
                    chunk_text = chunk["text"]
                    chunk_metadata = chunk["metadata"]
                    
                    # 生成嵌入向量
                    embedding = self.embedding_model.encode(chunk_text)
                    
                    processed_item = {
                        "text": chunk_text,
                        "embedding": embedding,
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "metadata": chunk_metadata
                    }
                    
                    processed_data.append(processed_item)
                    
            except Exception as e:
                logger.error(f"处理 FAQ 条目失败: {e}, 条目: {faq_item}")
                continue
        
        logger.info(f"FAQ 数据处理完成，生成 {len(processed_data)} 个文本块")
        return processed_data
    
    def build_index(self, faq_data: List[Dict[str, Any]]) -> bool:
        """
        构建 FAQ 索引
        
        Args:
            faq_data: FAQ 数据列表
            
        Returns:
            是否构建成功
        """
        try:
            if not faq_data:
                logger.warning("没有 FAQ 数据需要索引")
                return True
            
            # 处理 FAQ 数据
            processed_data = self.process_faq_data(faq_data)
            
            if not processed_data:
                logger.warning("没有有效的处理数据")
                return False
            
            # 插入到 Milvus
            success = self.milvus_client.insert_data(processed_data)
            
            if success:
                logger.info("FAQ 索引构建成功")
                # 打印统计信息
                stats = self.milvus_client.get_collection_stats()
                logger.info(f"集合统计信息: {stats}")
            
            return success
            
        except Exception as e:
            logger.error(f"构建索引失败: {e}")
            return False
    
    def rebuild_index(self, faq_data: List[Dict[str, Any]]) -> bool:
        """
        重建索引（删除现有数据后重新构建）
        
        Args:
            faq_data: FAQ 数据列表
            
        Returns:
            是否重建成功
        """
        try:
            logger.info("开始重建 FAQ 索引")
            
            # 删除现有集合
            self.milvus_client.delete_collection()
            
            # 重新初始化
            if not self.initialize():
                return False
            
            # 构建新索引
            return self.build_index(faq_data)
            
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False
    
    def update_index(self, file_path: str) -> bool:
        """
        更新索引（热更新）
        
        Args:
            file_path: FAQ 数据文件路径
            
        Returns:
            是否更新成功
        """
        try:
            logger.info(f"开始更新 FAQ 索引: {file_path}")
            
            # 加载新数据
            faq_data = self.load_faq_data(file_path)
            
            if not faq_data:
                logger.warning("没有新数据需要更新")
                return True
            
            # 重建索引（简单实现，实际可以做增量更新）
            return self.rebuild_index(faq_data)
            
        except Exception as e:
            logger.error(f"更新索引失败: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return self.milvus_client.get_collection_stats()
    
    def close(self):
        """关闭索引器"""
        self.milvus_client.disconnect()