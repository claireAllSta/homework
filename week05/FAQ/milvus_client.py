from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, Index
)
from loguru import logger
from config import settings

class MilvusClient:
    """Milvus 向量数据库客户端"""
    
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME
        self.dimension = settings.DIMENSION
        self.collection: Optional[Collection] = None
        
    def connect(self) -> bool:
        """连接到 Milvus 服务器"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD
            )
            logger.info(f"成功连接到 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            return True
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            return False
    
    def create_collection(self) -> bool:
        """创建集合"""
        try:
            # 检查集合是否已存在
            if utility.has_collection(self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在")
                self.collection = Collection(self.collection_name)
                return True
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(
                fields=fields,
                description="FAQ 检索系统集合"
            )
            
            # 创建集合
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def create_index(self) -> bool:
        """创建向量索引"""
        try:
            # 检查是否已有索引
            if self.collection.has_index():
                logger.info("索引已存在")
                return True
            
            # 创建索引参数
            index_params = {
                "metric_type": settings.METRIC_TYPE,
                "index_type": settings.INDEX_TYPE,
                "params": {"nlist": settings.NLIST}
            }
            
            # 创建索引
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("成功创建向量索引")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def insert_data(self, data: List[Dict[str, Any]]) -> bool:
        """插入数据"""
        try:
            if not data:
                logger.warning("没有数据需要插入")
                return True
            
            # 准备插入数据
            texts = [item["text"] for item in data]
            embeddings = [item["embedding"] for item in data]
            questions = [item["question"] for item in data]
            answers = [item["answer"] for item in data]
            categories = [item.get("category", "general") for item in data]
            metadata = [item.get("metadata", {}) for item in data]
            
            # 插入数据
            insert_data = [
                texts,
                embeddings,
                questions,
                answers,
                categories,
                metadata
            ]
            
            mr = self.collection.insert(insert_data)
            self.collection.flush()
            
            logger.info(f"成功插入 {len(data)} 条数据")
            return True
            
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            if top_k is None:
                top_k = settings.TOP_K
            
            # 加载集合到内存
            self.collection.load()
            
            # 搜索参数
            search_params = {
                "metric_type": settings.METRIC_TYPE,
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "question", "answer", "category", "metadata"]
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= settings.SIMILARITY_THRESHOLD:
                        result = {
                            "id": hit.id,
                            "score": hit.score,
                            "text": hit.entity.get("text"),
                            "question": hit.entity.get("question"),
                            "answer": hit.entity.get("answer"),
                            "category": hit.entity.get("category"),
                            "metadata": hit.entity.get("metadata", {})
                        }
                        search_results.append(result)
            
            logger.info(f"搜索完成，返回 {len(search_results)} 条结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def delete_collection(self) -> bool:
        """删除集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"成功删除集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    # pymilvus的新方法
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                return {}
            
            # 使用 num_entities 属性获取实体数量
            entity_count = self.collection.num_entities
            
            return {
                "entity_count": entity_count,
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {}
    
    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect("default")
            logger.info("已断开 Milvus 连接")
        except Exception as e:
            logger.error(f"断开连接失败: {e}")