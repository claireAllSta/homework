"""
简化的RAG管理器
避免llama-index的导入问题，使用基础的向量检索和生成
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
from loguru import logger
import openai
from neo4j import GraphDatabase
import requests

from config import settings


@dataclass
class RAGDocument:
    """RAG文档"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SimpleRAGManager:
    """简化的RAG管理器"""
    
    def __init__(self):
        """初始化RAG管理器"""
        # 配置OpenAI客户端
        self.openai_client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
     
        # 创建向量索引
        self._create_vector_index()
        
        logger.info("简化RAG管理器初始化完成")

    def _create_vector_index(self):
        """创建向量索引"""
        try:
            with self.driver.session() as session:
                # 创建向量索引
                session.run("""
                    CREATE VECTOR INDEX rag_embeddings IF NOT EXISTS
                    FOR (n:Document) ON (n.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Neo4j向量索引创建完成")
        except Exception as e:
            logger.error(f"创建向量索引失败: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
            return []
    
    def add_documents(self, documents: List[RAGDocument]) -> bool:
        """添加文档到向量存储"""
        try:
            with self.driver.session() as session:
                for doc in documents:
                    # 获取嵌入
                    if not doc.embedding:
                        doc.embedding = self._get_embedding(doc.content)
                    
                    if doc.embedding:
                        # 存储到Neo4j
                        session.run("""
                            MERGE (d:Document {id: $doc_id})
                            SET d.content = $content,
                                d.metadata = $metadata,
                                d.embedding = $embedding
                        """, {
                            "doc_id": doc.id,
                            "content": doc.content,
                            "metadata": json.dumps(doc.metadata),
                            "embedding": doc.embedding
                        })
            
            logger.info(f"成功添加 {len(documents)} 个文档到RAG系统")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """检索相关文档"""
        try:
            # 获取查询嵌入
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []
            
            with self.driver.session() as session:
                # 使用向量相似度搜索
                result = session.run("""
                    CALL db.index.vector.queryNodes('rag_embeddings', $top_k, $query_embedding)
                    YIELD node, score
                    RETURN node.id as doc_id, node.content as content, 
                           node.metadata as metadata, score
                    ORDER BY score DESC
                """, {
                    "top_k": top_k,
                    "query_embedding": query_embedding
                })
                
                results = []
                for record in result:
                    metadata = json.loads(record["metadata"]) if record["metadata"] else {}
                    results.append(RetrievalResult(
                        document_id=record["doc_id"],
                        content=record["content"],
                        score=record["score"],
                        metadata=metadata
                    ))
                
                logger.info(f"检索到 {len(results)} 个相关文档")
                return results
                
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[RetrievalResult] = None) -> Dict[str, Any]:
        """基于检索结果生成答案"""
        try:
            if context_docs is None:
                context_docs = self.retrieve_relevant_docs(query)
            
            if not context_docs:
                return {
                    "answer": "抱歉，没有找到相关信息来回答您的问题。",
                    "confidence": 0.0,
                    "sources": [],
                    "reasoning": "未找到相关文档"
                }
            
            # 构建上下文
            context_text = "\n\n".join([
                f"文档{i+1}: {doc.content}" 
                for i, doc in enumerate(context_docs)
            ])
            
            prompt = f"""
            基于以下上下文信息回答问题：
            
            上下文：
            {context_text}
            
            问题：{query}
            
            请提供准确、简洁的答案，并说明答案的来源。
            """
            
            # 调用OpenAI生成答案
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识问答助手，基于提供的上下文信息准确回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.temperature
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "confidence": self._calculate_answer_confidence(context_docs),
                "sources": [doc.document_id for doc in context_docs],
                "reasoning": f"基于 {len(context_docs)} 个相关文档的分析"
            }
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return {
                "answer": f"答案生成失败: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "reasoning": "系统错误"
            }
    
    def hybrid_search(self, query: str, entity_filter: str = None) -> Dict[str, Any]:
        """混合搜索：结合向量检索和实体过滤"""
        try:
            # 1. 向量检索
            vector_results = self.retrieve_relevant_docs(query, top_k=10)
            
            # 2. 实体过滤（如果提供）
            if entity_filter:
                filtered_results = []
                for result in vector_results:
                    if (entity_filter.lower() in result.content.lower() or 
                        entity_filter.lower() in str(result.metadata).lower()):
                        filtered_results.append(result)
                vector_results = filtered_results
            
            # 3. 重新排序（基于分数和相关性）
            vector_results.sort(key=lambda x: x.score, reverse=True)
            
            # 4. 生成答案
            answer_result = self.generate_answer(query, vector_results[:5])
            
            return {
                "query": query,
                "retrieved_docs": len(vector_results),
                "top_docs": vector_results[:3],  # 返回前3个文档
                "answer": answer_result["answer"],
                "confidence": answer_result["confidence"],
                "sources": answer_result["sources"],
                "reasoning": answer_result["reasoning"]
            }
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return {
                "query": query,
                "retrieved_docs": 0,
                "top_docs": [],
                "answer": f"搜索失败: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "reasoning": "系统错误"
            }
    
    def _calculate_answer_confidence(self, context_docs: List[RetrievalResult]) -> float:
        """计算答案置信度"""
        if not context_docs:
            return 0.0
        
        # 基于检索分数计算置信度
        avg_score = sum(doc.score for doc in context_docs) / len(context_docs)
        
        # 归一化到0-1范围
        confidence = min(1.0, max(0.0, avg_score))
        
        # 根据文档数量调整
        doc_count_factor = min(1.0, len(context_docs) / 3.0)  # 3个文档为最佳
        
        return confidence * doc_count_factor
    
    def load_sample_documents(self):
        """加载示例文档"""
        sample_docs = [
            RAGDocument(
                id="doc_001",
                content="腾讯控股有限公司是中国领先的互联网增值服务提供商之一。马化腾是腾讯的创始人和最大个人股东，持有约8.5%的股份。",
                metadata={"type": "company_info", "entity": "腾讯控股", "category": "股权信息"}
            ),
            RAGDocument(
                id="doc_002", 
                content="阿里巴巴集团控股有限公司是中国最大的电子商务公司。软银集团是阿里巴巴的重要股东，持有约25.2%的股份。",
                metadata={"type": "company_info", "entity": "阿里巴巴", "category": "股权信息"}
            ),
            RAGDocument(
                id="doc_003",
                content="字节跳动是中国知名的互联网技术公司，旗下拥有抖音、今日头条等产品。张一鸣是公司创始人，红杉资本是重要投资方。",
                metadata={"type": "company_info", "entity": "字节跳动", "category": "股权信息"}
            ),
            RAGDocument(
                id="doc_004",
                content="美团是中国领先的生活服务电子商务平台。王兴是美团的创始人兼CEO，持有公司约10.4%的股份。",
                metadata={"type": "company_info", "entity": "美团", "category": "股权信息"}
            ),
            RAGDocument(
                id="doc_005",
                content="小米集团是一家以智能手机、智能硬件和IoT平台为核心的互联网公司。雷军是小米创始人，持有约13.4%的股份。",
                metadata={"type": "company_info", "entity": "小米集团", "category": "股权信息"}
            )
        ]
        
        success = self.add_documents(sample_docs)
        if success:
            logger.info("示例文档加载完成")
        else:
            logger.error("示例文档加载失败")
        
        return success
    
    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN count(d) as total")
                total_docs = result.single()["total"]
                
                return {
                    "total_documents": total_docs,
                    "index_type": "Neo4j Vector Index",
                    "embedding_model": settings.embedding_model,
                    "vector_store": "Neo4j"
                }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭连接"""
        try:
            self.driver.close()
            logger.info("RAG管理器连接已关闭")
        except Exception as e:
            logger.error(f"关闭RAG管理器失败: {e}")


# 测试函数
def test_simple_rag_manager():
    """测试简化RAG管理器"""
    rag = SimpleRAGManager()
    
    try:
        # 加载示例文档
        rag.load_sample_documents()
        
        # 测试检索
        query = "腾讯的最大股东是谁？"
        results = rag.retrieve_relevant_docs(query)
        logger.info(f"检索结果数量: {len(results)}")
        
        # 测试混合搜索
        hybrid_result = rag.hybrid_search(query, entity_filter="腾讯")
        logger.info(f"混合搜索答案: {hybrid_result['answer']}")
        
        # 获取统计信息
        stats = rag.get_document_stats()
        logger.info(f"文档统计: {stats}")
        
    finally:
        rag.close()


if __name__ == "__main__":
    test_simple_rag_manager()