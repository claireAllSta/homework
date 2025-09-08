"""
多跳查询协调器 - 协调图谱查询、RAG检索和LLM推理
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from loguru import logger
from openai import OpenAI
from datetime import datetime

from config import settings
from graph_manager import Neo4jManager
from kg_manager import KnowledgeGraphManager, QueryPath, Entity, Relation


class QueryType(Enum):
    """查询类型枚举"""
    SHAREHOLDER = "shareholder"  # 股东查询
    CONTROL_CHAIN = "control_chain"  # 控制链查询
    RELATIONSHIP = "relationship"  # 关系查询
    ENTITY_INFO = "entity_info"  # 实体信息查询


@dataclass
class QueryRequest:
    """查询请求"""
    query: str
    query_type: QueryType
    entity: str
    max_hops: int = 3
    similarity_threshold: float = 0.7
    context: Dict[str, Any] = None


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    query_type: QueryType
    paths: List[QueryPath]
    answer: str
    confidence: float
    reasoning_steps: List[str]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    description: str
    data_source: str  # graph, rag, llm
    result: Dict[str, Any]
    confidence: float


class MultihopCoordinator:
    """多跳查询协调器"""
    
    def __init__(self):
        """初始化协调器"""
        self.neo4j_manager = Neo4jManager()
        self.kg_manager = KnowledgeGraphManager(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password
        )
        
        # 初始化OpenAI客户端
        self.openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        
        logger.info("多跳查询协调器初始化完成")
    
    def close(self):
        """关闭所有连接"""
        self.neo4j_manager.close()
        self.kg_manager.close()
    
    async def process_query(self, request: QueryRequest) -> QueryResult:
        """处理查询请求"""
        start_time = datetime.now()
        reasoning_steps = []
        
        try:
            logger.info(f"开始处理查询: {request.query}")
            
            # 步骤1: 查询类型识别和实体提取
            step1 = await self._identify_query_intent(request)
            reasoning_steps.append(step1)
            
            # 步骤2: 图谱多跳查询
            step2 = await self._execute_graph_query(request)
            reasoning_steps.append(step2)
            
            # 步骤3: 结果整合和推理
            step3 = await self._integrate_and_reason(request, step2.result)
            reasoning_steps.append(step3)
            
            # 步骤4: 生成最终答案
            final_answer = await self._generate_final_answer(request, reasoning_steps)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 构建查询结果
            result = QueryResult(
                query=request.query,
                query_type=request.query_type,
                paths=step2.result.get('paths', []),
                answer=final_answer['answer'],
                confidence=final_answer['confidence'],
                reasoning_steps=[step.description for step in reasoning_steps],
                execution_time=execution_time,
                metadata={
                    'entity': request.entity,
                    'max_hops': request.max_hops,
                    'steps_count': len(reasoning_steps),
                    'graph_results_count': len(step2.result.get('paths', []))
                }
            )
            
            logger.info(f"查询处理完成，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=request.query,
                query_type=request.query_type,
                paths=[],
                answer=f"查询处理失败: {str(e)}",
                confidence=0.0,
                reasoning_steps=[f"错误: {str(e)}"],
                execution_time=execution_time,
                metadata={'error': str(e)}
            )
    
    async def _identify_query_intent(self, request: QueryRequest) -> ReasoningStep:
        """识别查询意图和提取实体"""
        try:
            # 使用LLM进行意图识别
            prompt = f"""
            分析以下查询的意图和关键实体：
            查询: {request.query}
            
            请识别：
            1. 查询类型（股东查询/控制链查询/关系查询/实体信息查询）
            2. 关键实体名称
            3. 查询的具体目标
            
            以JSON格式返回结果。
            """
            
            response = await self._call_llm(prompt)
            intent_result = json.loads(response)
            
            return ReasoningStep(
                step_id=1,
                description=f"意图识别: {intent_result.get('query_type', '未知')}",
                data_source="llm",
                result=intent_result,
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return ReasoningStep(
                step_id=1,
                description=f"意图识别失败: {str(e)}",
                data_source="llm",
                result={},
                confidence=0.0
            )
    
    async def _execute_graph_query(self, request: QueryRequest) -> ReasoningStep:
        """执行图谱查询"""
        try:
            paths = []
            
            if request.query_type == QueryType.SHAREHOLDER:
                # 股东查询
                paths = self.kg_manager.find_shareholders(
                    company_name=request.entity,
                    max_hops=request.max_hops
                )
                
            elif request.query_type == QueryType.CONTROL_CHAIN:
                # 控制链查询
                controlling_path = self.kg_manager.find_controlling_shareholder(request.entity)
                if controlling_path:
                    paths = [controlling_path]
                    
            elif request.query_type == QueryType.RELATIONSHIP:
                # 关系查询 - 使用Neo4j管理器
                multi_hop_results = self.neo4j_manager.find_multi_hop_relationships(
                    start_entity=request.entity,
                    relationship_type="HOLDS",
                    max_hops=request.max_hops
                )
                
                # 转换为QueryPath格式
                for result in multi_hop_results:
                    entities = [Entity(
                        id=node.get('id', ''),
                        name=node.get('name', ''),
                        type=node.get('labels', ['Unknown'])[0],
                        properties={}
                    ) for node in result['nodes']]
                    
                    relations = [Relation(
                        source="",
                        target="",
                        relation_type=rel['type'],
                        properties=rel['properties']
                    ) for rel in result['relationships']]
                    
                    paths.append(QueryPath(
                        entities=entities,
                        relations=relations,
                        confidence=max(0.1, 1.0 - result['hop_count'] * 0.2),
                        reasoning=f"多跳路径，跳数: {result['hop_count']}"
                    ))
                    
            elif request.query_type == QueryType.ENTITY_INFO:
                # 实体信息查询
                neighbors = self.neo4j_manager.get_entity_neighbors(
                    entity_name=request.entity,
                    max_depth=2
                )
                
                # 构建实体信息路径
                if neighbors['neighbors']:
                    entities = [Entity(
                        id=request.entity,
                        name=request.entity,
                        type="Entity",
                        properties={}
                    )]
                    
                    for neighbor in neighbors['neighbors'][:5]:  # 限制邻居数量
                        entities.append(Entity(
                            id=neighbor['name'],
                            name=neighbor['name'],
                            type=neighbor['labels'][0] if neighbor['labels'] else 'Unknown',
                            properties={'distance': neighbor['distance']}
                        ))
                    
                    paths.append(QueryPath(
                        entities=entities,
                        relations=[],
                        confidence=0.7,
                        reasoning=f"实体 {request.entity} 的邻居信息"
                    ))
            
            return ReasoningStep(
                step_id=2,
                description=f"图谱查询完成，找到 {len(paths)} 条路径",
                data_source="graph",
                result={'paths': paths},
                confidence=0.9 if paths else 0.1
            )
            
        except Exception as e:
            logger.error(f"图谱查询失败: {e}")
            return ReasoningStep(
                step_id=2,
                description=f"图谱查询失败: {str(e)}",
                data_source="graph",
                result={'paths': []},
                confidence=0.0
            )
    
    async def _integrate_and_reason(self, request: QueryRequest, graph_results: Dict[str, Any]) -> ReasoningStep:
        """整合结果并进行推理"""
        try:
            paths = graph_results.get('paths', [])
            
            if not paths:
                return ReasoningStep(
                    step_id=3,
                    description="未找到相关路径，无法进行推理",
                    data_source="integration",
                    result={'integrated_info': '无相关信息'},
                    confidence=0.0
                )
            
            # 整合路径信息
            integrated_info = {
                'total_paths': len(paths),
                'high_confidence_paths': [p for p in paths if p.confidence > 0.7],
                'entities_involved': set(),
                'relationship_types': set()
            }
            
            for path in paths:
                for entity in path.entities:
                    integrated_info['entities_involved'].add(entity.name)
                for relation in path.relations:
                    integrated_info['relationship_types'].add(relation.relation_type)
            
            # 转换集合为列表以便JSON序列化
            integrated_info['entities_involved'] = list(integrated_info['entities_involved'])
            integrated_info['relationship_types'] = list(integrated_info['relationship_types'])
            
            # 计算整合置信度
            avg_confidence = sum(p.confidence for p in paths) / len(paths) if paths else 0.0
            
            return ReasoningStep(
                step_id=3,
                description=f"结果整合完成，涉及 {len(integrated_info['entities_involved'])} 个实体",
                data_source="integration",
                result={'integrated_info': integrated_info, 'paths': paths},
                confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            return ReasoningStep(
                step_id=3,
                description=f"结果整合失败: {str(e)}",
                data_source="integration",
                result={'integrated_info': {}},
                confidence=0.0
            )
    
    async def _generate_final_answer(self, request: QueryRequest, reasoning_steps: List[ReasoningStep]) -> Dict[str, Any]:
        """生成最终答案"""
        try:
            # 收集所有推理信息
            context_info = []
            for step in reasoning_steps:
                context_info.append(f"步骤{step.step_id}: {step.description}")
                if step.data_source == "graph" and 'paths' in step.result:
                    paths = step.result['paths']
                    for i, path in enumerate(paths[:3]):  # 只展示前3条路径
                        context_info.append(f"  路径{i+1}: {path.reasoning} (置信度: {path.confidence:.2f})")
            
            # 构建LLM提示
            prompt = f"""
            基于以下信息回答用户查询：
            
            用户查询: {request.query}
            查询实体: {request.entity}
            查询类型: {request.query_type.value}
            
            推理过程:
            {chr(10).join(context_info)}
            
            请提供：
            1. 简洁明确的答案
            2. 答案的置信度评分（0-1）
            3. 简要的推理说明
            
            以JSON格式返回结果，包含answer、confidence、explanation字段。
            """
            
            response = await self._call_llm(prompt)
            result = json.loads(response)
            
            return {
                'answer': result.get('answer', '无法生成答案'),
                'confidence': float(result.get('confidence', 0.0)),
                'explanation': result.get('explanation', '无推理说明')
            }
            
        except Exception as e:
            logger.error(f"生成最终答案失败: {e}")
            return {
                'answer': f'答案生成失败: {str(e)}',
                'confidence': 0.0,
                'explanation': '系统错误'
            }
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的企业关系分析助手，擅长分析股权结构和企业关系。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return json.dumps({
                "error": f"LLM调用失败: {str(e)}",
                "answer": "系统暂时无法处理该查询",
                "confidence": 0.0,
                "explanation": "LLM服务不可用"
            })
    
    def batch_process_queries(self, requests: List[QueryRequest]) -> List[QueryResult]:
        """批量处理查询"""
        results = []
        
        for request in requests:
            try:
                # 同步调用异步方法
                result = asyncio.run(self.process_query(request))
                results.append(result)
            except Exception as e:
                logger.error(f"批量处理查询失败: {e}")
                results.append(QueryResult(
                    query=request.query,
                    query_type=request.query_type,
                    paths=[],
                    answer=f"处理失败: {str(e)}",
                    confidence=0.0,
                    reasoning_steps=[],
                    execution_time=0.0,
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def get_query_statistics(self, results: List[QueryResult]) -> Dict[str, Any]:
        """获取查询统计信息"""
        if not results:
            return {}
        
        total_queries = len(results)
        successful_queries = len([r for r in results if r.confidence > 0.5])
        avg_confidence = sum(r.confidence for r in results) / total_queries
        avg_execution_time = sum(r.execution_time for r in results) / total_queries
        
        query_type_stats = {}
        for result in results:
            query_type = result.query_type.value
            if query_type not in query_type_stats:
                query_type_stats[query_type] = {'count': 0, 'success_rate': 0}
            query_type_stats[query_type]['count'] += 1
            if result.confidence > 0.5:
                query_type_stats[query_type]['success_rate'] += 1
        
        # 计算成功率
        for query_type in query_type_stats:
            count = query_type_stats[query_type]['count']
            success_count = query_type_stats[query_type]['success_rate']
            query_type_stats[query_type]['success_rate'] = success_count / count if count > 0 else 0
        
        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / total_queries,
            'average_confidence': avg_confidence,
            'average_execution_time': avg_execution_time,
            'query_type_statistics': query_type_stats
        }

