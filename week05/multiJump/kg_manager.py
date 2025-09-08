"""
知识图谱管理器 - 使用 Neo4j 构建企业股权图谱
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class Entity:
    """实体类"""
    id: str
    name: str
    type: str  # company, person, organization
    properties: Dict[str, Any]

@dataclass
class Relation:
    """关系类"""
    source: str
    target: str
    relation_type: str  # shareholder, subsidiary, control
    properties: Dict[str, Any]

@dataclass
class QueryPath:
    """查询路径"""
    entities: List[Entity]
    relations: List[Relation]
    confidence: float
    reasoning: str

class KnowledgeGraphManager:
    """知识图谱管理器"""
    
    def __init__(self, uri: str, username: str, password: str):
        """初始化 Neo4j 连接"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = logging.getLogger(__name__)
        
    def close(self):
        """关闭连接"""
        self.driver.close()
        
    def create_indexes(self):
        """创建索引以提高查询性能"""
        with self.driver.session() as session:
            # 为实体名称创建索引
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            # 为公司创建索引
            session.run("CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.name)")
            # 为人员创建索引
            session.run("CREATE INDEX person_name IF NOT EXISTS FOR (n:Person) ON (n.name)")
            
    def add_entity(self, entity: Entity) -> bool:
        """添加实体到图谱"""
        try:
            with self.driver.session() as session:
                query = f"""
                MERGE (e:{entity.type.capitalize()} {{id: $id}})
                SET e.name = $name
                SET e += $properties
                RETURN e
                """
                result = session.run(query, 
                                   id=entity.id, 
                                   name=entity.name, 
                                   properties=entity.properties)
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"添加实体失败: {e}")
            return False
            
    def add_relation(self, relation: Relation) -> bool:
        """添加关系到图谱"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (a {{id: $source}})
                MATCH (b {{id: $target}})
                MERGE (a)-[r:{relation.relation_type.upper()}]->(b)
                SET r += $properties
                RETURN r
                """
                result = session.run(query,
                                   source=relation.source,
                                   target=relation.target,
                                   properties=relation.properties)
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"添加关系失败: {e}")
            return False
            
    def find_shareholders(self, company_name: str, max_hops: int = 3) -> List[QueryPath]:
        """查找公司股东（多跳查询）"""
        query = """
        MATCH path = (c:Company {name: $company_name})<-[:SHAREHOLDER*1..%d]-(s)
        WHERE s.name IS NOT NULL
        RETURN path, 
               [node in nodes(path) | {id: node.id, name: node.name, type: labels(node)[0]}] as entities,
               [rel in relationships(path) | {type: type(rel), properties: properties(rel)}] as relations,
               length(path) as hops
        ORDER BY hops ASC
        LIMIT 10
        """ % max_hops
        
        paths = []
        try:
            with self.driver.session() as session:
                result = session.run(query, company_name=company_name)
                
                for record in result:
                    entities = [Entity(
                        id=e['id'],
                        name=e['name'],
                        type=e['type'],
                        properties={}
                    ) for e in record['entities']]
                    
                    relations = [Relation(
                        source="",  # 需要从路径中推导
                        target="",
                        relation_type=r['type'],
                        properties=r['properties']
                    ) for r in record['relations']]
                    
                    # 计算置信度（基于路径长度和关系强度）
                    confidence = self._calculate_path_confidence(record['hops'], relations)
                    
                    # 生成推理说明
                    reasoning = self._generate_reasoning(entities, relations)
                    
                    paths.append(QueryPath(
                        entities=entities,
                        relations=relations,
                        confidence=confidence,
                        reasoning=reasoning
                    ))
                    
        except Exception as e:
            self.logger.error(f"查询股东失败: {e}")
            
        return paths
        
    def find_controlling_shareholder(self, company_name: str) -> Optional[QueryPath]:
        """查找控股股东"""
        query = """
        MATCH (c:Company {name: $company_name})<-[r:SHAREHOLDER]-(s)
        WHERE r.percentage > 50 OR r.control_type = 'controlling'
        RETURN s, r
        ORDER BY r.percentage DESC
        LIMIT 1
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, company_name=company_name)
                record = result.single()
                
                if record:
                    shareholder = record['s']
                    relation = record['r']
                    
                    entities = [
                        Entity(
                            id=shareholder['id'],
                            name=shareholder['name'],
                            type=list(shareholder.labels)[0],
                            properties=dict(shareholder)
                        )
                    ]
                    
                    relations = [Relation(
                        source=shareholder['id'],
                        target=company_name,
                        relation_type='SHAREHOLDER',
                        properties=dict(relation)
                    )]
                    
                    confidence = 0.9 if relation.get('percentage', 0) > 50 else 0.7
                    reasoning = f"{shareholder['name']} 持有 {company_name} {relation.get('percentage', 'N/A')}% 股份"
                    
                    return QueryPath(
                        entities=entities,
                        relations=relations,
                        confidence=confidence,
                        reasoning=reasoning
                    )
                    
        except Exception as e:
            self.logger.error(f"查询控股股东失败: {e}")
            
        return None
        
    def _calculate_path_confidence(self, hops: int, relations: List[Relation]) -> float:
        """计算路径置信度"""
        # 基础置信度随跳数递减
        base_confidence = max(0.1, 1.0 - (hops - 1) * 0.2)
        
        # 根据关系强度调整
        relation_bonus = 0
        for relation in relations:
            if relation.properties.get('percentage', 0) > 50:
                relation_bonus += 0.1
            elif relation.properties.get('control_type') == 'controlling':
                relation_bonus += 0.15
                
        return min(1.0, base_confidence + relation_bonus)
        
    def _generate_reasoning(self, entities: List[Entity], relations: List[Relation]) -> str:
        """生成推理说明"""
        if not entities or not relations:
            return "无法生成推理路径"
            
        reasoning_parts = []
        for i, entity in enumerate(entities[:-1]):
            if i < len(relations):
                relation = relations[i]
                next_entity = entities[i + 1] if i + 1 < len(entities) else None
                if next_entity:
                    percentage = relation.properties.get('percentage', '')
                    percentage_str = f"({percentage}%)" if percentage else ""
                    reasoning_parts.append(f"{entity.name} -> {relation.relation_type} {percentage_str} -> {next_entity.name}")
                    
        return " -> ".join(reasoning_parts)
        
    def execute_cypher_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行自定义 Cypher 查询"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"执行 Cypher 查询失败: {e}")
            return []