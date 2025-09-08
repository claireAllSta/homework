"""
Neo4j图谱管理模块
"""
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from loguru import logger
import json

from config import settings

class Neo4jManager:
    """Neo4j图谱管理器"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        logger.info(f"连接到Neo4j: {settings.neo4j_uri}")
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def create_indexes(self):
        """创建索引"""
        with self.driver.session() as session:
            # 为公司节点创建索引
            session.run("CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)")
            session.run("CREATE INDEX company_id IF NOT EXISTS FOR (c:Company) ON (c.id)")
            
            # 为股东节点创建索引
            session.run("CREATE INDEX shareholder_name IF NOT EXISTS FOR (s:Shareholder) ON (s.name)")
            session.run("CREATE INDEX shareholder_id IF NOT EXISTS FOR (s:Shareholder) ON (s.id)")
            
            logger.info("创建图谱索引完成")
    
    def load_sample_data(self):
        """加载示例数据"""
        with self.driver.session() as session:
            # 清空现有数据
            session.run("MATCH (n) DETACH DELETE n")
            
            # 创建公司节点
            companies = [
                {"id": "comp_001", "name": "腾讯控股", "type": "上市公司", "industry": "互联网"},
                {"id": "comp_002", "name": "阿里巴巴", "type": "上市公司", "industry": "电商"},
                {"id": "comp_003", "name": "字节跳动", "type": "私人公司", "industry": "互联网"},
                {"id": "comp_004", "name": "美团", "type": "上市公司", "industry": "生活服务"},
                {"id": "comp_005", "name": "小米集团", "type": "上市公司", "industry": "智能硬件"}
            ]
            
            for company in companies:
                session.run("""
                    CREATE (c:Company {
                        id: $id, 
                        name: $name, 
                        type: $type, 
                        industry: $industry
                    })
                """, **company)
            
            # 创建股东节点
            shareholders = [
                {"id": "sh_001", "name": "马化腾", "type": "个人", "nationality": "中国"},
                {"id": "sh_002", "name": "马云", "type": "个人", "nationality": "中国"},
                {"id": "sh_003", "name": "张一鸣", "type": "个人", "nationality": "中国"},
                {"id": "sh_004", "name": "王兴", "type": "个人", "nationality": "中国"},
                {"id": "sh_005", "name": "雷军", "type": "个人", "nationality": "中国"},
                {"id": "sh_006", "name": "红杉资本", "type": "机构", "nationality": "美国"},
                {"id": "sh_007", "name": "软银集团", "type": "机构", "nationality": "日本"}
            ]
            
            for shareholder in shareholders:
                session.run("""
                    CREATE (s:Shareholder {
                        id: $id, 
                        name: $name, 
                        type: $type, 
                        nationality: $nationality
                    })
                """, **shareholder)
            
            # 创建持股关系
            holdings = [
                {"shareholder": "sh_001", "company": "comp_001", "percentage": 8.5, "is_largest": True},
                {"shareholder": "sh_002", "company": "comp_002", "percentage": 4.8, "is_largest": True},
                {"shareholder": "sh_003", "company": "comp_003", "percentage": 20.2, "is_largest": True},
                {"shareholder": "sh_004", "company": "comp_004", "percentage": 10.4, "is_largest": True},
                {"shareholder": "sh_005", "company": "comp_005", "percentage": 13.4, "is_largest": True},
                {"shareholder": "sh_006", "company": "comp_003", "percentage": 8.0, "is_largest": False},
                {"shareholder": "sh_007", "company": "comp_002", "percentage": 25.2, "is_largest": False}
            ]
            
            for holding in holdings:
                session.run("""
                    MATCH (s:Shareholder {id: $shareholder})
                    MATCH (c:Company {id: $company})
                    CREATE (s)-[:HOLDS {
                        percentage: $percentage, 
                        is_largest_shareholder: $is_largest,
                        created_at: datetime()
                    }]->(c)
                """, **holding)
            
            logger.info("加载示例数据完成")
 
    def find_multi_hop_relationships(self, start_entity: str, relationship_type: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """多跳关系查询"""
        with self.driver.session() as session:
            # 构建动态Cypher查询
            cypher_query = f"""
                MATCH path = (start)-[:{relationship_type}*1..{max_hops}]-(end)
                WHERE start.name = $start_entity
                RETURN path, length(path) as hop_count
                ORDER BY hop_count
                LIMIT 10
            """
            
            result = session.run(cypher_query, start_entity=start_entity)
            
            paths = []
            for record in result:
                path = record["path"]
                hop_count = record["hop_count"]
                
                # 解析路径
                nodes = []
                relationships = []
                
                for i, node in enumerate(path.nodes):
                    nodes.append({
                        "id": node.get("id", ""),
                        "name": node.get("name", ""),
                        "labels": list(node.labels)
                    })
                
                for rel in path.relationships:
                    relationships.append({
                        "type": rel.type,
                        "properties": dict(rel)
                    })
                
                paths.append({
                    "nodes": nodes,
                    "relationships": relationships,
                    "hop_count": hop_count
                })
            
            return paths
   
    def get_entity_neighbors(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """获取实体的邻居节点"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start {name: $entity_name})-[r*1..2]-(neighbor)
                RETURN DISTINCT neighbor.name as neighbor_name,
                       labels(neighbor) as neighbor_labels,
                       type(r[0]) as relationship_type,
                       length(r) as distance
                ORDER BY distance, neighbor_name
                LIMIT 20
            """, entity_name=entity_name)
            
            neighbors = []
            for record in result:
                neighbors.append({
                    "name": record["neighbor_name"],
                    "labels": record["neighbor_labels"],
                    "relationship_type": record["relationship_type"],
                    "distance": record["distance"]
                })
            
            return {
                "entity": entity_name,
                "neighbors": neighbors
            }