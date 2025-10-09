"""
研究代理 (Research Agent)
负责收集和整理相关信息，为后续写作提供基础材料
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


class ResearchAgent:
    """研究代理"""
    
    def __init__(self, context):
        self.context = context
        self.logger = logging.getLogger(__name__)
        self.enhanced_mode = False
        
    def set_enhanced_mode(self, enabled: bool):
        """设置增强模式（用于备用代理）"""
        self.enhanced_mode = enabled
        
    async def execute(self, context) -> Dict[str, Any]:
        """执行研究任务"""
        try:
            self.logger.info("开始执行研究任务")
            
            # 获取上下文信息
            agent_context = context.get_context_for_agent('research')
            user_prompt = agent_context['user_prompt']
            requirements = agent_context.get('requirements', {})
            
            # 分析研究主题
            research_plan = await self._analyze_research_topic(user_prompt, requirements)
            
            # 执行信息收集
            research_results = await self._collect_information(research_plan)
            
            # 整理和结构化数据
            structured_data = await self._structure_research_data(research_results, requirements)
            
            result = {
                'success': True,
                'data': structured_data,
                'metadata': {
                    'agent': 'research',
                    'execution_time': datetime.now().isoformat(),
                    'research_plan': research_plan,
                    'enhanced_mode': self.enhanced_mode
                }
            }
            
            self.logger.info("研究任务执行完成")
            return result
            
        except Exception as e:
            self.logger.error(f"研究任务执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'agent': 'research',
                    'execution_time': datetime.now().isoformat()
                }
            }
    
    async def _analyze_research_topic(self, user_prompt: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """分析研究主题，制定研究计划"""
        
        # 提取关键词
        keywords = self._extract_keywords(user_prompt)
        
        # 确定研究范围
        research_scope = self._determine_research_scope(user_prompt, requirements)
        
        # 制定研究计划
        plan = {
            'main_topic': user_prompt,
            'keywords': keywords,
            'research_scope': research_scope,
            'search_queries': self._generate_search_queries(keywords, research_scope),
            'information_categories': self._define_information_categories(user_prompt),
            'depth_level': requirements.get('length', 'medium')
        }
        
        self.logger.debug(f"研究计划: {plan}")
        return plan
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """从用户提示中提取关键词"""
        # 移除常见停用词
        stop_words = {'关于', '的', '一篇', '文章', '帮我', '写', '介绍', '分析', '探讨', '研究'}
        
        # 简单分词（实际应用中可以使用更复杂的NLP工具）
        words = re.findall(r'[\w]+', prompt)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 保留最重要的关键词
        return keywords[:5]
    
    def _determine_research_scope(self, prompt: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """确定研究范围"""
        scope = {
            'technical_depth': 'medium',
            'historical_context': False,
            'current_trends': True,
            'practical_applications': True,
            'theoretical_background': False
        }
        
        # 根据提示调整范围
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['技术', '原理', '算法', 'technical', 'algorithm']):
            scope['technical_depth'] = 'high'
            scope['theoretical_background'] = True
            
        if any(word in prompt_lower for word in ['历史', '发展', 'history', 'evolution']):
            scope['historical_context'] = True
            
        if requirements.get('style') == 'academic':
            scope['theoretical_background'] = True
            scope['technical_depth'] = 'high'
            
        return scope
    
    def _generate_search_queries(self, keywords: List[str], scope: Dict[str, Any]) -> List[str]:
        """生成搜索查询"""
        queries = []
        
        # 基础查询
        main_query = ' '.join(keywords)
        queries.append(main_query)
        
        # 扩展查询
        for keyword in keywords[:3]:  # 限制查询数量
            if scope['technical_depth'] == 'high':
                queries.append(f"{keyword} 技术原理")
                queries.append(f"{keyword} 实现方法")
                
            if scope['current_trends']:
                queries.append(f"{keyword} 最新发展")
                queries.append(f"{keyword} 趋势分析")
                
            if scope['practical_applications']:
                queries.append(f"{keyword} 应用场景")
                queries.append(f"{keyword} 实际案例")
        
        return queries[:8]  # 限制查询总数
    
    def _define_information_categories(self, prompt: str) -> List[str]:
        """定义信息分类"""
        categories = ['概念定义', '主要特点', '应用场景', '发展趋势']
        
        # 根据主题添加特定分类
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['技术', 'ai', '人工智能', 'algorithm']):
            categories.extend(['技术原理', '算法实现', '性能评估'])
            
        if any(word in prompt_lower for word in ['商业', '市场', 'business', 'market']):
            categories.extend(['市场分析', '商业模式', '竞争格局'])
            
        if any(word in prompt_lower for word in ['教育', '学习', 'education', 'learning']):
            categories.extend(['教学方法', '学习资源', '能力培养'])
        
        return categories
    
    async def _collect_information(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """收集信息（模拟搜索和数据收集）"""
        
        # 模拟网络搜索结果
        search_results = await self._simulate_web_search(research_plan['search_queries'])
        
        # 模拟知识库查询
        knowledge_base_results = await self._simulate_knowledge_base_query(research_plan['keywords'])
        
        # 整合结果
        collected_info = {
            'web_search_results': search_results,
            'knowledge_base_results': knowledge_base_results,
            'collection_timestamp': datetime.now().isoformat(),
            'total_sources': len(search_results) + len(knowledge_base_results)
        }
        
        return collected_info
    
    async def _simulate_web_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """模拟网络搜索（实际实现中应该调用真实的搜索API）"""
        
        # 模拟搜索延迟
        await asyncio.sleep(0.1)
        
        search_results = []
        
        for i, query in enumerate(queries[:5]):  # 限制搜索数量
            # 基于查询生成模拟结果
            result = {
                'query': query,
                'title': f"关于{query}的详细分析",
                'content': f"这是关于{query}的详细内容。包含了相关的概念、应用和发展趋势等信息。",
                'source': f"权威网站{i+1}",
                'url': f"https://example.com/article{i+1}",
                'relevance_score': max(0.6, 1.0 - i * 0.1),
                'publish_date': "2024-01-01"
            }
            search_results.append(result)
        
        return search_results
    
    async def _simulate_knowledge_base_query(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """模拟知识库查询"""
        
        await asyncio.sleep(0.05)
        
        kb_results = []
        
        for keyword in keywords[:3]:
            result = {
                'keyword': keyword,
                'definition': f"{keyword}是一个重要的概念，具有广泛的应用价值。",
                'related_concepts': [f"{keyword}相关概念1", f"{keyword}相关概念2"],
                'applications': [f"{keyword}应用场景1", f"{keyword}应用场景2"],
                'source': "内部知识库",
                'confidence': 0.9
            }
            kb_results.append(result)
        
        return kb_results
    
    async def _structure_research_data(self, research_results: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """结构化研究数据"""
        
        web_results = research_results['web_search_results']
        kb_results = research_results['knowledge_base_results']
        
        # 提取关键信息
        key_points = []
        sources = []
        definitions = {}
        applications = []
        
        # 处理网络搜索结果
        for result in web_results:
            key_points.append({
                'point': result['title'],
                'content': result['content'],
                'source': result['source'],
                'relevance': result['relevance_score']
            })
            sources.append({
                'title': result['title'],
                'url': result['url'],
                'source': result['source']
            })
        
        # 处理知识库结果
        for result in kb_results:
            definitions[result['keyword']] = result['definition']
            applications.extend(result['applications'])
        
        # 生成研究摘要
        research_summary = self._generate_research_summary(key_points, definitions, requirements)
        
        structured_data = {
            'topic': requirements.get('topic', '未指定主题'),
            'research_summary': research_summary,
            'key_points': key_points,
            'definitions': definitions,
            'applications': applications,
            'sources': sources,
            'research_notes': f"基于 {len(sources)} 个来源的研究结果",
            'confidence_level': self._calculate_confidence_level(web_results, kb_results),
            'research_depth': requirements.get('length', 'medium'),
            'structure_version': '1.0'
        }
        
        return structured_data
    
    def _generate_research_summary(self, key_points: List[Dict], definitions: Dict[str, str], requirements: Dict[str, Any]) -> str:
        """生成研究摘要"""
        
        topic = requirements.get('topic', '该主题')
        
        summary_parts = []
        
        # 添加定义部分
        if definitions:
            main_concepts = list(definitions.keys())[:2]
            summary_parts.append(f"{topic}涉及{', '.join(main_concepts)}等核心概念。")
        
        # 添加关键点
        if key_points:
            high_relevance_points = [p for p in key_points if p.get('relevance', 0) > 0.7]
            if high_relevance_points:
                summary_parts.append(f"研究发现，{topic}具有以下重要特点：")
                for point in high_relevance_points[:3]:
                    summary_parts.append(f"- {point['point']}")
        
        # 添加结论
        summary_parts.append(f"综合分析表明，{topic}是一个值得深入探讨的重要话题。")
        
        return '\n'.join(summary_parts)
    
    def _calculate_confidence_level(self, web_results: List[Dict], kb_results: List[Dict]) -> float:
        """计算研究结果的置信度"""
        
        total_sources = len(web_results) + len(kb_results)
        if total_sources == 0:
            return 0.0
        
        # 基于来源数量和质量计算置信度
        web_confidence = sum(r.get('relevance_score', 0.5) for r in web_results) / max(len(web_results), 1)
        kb_confidence = sum(r.get('confidence', 0.8) for r in kb_results) / max(len(kb_results), 1)
        
        # 综合置信度
        overall_confidence = (web_confidence * len(web_results) + kb_confidence * len(kb_results)) / total_sources
        
        # 根据来源数量调整
        source_bonus = min(0.2, total_sources * 0.05)
        
        return min(1.0, overall_confidence + source_bonus)