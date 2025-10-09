"""
撰写代理 (Writing Agent)
基于研究结果生成文章初稿，支持调整文章风格和长度
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any


class WritingAgent:
    """撰写代理"""
    
    def __init__(self, context):
        self.context = context
        self.logger = logging.getLogger(__name__)
        self.enhanced_mode = False
        
        # 文章模板
        self.templates = {
            'formal': {
                'structure': ['引言', '背景介绍', '主要内容', '案例分析', '结论'],
                'style': '正式严谨',
                'tone': '客观专业'
            },
            'casual': {
                'structure': ['开场', '核心要点', '实际应用', '个人见解', '总结'],
                'style': '轻松易懂',
                'tone': '友好亲切'
            },
            'academic': {
                'structure': ['摘要', '引言', '文献综述', '研究内容', '分析讨论', '结论与展望'],
                'style': '学术规范',
                'tone': '严谨客观'
            }
        }
        
    def set_enhanced_mode(self, enabled: bool):
        """设置增强模式（用于备用代理）"""
        self.enhanced_mode = enabled
        
    async def execute(self, context) -> Dict[str, Any]:
        """执行撰写任务"""
        try:
            self.logger.info("开始执行撰写任务")
            
            # 获取上下文信息
            agent_context = context.get_context_for_agent('writing')
            research_data = agent_context.get('research_data', {})
            user_prompt = agent_context['user_prompt']
            
            # 分析写作要求
            writing_requirements = await self._analyze_writing_requirements(user_prompt, research_data)
            
            # 生成文章大纲
            outline = await self._generate_outline(writing_requirements, research_data)
            
            # 撰写文章内容
            article_content = await self._write_article(outline, research_data, writing_requirements)
            
            # 生成元数据
            metadata = await self._generate_writing_metadata(article_content, writing_requirements)
            
            result = {
                'success': True,
                'data': {
                    'article_draft': article_content,
                    'outline': outline,
                    'writing_requirements': writing_requirements,
                    'word_count': len(article_content),
                    'structure': outline['sections'],
                    'style_info': metadata['style_analysis']
                },
                'metadata': {
                    'agent': 'writing',
                    'execution_time': datetime.now().isoformat(),
                    'enhanced_mode': self.enhanced_mode,
                    'template_used': writing_requirements['style'],
                    'writing_stats': metadata['statistics']
                }
            }
            
            self.logger.info("撰写任务执行完成")
            return result
            
        except Exception as e:
            self.logger.error(f"撰写任务执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'agent': 'writing',
                    'execution_time': datetime.now().isoformat()
                }
            }
    
    async def _analyze_writing_requirements(self, user_prompt: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析写作要求"""
        
        # 从用户提示中推断要求
        prompt_lower = user_prompt.lower()
        
        # 确定文章风格
        style = 'formal'  # 默认
        if any(word in prompt_lower for word in ['学术', '论文', 'academic', 'research']):
            style = 'academic'
        elif any(word in prompt_lower for word in ['通俗', '简单', 'casual', 'simple', '易懂']):
            style = 'casual'
        
        # 确定文章长度
        length = 'medium'  # 默认
        if any(word in prompt_lower for word in ['简短', '简单', 'brief', 'short']):
            length = 'short'
        elif any(word in prompt_lower for word in ['详细', '深入', 'detailed', 'comprehensive', 'long']):
            length = 'long'
        
        # 确定是否需要示例
        include_examples = any(word in prompt_lower for word in ['例子', '示例', '案例', 'example', 'case'])
        
        # 确定目标受众
        audience = 'general'
        if any(word in prompt_lower for word in ['专业', '技术', 'technical', 'professional']):
            audience = 'professional'
        elif any(word in prompt_lower for word in ['初学者', '新手', 'beginner', 'novice']):
            audience = 'beginner'
        
        requirements = {
            'style': style,
            'length': length,
            'include_examples': include_examples,
            'audience': audience,
            'topic': research_data.get('topic', user_prompt),
            'language': 'chinese',
            'structure_preference': self.templates[style]['structure']
        }
        
        self.logger.debug(f"写作要求分析结果: {requirements}")
        return requirements
    
    async def _generate_outline(self, requirements: Dict[str, Any], research_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成文章大纲"""
        
        topic = requirements['topic']
        style = requirements['style']
        template = self.templates[style]
        
        # 基于模板生成大纲结构
        sections = []
        
        for section_name in template['structure']:
            section = {
                'title': section_name,
                'content_points': [],
                'estimated_length': 0
            }
            
            # 根据研究数据为每个部分分配内容点
            if section_name in ['引言', '开场', '摘要']:
                section['content_points'] = [
                    f"介绍{topic}的基本概念",
                    f"说明{topic}的重要性",
                    "概述文章结构"
                ]
                section['estimated_length'] = 200
                
            elif section_name in ['背景介绍', '文献综述']:
                section['content_points'] = [
                    f"{topic}的发展历程",
                    f"{topic}的理论基础",
                    "相关研究现状"
                ]
                section['estimated_length'] = 300
                
            elif section_name in ['主要内容', '核心要点', '研究内容']:
                key_points = research_data.get('key_points', [])
                section['content_points'] = [
                    point.get('point', point) if isinstance(point, dict) else str(point)
                    for point in key_points[:5]
                ]
                if not section['content_points']:
                    section['content_points'] = [f"{topic}的核心特征", f"{topic}的主要优势"]
                section['estimated_length'] = 500
                
            elif section_name in ['案例分析', '实际应用']:
                applications = research_data.get('applications', [])
                section['content_points'] = applications[:3] if applications else [f"{topic}在实际中的应用"]
                section['estimated_length'] = 300
                
            elif section_name in ['分析讨论']:
                section['content_points'] = [
                    f"{topic}的优势分析",
                    f"{topic}的挑战与限制",
                    f"{topic}的发展前景"
                ]
                section['estimated_length'] = 400
                
            elif section_name in ['结论', '总结', '结论与展望']:
                section['content_points'] = [
                    f"总结{topic}的重要价值",
                    f"{topic}的未来发展趋势",
                    "对读者的建议"
                ]
                section['estimated_length'] = 200
                
            sections.append(section)
        
        # 根据长度要求调整
        total_estimated_length = sum(section['estimated_length'] for section in sections)
        length_multiplier = self._get_length_multiplier(requirements['length'])
        
        for section in sections:
            section['estimated_length'] = int(section['estimated_length'] * length_multiplier)
        
        outline = {
            'title': f"关于{topic}的{self._get_article_type_name(style)}",
            'sections': [section['title'] for section in sections],
            'detailed_sections': sections,
            'total_estimated_length': int(total_estimated_length * length_multiplier),
            'style': style,
            'structure_rationale': f"采用{template['style']}风格，{template['tone']}基调"
        }
        
        return outline
    
    def _get_length_multiplier(self, length: str) -> float:
        """获取长度调整系数"""
        multipliers = {
            'short': 0.6,
            'medium': 1.0,
            'long': 1.5
        }
        return multipliers.get(length, 1.0)
    
    def _get_article_type_name(self, style: str) -> str:
        """获取文章类型名称"""
        type_names = {
            'formal': '分析文章',
            'casual': '介绍文章',
            'academic': '学术论文'
        }
        return type_names.get(style, '文章')
    
    async def _write_article(self, outline: Dict[str, Any], research_data: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """撰写文章内容"""
        
        article_parts = []
        
        # 添加标题
        article_parts.append(f"# {outline['title']}\n")
        
        # 逐节撰写内容
        for section_detail in outline['detailed_sections']:
            section_content = await self._write_section(
                section_detail, research_data, requirements
            )
            article_parts.append(section_content)
        
        # 添加参考资料（如果有）
        if research_data.get('sources'):
            references = self._generate_references(research_data['sources'])
            article_parts.append(references)
        
        return '\n\n'.join(article_parts)
    
    async def _write_section(self, section_detail: Dict[str, Any], research_data: Dict[str, Any], requirements: Dict[str, Any]) -> str:
        """撰写单个章节"""
        
        section_title = section_detail['title']
        content_points = section_detail['content_points']
        style = requirements['style']
        
        section_parts = [f"## {section_title}"]
        
        # 根据内容点生成段落
        for point in content_points:
            paragraph = await self._write_paragraph(point, research_data, style)
            section_parts.append(paragraph)
        
        # 如果需要示例且适合的章节
        if (requirements.get('include_examples') and 
            section_title in ['主要内容', '案例分析', '实际应用', '核心要点']):
            example = self._generate_example(research_data, requirements['topic'])
            if example:
                section_parts.append(f"**示例：**\n{example}")
        
        return '\n\n'.join(section_parts)
    
    async def _write_paragraph(self, content_point: str, research_data: Dict[str, Any], style: str) -> str:
        """撰写段落内容"""
        
        # 从研究数据中查找相关信息
        relevant_info = self._find_relevant_info(content_point, research_data)
        
        # 根据风格生成段落
        if style == 'academic':
            paragraph = self._write_academic_paragraph(content_point, relevant_info)
        elif style == 'casual':
            paragraph = self._write_casual_paragraph(content_point, relevant_info)
        else:  # formal
            paragraph = self._write_formal_paragraph(content_point, relevant_info)
        
        return paragraph
    
    def _find_relevant_info(self, content_point: str, research_data: Dict[str, Any]) -> List[str]:
        """从研究数据中查找相关信息"""
        relevant_info = []
        
        # 检查关键点
        key_points = research_data.get('key_points', [])
        for point in key_points:
            if isinstance(point, dict):
                if any(word in point.get('point', '').lower() for word in content_point.lower().split()):
                    relevant_info.append(point.get('content', point.get('point', '')))
            elif isinstance(point, str):
                if any(word in point.lower() for word in content_point.lower().split()):
                    relevant_info.append(point)
        
        # 检查定义
        definitions = research_data.get('definitions', {})
        for key, definition in definitions.items():
            if any(word in content_point.lower() for word in key.lower().split()):
                relevant_info.append(definition)
        
        # 检查应用场景
        applications = research_data.get('applications', [])
        if '应用' in content_point or '案例' in content_point:
            relevant_info.extend(applications[:2])
        
        return relevant_info[:3]  # 限制相关信息数量
    
    def _write_academic_paragraph(self, content_point: str, relevant_info: List[str]) -> str:
        """撰写学术风格段落"""
        if relevant_info:
            base_content = relevant_info[0]
            paragraph = f"{content_point}。根据研究分析，{base_content}。"
            
            if len(relevant_info) > 1:
                paragraph += f"进一步的研究表明，{relevant_info[1]}。"
                
            paragraph += "这一发现为相关理论提供了重要的实证支持。"
        else:
            paragraph = f"{content_point}。这一方面需要进行深入的理论分析和实证研究。"
        
        return paragraph
    
    def _write_casual_paragraph(self, content_point: str, relevant_info: List[str]) -> str:
        """撰写通俗风格段落"""
        if relevant_info:
            base_content = relevant_info[0]
            paragraph = f"说到{content_point}，{base_content}。"
            
            if len(relevant_info) > 1:
                paragraph += f"另外，{relevant_info[1]}。"
                
            paragraph += "这些都是很有意思的发现呢！"
        else:
            paragraph = f"关于{content_point}，这确实是一个值得我们关注的重要方面。"
        
        return paragraph
    
    def _write_formal_paragraph(self, content_point: str, relevant_info: List[str]) -> str:
        """撰写正式风格段落"""
        if relevant_info:
            base_content = relevant_info[0]
            paragraph = f"在{content_point}方面，{base_content}。"
            
            if len(relevant_info) > 1:
                paragraph += f"同时，{relevant_info[1]}。"
                
            paragraph += "这些特点体现了其重要的价值和意义。"
        else:
            paragraph = f"{content_point}是一个重要的考量因素，需要给予充分的重视和分析。"
        
        return paragraph
    
    def _generate_example(self, research_data: Dict[str, Any], topic: str) -> Optional[str]:
        """生成示例内容"""
        applications = research_data.get('applications', [])
        
        if applications:
            example_app = applications[0]
            return f"以{example_app}为例，{topic}在这个领域发挥了重要作用，展现出了其强大的实用价值。"
        
        # 通用示例
        return f"例如，在实际应用中，{topic}可以帮助解决具体的问题，提高工作效率和质量。"
    
    def _generate_references(self, sources: List[Dict[str, Any]]) -> str:
        """生成参考文献"""
        if not sources:
            return ""
        
        ref_parts = ["## 参考资料"]
        
        for i, source in enumerate(sources[:5], 1):
            if isinstance(source, dict):
                title = source.get('title', '未知标题')
                url = source.get('url', '')
                ref_line = f"{i}. {title}"
                if url:
                    ref_line += f" - {url}"
            else:
                ref_line = f"{i}. {source}"
            
            ref_parts.append(ref_line)
        
        return '\n'.join(ref_parts)
    
    async def _generate_writing_metadata(self, article_content: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """生成写作元数据"""
        
        # 统计信息
        word_count = len(article_content)
        paragraph_count = len([p for p in article_content.split('\n\n') if p.strip()])
        section_count = len(re.findall(r'^## ', article_content, re.MULTILINE))
        
        # 风格分析
        style_analysis = {
            'formality_level': self._analyze_formality(article_content),
            'readability_level': self._analyze_readability(article_content),
            'technical_depth': self._analyze_technical_depth(article_content)
        }
        
        metadata = {
            'statistics': {
                'word_count': word_count,
                'paragraph_count': paragraph_count,
                'section_count': section_count,
                'average_paragraph_length': word_count // max(paragraph_count, 1)
            },
            'style_analysis': style_analysis,
            'quality_metrics': {
                'structure_completeness': section_count >= 3,
                'content_richness': word_count >= 500,
                'style_consistency': True  # 简化评估
            }
        }
        
        return metadata
    
    def _analyze_formality(self, content: str) -> str:
        """分析正式程度"""
        formal_indicators = ['根据', '分析', '研究', '表明', '显示', '证明']
        casual_indicators = ['说到', '呢', '吧', '哦', '嘛']
        
        formal_count = sum(content.count(word) for word in formal_indicators)
        casual_count = sum(content.count(word) for word in casual_indicators)
        
        if formal_count > casual_count * 2:
            return 'formal'
        elif casual_count > formal_count:
            return 'casual'
        else:
            return 'neutral'
    
    def _analyze_readability(self, content: str) -> str:
        """分析可读性"""
        sentences = re.split(r'[。！？]', content)
        avg_sentence_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length < 20:
            return 'easy'
        elif avg_sentence_length < 40:
            return 'medium'
        else:
            return 'difficult'
    
    def _analyze_technical_depth(self, content: str) -> str:
        """分析技术深度"""
        technical_terms = ['算法', '技术', '系统', '方法', '模型', '架构', '实现']
        term_count = sum(content.count(term) for term in technical_terms)
        
        content_length = len(content)
        term_density = term_count / max(content_length, 1) * 1000  # 每千字术语数量
        
        if term_density > 10:
            return 'high'
        elif term_density > 5:
            return 'medium'
        else:
            return 'low'