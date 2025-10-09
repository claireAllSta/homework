"""
审核代理 (Review Agent)
检查内容质量和逻辑一致性，提供修改建议
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


class ReviewAgent:
    """审核代理"""
    
    def __init__(self, context):
        self.context = context
        self.logger = logging.getLogger(__name__)
        self.enhanced_mode = False
        
    def set_enhanced_mode(self, enabled: bool):
        """设置增强模式（用于备用代理）"""
        self.enhanced_mode = enabled
        
    async def execute(self, context) -> Dict[str, Any]:
        """执行审核任务"""
        try:
            self.logger.info("开始执行审核任务")
            
            # 获取上下文信息
            agent_context = context.get_context_for_agent('review')
            research_data = agent_context.get('research_data', {})
            draft_content = agent_context.get('draft_content', {})
            
            # 获取文章草稿
            article_draft = draft_content.get('article_draft', '')
            if not article_draft:
                raise ValueError("无法获取文章草稿进行审核")
            
            # 执行审核
            review_results = await self._conduct_review(article_draft, research_data)
            
            # 生成修改建议
            suggestions = await self._generate_suggestions(review_results, article_draft)
            
            # 应用修改
            revised_article = await self._apply_revisions(article_draft, suggestions)
            
            result = {
                'success': True,
                'data': {
                    'overall_score': review_results['overall_score'],
                    'review_results': review_results,
                    'suggestions': suggestions,
                    'revised_article': revised_article,
                    'improvement_summary': f"评分 {review_results['overall_score']:.1f}/10，提出 {len(suggestions)} 项改进建议"
                },
                'metadata': {
                    'agent': 'review',
                    'execution_time': datetime.now().isoformat(),
                    'enhanced_mode': self.enhanced_mode,
                    'total_issues_found': len(suggestions)
                }
            }
            
            self.logger.info("审核任务执行完成")
            return result
            
        except Exception as e:
            self.logger.error(f"审核任务执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'agent': 'review',
                    'execution_time': datetime.now().isoformat()
                }
            }
    
    async def _conduct_review(self, article: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """进行审核"""
        
        issues = []
        scores = {}
        
        # 内容完整性检查
        completeness_score, completeness_issues = self._check_completeness(article)
        scores['completeness'] = completeness_score
        issues.extend(completeness_issues)
        
        # 结构质量检查
        structure_score, structure_issues = self._check_structure(article)
        scores['structure'] = structure_score
        issues.extend(structure_issues)
        
        # 语言质量检查
        language_score, language_issues = self._check_language(article)
        scores['language'] = language_score
        issues.extend(language_issues)
        
        # 内容相关性检查
        relevance_score, relevance_issues = self._check_relevance(article, research_data)
        scores['relevance'] = relevance_score
        issues.extend(relevance_issues)
        
        overall_score = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            'overall_score': overall_score,
            'detailed_scores': scores,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def _check_completeness(self, article: str) -> Tuple[float, List[str]]:
        """检查内容完整性"""
        issues = []
        score = 8.0
        
        # 检查基本结构
        sections = re.findall(r'^##?\s+(.+)$', article, re.MULTILINE)
        if len(sections) < 3:
            issues.append("文章结构过于简单，建议增加更多章节")
            score -= 2
        
        # 检查引言和结论
        if not re.search(r'(引言|介绍|背景)', article):
            issues.append("缺少引言或背景介绍部分")
            score -= 1
        
        if not re.search(r'(结论|总结|小结)', article):
            issues.append("缺少结论或总结部分")
            score -= 1
        
        # 检查内容长度
        if len(article) < 500:
            issues.append("文章内容过短，建议增加更多详细内容")
            score -= 1
        
        return max(score, 0), issues
    
    def _check_structure(self, article: str) -> Tuple[float, List[str]]:
        """检查结构质量"""
        issues = []
        score = 8.0
        
        # 检查标题层次
        h1_count = len(re.findall(r'^#\s+', article, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+', article, re.MULTILINE))
        
        if h1_count == 0:
            issues.append("缺少主标题")
            score -= 2
        
        if h2_count < 2:
            issues.append("章节过少，建议增加更多子标题")
            score -= 1
        
        # 检查段落数量
        paragraphs = [p.strip() for p in article.split('\n\n') if p.strip() and not p.startswith('#')]
        if len(paragraphs) < 3:
            issues.append("段落过少，可能影响逻辑展开")
            score -= 1
        
        return max(score, 0), issues
    
    def _check_language(self, article: str) -> Tuple[float, List[str]]:
        """检查语言质量"""
        issues = []
        score = 8.0
        
        # 检查句子长度
        sentences = re.split(r'[。！？]', article)
        long_sentences = [s for s in sentences if len(s.strip()) > 100]
        
        if len(long_sentences) > len(sentences) * 0.3:
            issues.append("过多长句可能影响表达清晰度")
            score -= 1
        
        # 检查重复用词
        words = re.findall(r'[\u4e00-\u9fff]+', article)
        word_freq = {}
        for word in words:
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        overused_words = [word for word, freq in word_freq.items() if freq > 10]
        if overused_words:
            issues.append(f"部分词汇使用过频: {', '.join(overused_words[:3])}")
            score -= 0.5
        
        return max(score, 0), issues
    
    def _check_relevance(self, article: str, research_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """检查内容相关性"""
        issues = []
        score = 8.0
        
        # 检查是否使用了研究数据
        key_points = research_data.get('key_points', [])
        definitions = research_data.get('definitions', {})
        
        used_research_info = False
        for point in key_points:
            point_text = point if isinstance(point, str) else str(point.get('point', ''))
            if point_text and point_text in article:
                used_research_info = True
                break
        
        if not used_research_info and not any(key in article for key in definitions.keys()):
            issues.append("文章未充分利用研究数据")
            score -= 2
        
        return max(score, 0), issues
    
    async def _generate_suggestions(self, review_results: Dict[str, Any], article: str) -> List[Dict[str, Any]]:
        """生成修改建议"""
        suggestions = []
        
        for issue in review_results['issues']:
            suggestion = self._create_suggestion(issue, article)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _create_suggestion(self, issue: str, article: str) -> Optional[Dict[str, Any]]:
        """创建修改建议"""
        if '缺少引言' in issue:
            return {
                'type': 'add_intro',
                'priority': 9,
                'description': '添加引言部分',
                'action': '在文章开头添加引言章节'
            }
        elif '缺少结论' in issue:
            return {
                'type': 'add_conclusion',
                'priority': 8,
                'description': '添加结论部分',
                'action': '在文章末尾添加结论章节'
            }
        elif '内容过短' in issue:
            return {
                'type': 'expand_content',
                'priority': 7,
                'description': '扩充文章内容',
                'action': '增加具体例子和详细分析'
            }
        else:
            return {
                'type': 'general_improvement',
                'priority': 5,
                'description': '一般性改进',
                'action': f'针对"{issue}"进行改进'
            }
    
    async def _apply_revisions(self, article: str, suggestions: List[Dict[str, Any]]) -> str:
        """应用修改建议"""
        revised_article = article
        
        for suggestion in suggestions:
            if suggestion['type'] == 'add_intro':
                intro = "\n\n## 引言\n\n本文将详细探讨相关主题，为读者提供全面的分析和见解。\n\n"
                revised_article = revised_article.replace('# ', intro + '# ', 1)
            
            elif suggestion['type'] == 'add_conclusion':
                conclusion = "\n\n## 结论\n\n通过以上分析，我们可以得出重要结论，为进一步研究提供基础。\n"
                revised_article += conclusion
            
            elif suggestion['type'] == 'expand_content':
                # 在主要章节后添加扩展内容
                sections = re.split(r'(^##\s+.+$)', revised_article, flags=re.MULTILINE)
                expanded_sections = []
                
                for i, section in enumerate(sections):
                    expanded_sections.append(section)
                    if re.match(r'^##\s+', section) and i < len(sections) - 1:
                        expanded_sections.append("\n\n这个方面需要进一步的深入分析和详细探讨。")
                
                revised_article = ''.join(expanded_sections)
        
        return revised_article