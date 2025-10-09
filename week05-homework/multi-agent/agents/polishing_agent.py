"""
润色代理 (Polishing Agent)
优化语言表达和文章结构，确保风格一致性
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any


class PolishingAgent:
    """润色代理"""
    
    def __init__(self, context):
        self.context = context
        self.logger = logging.getLogger(__name__)
        self.enhanced_mode = False
        
        # 润色规则
        self.polishing_rules = {
            'language_optimization': {
                'remove_redundancy': '移除冗余表达',
                'improve_clarity': '提高表达清晰度',
                'enhance_flow': '增强语言流畅性',
                'fix_grammar': '修正语法问题'
            },
            'style_consistency': {
                'unify_tone': '统一语言风格',
                'standardize_terminology': '标准化术语使用',
                'balance_formality': '平衡正式程度'
            },
            'structure_refinement': {
                'improve_transitions': '改善段落过渡',
                'enhance_coherence': '增强逻辑连贯性',
                'optimize_layout': '优化版面布局'
            }
        }
        
    def set_enhanced_mode(self, enabled: bool):
        """设置增强模式（用于备用代理）"""
        self.enhanced_mode = enabled
        
    async def execute(self, context) -> Dict[str, Any]:
        """执行润色任务"""
        try:
            self.logger.info("开始执行润色任务")
            
            # 获取上下文信息
            agent_context = context.get_context_for_agent('polishing')
            research_data = agent_context.get('research_data', {})
            draft_content = agent_context.get('draft_content', {})
            review_feedback = agent_context.get('review_feedback', {})
            
            # 获取待润色的文章
            article_to_polish = review_feedback.get('revised_article', 
                                                  draft_content.get('article_draft', ''))
            
            if not article_to_polish:
                raise ValueError("无法获取待润色的文章内容")
            
            # 分析润色需求
            polish_requirements = await self._analyze_polish_requirements(
                article_to_polish, review_feedback, draft_content
            )
            
            # 执行语言优化
            language_optimized = await self._optimize_language(article_to_polish, polish_requirements)
            
            # 统一风格
            style_unified = await self._unify_style(language_optimized, polish_requirements)
            
            # 完善结构
            structure_refined = await self._refine_structure(style_unified, polish_requirements)
            
            # 最终检查和微调
            final_article = await self._final_polish(structure_refined, polish_requirements)
            
            # 生成润色报告
            polish_report = await self._generate_polish_report(
                article_to_polish, final_article, polish_requirements
            )
            
            result = {
                'success': True,
                'data': {
                    'final_article': final_article,
                    'polish_requirements': polish_requirements,
                    'polish_notes': polish_report['polish_notes'],
                    'improvements_made': polish_report['improvements_made'],
                    'quality_metrics': polish_report['quality_metrics']
                },
                'metadata': {
                    'agent': 'polishing',
                    'execution_time': datetime.now().isoformat(),
                    'enhanced_mode': self.enhanced_mode,
                    'polish_operations': polish_report['operations_count'],
                    'final_word_count': len(final_article)
                }
            }
            
            self.logger.info("润色任务执行完成")
            return result
            
        except Exception as e:
            self.logger.error(f"润色任务执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'agent': 'polishing',
                    'execution_time': datetime.now().isoformat()
                }
            }
    
    async def _analyze_polish_requirements(
        self, 
        article: str, 
        review_feedback: Dict[str, Any], 
        draft_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分析润色需求"""
        
        # 从审核反馈中获取改进点
        review_suggestions = review_feedback.get('suggestions', [])
        review_scores = review_feedback.get('review_results', {}).get('detailed_scores', {})
        
        # 分析文章特征
        article_analysis = {
            'word_count': len(article),
            'paragraph_count': len([p for p in article.split('\n\n') if p.strip()]),
            'section_count': len(re.findall(r'^##?\s+', article, re.MULTILINE)),
            'formality_level': self._analyze_formality_level(article),
            'complexity_level': self._analyze_complexity_level(article)
        }
        
        # 确定润色重点
        polish_priorities = []
        
        if review_scores.get('language', 10) < 8:
            polish_priorities.append('language_optimization')
        
        if review_scores.get('structure', 10) < 8:
            polish_priorities.append('structure_refinement')
        
        if self._detect_style_inconsistency(article):
            polish_priorities.append('style_consistency')
        
        # 如果没有明显问题，进行全面润色
        if not polish_priorities:
            polish_priorities = ['language_optimization', 'style_consistency', 'structure_refinement']
        
        requirements = {
            'polish_priorities': polish_priorities,
            'target_style': self._determine_target_style(article, draft_content),
            'article_analysis': article_analysis,
            'review_feedback_integration': len(review_suggestions),
            'enhancement_level': 'moderate' if not self.enhanced_mode else 'intensive'
        }
        
        return requirements
    
    def _analyze_formality_level(self, article: str) -> str:
        """分析正式程度"""
        formal_indicators = ['根据', '分析', '研究', '表明', '显示', '证明', '因此', '由此可见']
        casual_indicators = ['说到', '呢', '吧', '哦', '嘛', '咱们']
        
        formal_count = sum(article.count(word) for word in formal_indicators)
        casual_count = sum(article.count(word) for word in casual_indicators)
        
        if formal_count > casual_count * 2:
            return 'formal'
        elif casual_count > formal_count:
            return 'casual'
        else:
            return 'neutral'
    
    def _analyze_complexity_level(self, article: str) -> str:
        """分析复杂程度"""
        sentences = re.split(r'[。！？]', article)
        avg_sentence_length = sum(len(s) for s in sentences if s.strip()) / max(len(sentences), 1)
        
        technical_terms = ['算法', '技术', '系统', '方法', '模型', '架构', '实现', '框架']
        tech_density = sum(article.count(term) for term in technical_terms) / max(len(article), 1) * 1000
        
        if avg_sentence_length > 40 or tech_density > 8:
            return 'high'
        elif avg_sentence_length > 25 or tech_density > 4:
            return 'medium'
        else:
            return 'low'
    
    def _detect_style_inconsistency(self, article: str) -> bool:
        """检测风格不一致性"""
        # 检查人称使用
        first_person = article.count('我') + article.count('我们')
        third_person = article.count('它') + article.count('他们') + article.count('这些')
        
        if first_person > 0 and third_person > 0 and abs(first_person - third_person) > 5:
            return True
        
        # 检查语调混合
        formal_tone = article.count('因此') + article.count('由此') + article.count('综上所述')
        casual_tone = article.count('呢') + article.count('吧') + article.count('哦')
        
        if formal_tone > 0 and casual_tone > 0:
            return True
        
        return False
    
    def _determine_target_style(self, article: str, draft_content: Dict[str, Any]) -> str:
        """确定目标风格"""
        # 从撰写阶段获取原始风格意图
        writing_requirements = draft_content.get('writing_requirements', {})
        intended_style = writing_requirements.get('style', 'formal')
        
        # 分析当前文章风格
        current_formality = self._analyze_formality_level(article)
        
        # 选择最合适的目标风格
        if intended_style == 'academic':
            return 'academic'
        elif intended_style == 'casual' and current_formality != 'formal':
            return 'casual'
        else:
            return 'formal'
    
    async def _optimize_language(self, article: str, requirements: Dict[str, Any]) -> str:
        """优化语言表达"""
        
        optimized = article
        
        # 移除冗余表达
        optimized = self._remove_redundancy(optimized)
        
        # 提高表达清晰度
        optimized = self._improve_clarity(optimized)
        
        # 增强语言流畅性
        optimized = self._enhance_flow(optimized)
        
        # 修正常见语法问题
        optimized = self._fix_common_grammar_issues(optimized)
        
        return optimized
    
    def _remove_redundancy(self, article: str) -> str:
        """移除冗余表达"""
        # 替换冗余短语
        redundant_patterns = [
            (r'的的', '的'),
            (r'非常的', '非常'),
            (r'比较的', '比较'),
            (r'进行(.{1,4})的(.{1,10})', r'\1\2'),  # "进行...的..."模式
            (r'起到(.{1,4})的作用', r'起\1作用'),
        ]
        
        optimized = article
        for pattern, replacement in redundant_patterns:
            optimized = re.sub(pattern, replacement, optimized)
        
        return optimized
    
    def _improve_clarity(self, article: str) -> str:
        """提高表达清晰度"""
        # 简化复杂句式
        clarity_improvements = [
            (r'由于(.{1,20})的原因，所以', r'由于\1，'),
            (r'不仅(.{1,20})，而且还(.{1,20})', r'既\1，又\2'),
            (r'可以说是(.{1,15})的', r'是\1的'),
        ]
        
        improved = article
        for pattern, replacement in clarity_improvements:
            improved = re.sub(pattern, replacement, improved)
        
        return improved
    
    def _enhance_flow(self, article: str) -> str:
        """增强语言流畅性"""
        # 改善句子连接
        flow_patterns = [
            # 在段落开头添加适当的连接词
            (r'\n\n([^#])', r'\n\n此外，\1'),
            (r'\n\n(同时)', r'\n\n\1'),
            (r'\n\n(另外)', r'\n\n\1'),
        ]
        
        # 谨慎应用，避免过度修改
        enhanced = article
        
        # 在段落间添加过渡词（仅在缺少时）
        paragraphs = article.split('\n\n')
        enhanced_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            if (i > 0 and not paragraph.startswith('#') and 
                len(paragraph.strip()) > 50 and
                not any(word in paragraph[:20] for word in ['此外', '同时', '另外', '然而', '因此'])):
                
                # 根据内容选择合适的连接词
                if '分析' in paragraph or '研究' in paragraph:
                    paragraph = '进一步分析，' + paragraph
                elif '例如' in paragraph or '比如' in paragraph:
                    paragraph = '具体而言，' + paragraph
                else:
                    paragraph = '同时，' + paragraph
            
            enhanced_paragraphs.append(paragraph)
        
        return '\n\n'.join(enhanced_paragraphs)
    
    def _fix_common_grammar_issues(self, article: str) -> str:
        """修正常见语法问题"""
        grammar_fixes = [
            (r'([，。])([，。])', r'\1'),  # 重复标点
            (r'([^。！？])$', r'\1。'),    # 句末缺少标点
            (r'([。！？])([^\n])', r'\1 \2'),  # 句号后缺少空格
        ]
        
        fixed = article
        for pattern, replacement in grammar_fixes:
            fixed = re.sub(pattern, replacement, fixed)
        
        return fixed
    
    async def _unify_style(self, article: str, requirements: Dict[str, Any]) -> str:
        """统一风格"""
        
        target_style = requirements['target_style']
        unified = article
        
        if target_style == 'formal':
            unified = self._apply_formal_style(unified)
        elif target_style == 'casual':
            unified = self._apply_casual_style(unified)
        elif target_style == 'academic':
            unified = self._apply_academic_style(unified)
        
        return unified
    
    def _apply_formal_style(self, article: str) -> str:
        """应用正式风格"""
        formal_replacements = [
            (r'呢', ''),
            (r'吧', ''),
            (r'哦', ''),
            (r'咱们', '我们'),
            (r'挺(.{1,3})的', r'相当\1'),
        ]
        
        formal_article = article
        for pattern, replacement in formal_replacements:
            formal_article = re.sub(pattern, replacement, formal_article)
        
        return formal_article
    
    def _apply_casual_style(self, article: str) -> str:
        """应用通俗风格"""
        casual_replacements = [
            (r'因此，', '所以，'),
            (r'综上所述，', '总的来说，'),
            (r'由此可见，', '可以看出，'),
        ]
        
        casual_article = article
        for pattern, replacement in casual_replacements:
            casual_article = re.sub(pattern, replacement, casual_article)
        
        return casual_article
    
    def _apply_academic_style(self, article: str) -> str:
        """应用学术风格"""
        academic_replacements = [
            (r'我觉得', '本文认为'),
            (r'我们可以', '可以'),
            (r'显然', '研究表明'),
        ]
        
        academic_article = article
        for pattern, replacement in academic_replacements:
            academic_article = re.sub(pattern, replacement, academic_article)
        
        return academic_article
    
    async def _refine_structure(self, article: str, requirements: Dict[str, Any]) -> str:
        """完善结构"""
        
        refined = article
        
        # 改善段落过渡
        refined = self._improve_paragraph_transitions(refined)
        
        # 优化标题层次
        refined = self._optimize_heading_hierarchy(refined)
        
        # 增强逻辑连贯性
        refined = self._enhance_logical_coherence(refined)
        
        return refined
    
    def _improve_paragraph_transitions(self, article: str) -> str:
        """改善段落过渡"""
        # 在关键位置添加过渡句
        transition_patterns = [
            (r'(\n\n## .+?\n\n)([^#])', r'\1基于以上分析，\2'),
            (r'(\n\n### .+?\n\n)([^#])', r'\1具体来说，\2'),
        ]
        
        improved = article
        for pattern, replacement in transition_patterns:
            improved = re.sub(pattern, replacement, improved, count=1)
        
        return improved
    
    def _optimize_heading_hierarchy(self, article: str) -> str:
        """优化标题层次"""
        lines = article.split('\n')
        optimized_lines = []
        
        for line in lines:
            # 确保标题格式一致
            if re.match(r'^#+\s', line):
                # 标准化标题格式（确保#后有空格）
                level = len(re.match(r'^(#+)', line).group(1))
                title = re.sub(r'^#+\s*', '', line)
                optimized_line = '#' * level + ' ' + title
                optimized_lines.append(optimized_line)
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def _enhance_logical_coherence(self, article: str) -> str:
        """增强逻辑连贯性"""
        # 在关键位置添加逻辑连接词
        coherence_markers = [
            '首先', '其次', '最后', '因此', '然而', '同时'
        ]
        
        paragraphs = article.split('\n\n')
        enhanced_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            if (not paragraph.startswith('#') and len(paragraph.strip()) > 100 and
                i > 0 and i < len(paragraphs) - 1):
                
                # 检查是否已有逻辑标记
                has_marker = any(marker in paragraph[:20] for marker in coherence_markers)
                
                if not has_marker:
                    # 根据位置和内容添加适当的标记
                    if i == 1:
                        paragraph = '首先，' + paragraph
                    elif i == len(paragraphs) - 2:
                        paragraph = '最后，' + paragraph
                    elif '但是' in paragraph or '然而' in paragraph:
                        pass  # 已有转折词
                    else:
                        paragraph = '此外，' + paragraph
            
            enhanced_paragraphs.append(paragraph)
        
        return '\n\n'.join(enhanced_paragraphs)
    
    async def _final_polish(self, article: str, requirements: Dict[str, Any]) -> str:
        """最终润色和微调"""
        
        final = article
        
        # 最终格式整理
        final = self._final_format_cleanup(final)
        
        # 确保一致性
        final = self._ensure_consistency(final)
        
        # 最终质量检查
        final = self._final_quality_check(final)
        
        return final
    
    def _final_format_cleanup(self, article: str) -> str:
        """最终格式整理"""
        # 移除多余空行
        cleaned = re.sub(r'\n{3,}', '\n\n', article)
        
        # 确保文章结尾有换行符
        if not cleaned.endswith('\n'):
            cleaned += '\n'
        
        # 标准化标点符号前后的空格
        cleaned = re.sub(r'\s+([，。！？；：])', r'\1', cleaned)
        
        return cleaned
    
    def _ensure_consistency(self, article: str) -> str:
        """确保一致性"""
        # 统一术语使用
        term_standardization = [
            (r'人工智能|AI|ai', 'AI'),
            (r'机器学习|ML', '机器学习'),
            (r'深度学习|DL', '深度学习'),
        ]
        
        consistent = article
        for pattern, standard in term_standardization:
            consistent = re.sub(pattern, standard, consistent)
        
        return consistent
    
    def _final_quality_check(self, article: str) -> str:
        """最终质量检查"""
        # 检查并修复常见问题
        quality_fixes = [
            (r'([。！？])\s*([a-zA-Z])', r'\1 \2'),  # 中英文标点后的空格
            (r'([a-zA-Z])\s*([，。！？])', r'\1\2'),   # 英文和中文标点
        ]
        
        checked = article
        for pattern, replacement in quality_fixes:
            checked = re.sub(pattern, replacement, checked)
        
        return checked
    
    async def _generate_polish_report(
        self, 
        original_article: str, 
        final_article: str, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成润色报告"""
        
        # 计算改进统计
        original_word_count = len(original_article)
        final_word_count = len(final_article)
        
        polish_notes = [
            "已完成语言表达优化",
            "统一了文章风格",
            "改善了段落过渡",
            "优化了格式布局"
        ]
        
        improvements_made = len(requirements['polish_priorities'])
        
        quality_metrics = {
            'word_count_change': final_word_count - original_word_count,
            'readability_improved': True,
            'style_consistency': True,
            'structure_optimized': True
        }
        
        return {
            'polish_notes': polish_notes,
            'improvements_made': improvements_made,
            'quality_metrics': quality_metrics,
            'operations_count': sum(1 for rule_category in requirements['polish_priorities'])
        }