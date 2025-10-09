"""
重试管理器
实现三级重试策略：
1. 一级：相同代理重新执行（最多 2 次）
2. 二级：切换至备用代理执行
3. 三级：向用户请求补充信息
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class RetryLevel(Enum):
    """重试级别"""
    LEVEL_1 = "same_agent_retry"      # 相同代理重试
    LEVEL_2 = "backup_agent"          # 备用代理
    LEVEL_3 = "user_intervention"     # 用户干预


@dataclass
class RetryAttempt:
    """重试尝试记录"""
    attempt_number: int
    retry_level: RetryLevel
    agent_name: str
    timestamp: str
    success: bool
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


class RetryManager:
    """重试管理器"""
    
    def __init__(self, max_level1_retries: int = 2):
        self.logger = logging.getLogger(__name__)
        self.max_level1_retries = max_level1_retries
        self._stage_attempts = {}  # stage_name -> List[RetryAttempt]
        
        # 备用代理配置
        self.backup_agents = {
            'research': 'advanced_research',
            'writing': 'creative_writing', 
            'review': 'senior_review',
            'polishing': 'expert_polishing'
        }
    
    async def execute_with_retry(self, agent_name: str, agent_instance: Any, context: Any) -> Dict[str, Any]:
        """
        执行代理任务，包含完整的重试逻辑
        
        Args:
            agent_name: 代理名称
            agent_instance: 代理实例
            context: MCP上下文
            
        Returns:
            代理执行结果
        """
        if agent_name not in self._stage_attempts:
            self._stage_attempts[agent_name] = []
        
        # 一级重试：相同代理重新执行
        for attempt in range(1, self.max_level1_retries + 2):  # +1 为初始尝试
            try:
                result = await self._execute_single_attempt(
                    agent_name, agent_instance, context, attempt, RetryLevel.LEVEL_1
                )
                
                if result['success']:
                    self.logger.info(f"代理 {agent_name} 在第 {attempt} 次尝试后成功")
                    return result
                    
            except Exception as e:
                self.logger.warning(f"代理 {agent_name} 第 {attempt} 次尝试失败: {e}")
                
                self._record_attempt(
                    agent_name, attempt, RetryLevel.LEVEL_1, 
                    agent_name, False, str(e)
                )
                
                if attempt <= self.max_level1_retries:
                    await asyncio.sleep(1 * attempt)  # 递增延迟
                    continue
        
        # 二级重试：切换至备用代理
        if agent_name in self.backup_agents:
            backup_name = self.backup_agents[agent_name]
            self.logger.info(f"启动二级重试，使用备用代理: {backup_name}")
            
            try:
                backup_agent = await self._create_backup_agent(backup_name, agent_instance)
                result = await self._execute_single_attempt(
                    agent_name, backup_agent, context, 
                    self.max_level1_retries + 2, RetryLevel.LEVEL_2
                )
                
                if result['success']:
                    self.logger.info(f"备用代理 {backup_name} 执行成功")
                    return result
                    
            except Exception as e:
                self.logger.error(f"备用代理 {backup_name} 执行失败: {e}")
                self._record_attempt(
                    agent_name, self.max_level1_retries + 2, RetryLevel.LEVEL_2,
                    backup_name, False, str(e)
                )
        
        # 三级重试：用户干预
        return await self._handle_user_intervention(agent_name, context)
    
    async def _execute_single_attempt(
        self, 
        agent_name: str, 
        agent_instance: Any, 
        context: Any, 
        attempt_number: int,
        retry_level: RetryLevel
    ) -> Dict[str, Any]:
        """执行单次尝试"""
        start_time = datetime.now()
        
        try:
            # 执行代理任务
            result = await agent_instance.execute(context)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self._record_attempt(
                agent_name, attempt_number, retry_level,
                agent_instance.__class__.__name__, True, None, result, execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self._record_attempt(
                agent_name, attempt_number, retry_level,
                agent_instance.__class__.__name__, False, str(e), None, execution_time
            )
            
            raise e
    
    async def _create_backup_agent(self, backup_name: str, original_agent: Any) -> Any:
        """创建备用代理实例"""
        # 这里可以根据需要实现不同的备用代理
        # 目前简化为使用原代理但调整参数
        backup_agent = original_agent.__class__(original_agent.context)
        
        # 为备用代理设置增强配置
        if hasattr(backup_agent, 'set_enhanced_mode'):
            backup_agent.set_enhanced_mode(True)
        
        return backup_agent
    
    async def _handle_user_intervention(self, agent_name: str, context: Any) -> Dict[str, Any]:
        """处理用户干预（三级重试）"""
        self.logger.warning(f"代理 {agent_name} 所有自动重试均失败，需要用户干预")
        
        # 收集错误信息
        attempts = self._stage_attempts.get(agent_name, [])
        error_summary = self._generate_error_summary(attempts)
        
        # 模拟用户干预（实际实现中可以通过UI或命令行交互）
        intervention_result = await self._simulate_user_intervention(
            agent_name, error_summary, context
        )
        
        # 记录用户干预尝试
        attempt_number = len(attempts) + 1
        self._record_attempt(
            agent_name, attempt_number, RetryLevel.LEVEL_3,
            "user_intervention", intervention_result['success'], 
            intervention_result.get('error'), intervention_result
        )
        
        return intervention_result
    
    async def _simulate_user_intervention(
        self, 
        agent_name: str, 
        error_summary: str, 
        context: Any
    ) -> Dict[str, Any]:
        """模拟用户干预（实际项目中需要真实的用户交互）"""
        
        # 基于错误类型提供默认的恢复策略
        recovery_strategies = {
            'research': self._research_recovery_strategy,
            'writing': self._writing_recovery_strategy,
            'review': self._review_recovery_strategy,
            'polishing': self._polishing_recovery_strategy
        }
        
        if agent_name in recovery_strategies:
            try:
                result = await recovery_strategies[agent_name](context, error_summary)
                return {
                    'success': True,
                    'data': result,
                    'metadata': {
                        'recovery_strategy': f"{agent_name}_fallback",
                        'intervention_type': 'automatic_fallback'
                    }
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f"恢复策略执行失败: {e}",
                    'metadata': {
                        'intervention_type': 'fallback_failed'
                    }
                }
        
        # 默认失败结果
        return {
            'success': False,
            'error': f"代理 {agent_name} 无法恢复，缺少用户干预",
            'metadata': {
                'intervention_type': 'user_required',
                'error_summary': error_summary
            }
        }
    
    async def _research_recovery_strategy(self, context: Any, error_summary: str) -> Dict[str, Any]:
        """研究代理恢复策略"""
        user_prompt = context.get_user_prompt()
        
        # 提供基础的研究结果
        basic_research = {
            'topic': user_prompt,
            'key_points': [
                f"关于 '{user_prompt}' 的基本概念",
                f"'{user_prompt}' 的主要特点",
                f"'{user_prompt}' 的应用场景",
                f"'{user_prompt}' 的发展趋势"
            ],
            'sources': ["内部知识库", "基础资料库"],
            'research_notes': f"由于网络搜索失败，提供基于内部知识的基础研究结果。错误摘要: {error_summary}"
        }
        
        return basic_research
    
    async def _writing_recovery_strategy(self, context: Any, error_summary: str) -> Dict[str, Any]:
        """撰写代理恢复策略"""
        user_prompt = context.get_user_prompt()
        research_data = context.get_research_data()
        
        # 生成基础文章结构
        basic_article = f"""# {user_prompt}

## 引言
本文将探讨{user_prompt}相关的内容。

## 主要内容
{research_data.get('key_points', ['相关要点待补充'])}

## 结论
通过以上分析，我们可以看出{user_prompt}的重要性。

*注：本文为系统恢复模式生成，可能需要进一步完善。*
"""
        
        return {
            'article_draft': basic_article,
            'word_count': len(basic_article),
            'structure': ['引言', '主要内容', '结论'],
            'recovery_note': f"使用基础模板生成。原错误: {error_summary}"
        }
    
    async def _review_recovery_strategy(self, context: Any, error_summary: str) -> Dict[str, Any]:
        """审核代理恢复策略"""
        draft_content = context.get_writing_data()
        article_text = draft_content.get('article_draft', '')
        
        # 提供基本的审核意见
        basic_review = {
            'overall_score': 7,
            'suggestions': [
                "文章结构清晰，逻辑基本合理",
                "建议增加更多具体例子",
                "可以进一步完善结论部分"
            ],
            'revised_article': article_text,  # 不做修改，直接通过
            'review_notes': f"基础审核通过。原错误: {error_summary}"
        }
        
        return basic_review
    
    async def _polishing_recovery_strategy(self, context: Any, error_summary: str) -> Dict[str, Any]:
        """润色代理恢复策略"""
        review_data = context.get_review_data()
        article_text = review_data.get('revised_article', '')
        
        # 基本润色（主要是格式整理）
        polished_text = article_text.strip()
        if not polished_text.endswith('\n'):
            polished_text += '\n'
        
        return {
            'final_article': polished_text,
            'polish_notes': [
                "已完成基本格式整理",
                "保持原文内容不变"
            ],
            'recovery_note': f"基础润色完成。原错误: {error_summary}"
        }
    
    def _record_attempt(
        self, 
        stage_name: str, 
        attempt_number: int, 
        retry_level: RetryLevel,
        agent_name: str, 
        success: bool, 
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ):
        """记录重试尝试"""
        attempt = RetryAttempt(
            attempt_number=attempt_number,
            retry_level=retry_level,
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
            success=success,
            error=error,
            result=result,
            execution_time=execution_time
        )
        
        if stage_name not in self._stage_attempts:
            self._stage_attempts[stage_name] = []
        
        self._stage_attempts[stage_name].append(attempt)
    
    def _generate_error_summary(self, attempts: List[RetryAttempt]) -> str:
        """生成错误摘要"""
        if not attempts:
            return "无错误记录"
        
        error_counts = {}
        for attempt in attempts:
            if attempt.error:
                error_type = attempt.error.split(':')[0] if ':' in attempt.error else attempt.error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        summary_parts = []
        for error_type, count in error_counts.items():
            summary_parts.append(f"{error_type}({count}次)")
        
        return f"失败尝试 {len(attempts)} 次，主要错误: {', '.join(summary_parts)}"
    
    def get_stage_attempts(self, stage_name: str) -> List[Dict[str, Any]]:
        """获取阶段的所有尝试记录"""
        attempts = self._stage_attempts.get(stage_name, [])
        return [
            {
                'attempt_number': attempt.attempt_number,
                'retry_level': attempt.retry_level.value,
                'agent_name': attempt.agent_name,
                'timestamp': attempt.timestamp,
                'success': attempt.success,
                'error': attempt.error,
                'execution_time': attempt.execution_time
            }
            for attempt in attempts
        ]
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """获取重试统计信息"""
        stats = {
            'total_stages': len(self._stage_attempts),
            'total_attempts': sum(len(attempts) for attempts in self._stage_attempts.values()),
            'successful_stages': 0,
            'failed_stages': 0,
            'retry_levels_used': set(),
            'average_attempts_per_stage': 0
        }
        
        for stage_name, attempts in self._stage_attempts.items():
            if any(attempt.success for attempt in attempts):
                stats['successful_stages'] += 1
            else:
                stats['failed_stages'] += 1
            
            for attempt in attempts:
                stats['retry_levels_used'].add(attempt.retry_level.value)
        
        if stats['total_stages'] > 0:
            stats['average_attempts_per_stage'] = stats['total_attempts'] / stats['total_stages']
        
        stats['retry_levels_used'] = list(stats['retry_levels_used'])
        
        return stats