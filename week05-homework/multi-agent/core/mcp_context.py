"""
MCP (Model Context Protocol) 上下文管理器
负责代理间的数据传递和状态管理
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class StageResult:
    """代理执行阶段的结果"""
    agent_name: str
    timestamp: str
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class MCPContext:
    """MCP 协议上下文管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._user_prompt = ""
        self._stage_results = {}
        self._shared_data = {}
        self._conversation_history = []
        self._current_stage = None
        
    def set_user_prompt(self, prompt: str):
        """设置用户原始请求"""
        self._user_prompt = prompt
        self._conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'user_input',
            'content': prompt
        })
        self.logger.info(f"设置用户提示: {prompt}")
    
    def get_user_prompt(self) -> str:
        """获取用户原始请求"""
        return self._user_prompt
    
    def update_stage_result(self, agent_name: str, result: Dict[str, Any]):
        """更新代理阶段执行结果"""
        stage_result = StageResult(
            agent_name=agent_name,
            timestamp=datetime.now().isoformat(),
            success=result.get('success', False),
            data=result.get('data', {}),
            metadata=result.get('metadata', {}),
            error=result.get('error')
        )
        
        self._stage_results[agent_name] = stage_result
        self._current_stage = agent_name
        
        # 记录到对话历史
        self._conversation_history.append({
            'timestamp': stage_result.timestamp,
            'type': 'agent_result',
            'agent': agent_name,
            'content': result
        })
        
        self.logger.info(f"更新阶段结果: {agent_name}")
    
    def get_stage_result(self, agent_name: str) -> Optional[StageResult]:
        """获取指定代理的执行结果"""
        return self._stage_results.get(agent_name)
    
    def get_previous_results(self) -> Dict[str, StageResult]:
        """获取所有已完成阶段的结果"""
        return self._stage_results.copy()
    
    def get_research_data(self) -> Dict[str, Any]:
        """获取研究代理的数据"""
        research_result = self.get_stage_result('research')
        if research_result and research_result.success:
            return research_result.data
        return {}
    
    def get_writing_data(self) -> Dict[str, Any]:
        """获取撰写代理的数据"""
        writing_result = self.get_stage_result('writing')
        if writing_result and writing_result.success:
            return writing_result.data
        return {}
    
    def get_review_data(self) -> Dict[str, Any]:
        """获取审核代理的数据"""
        review_result = self.get_stage_result('review')
        if review_result and review_result.success:
            return review_result.data
        return {}
    
    def set_shared_data(self, key: str, value: Any):
        """设置共享数据"""
        self._shared_data[key] = value
        self.logger.debug(f"设置共享数据: {key}")
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """获取共享数据"""
        return self._shared_data.get(key, default)
    
    def get_context_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """为特定代理获取相关的上下文信息"""
        context = {
            'user_prompt': self._user_prompt,
            'current_stage': agent_name,
            'shared_data': self._shared_data.copy(),
            'conversation_history': self._conversation_history.copy()
        }
        
        # 根据代理类型提供特定的上下文
        if agent_name == 'research':
            # 研究代理需要用户原始请求
            context['requirements'] = self._parse_user_requirements()
            
        elif agent_name == 'writing':
            # 撰写代理需要研究结果
            context['research_data'] = self.get_research_data()
            
        elif agent_name == 'review':
            # 审核代理需要研究结果和初稿
            context['research_data'] = self.get_research_data()
            context['draft_content'] = self.get_writing_data()
            
        elif agent_name == 'polishing':
            # 润色代理需要所有前序结果
            context['research_data'] = self.get_research_data()
            context['draft_content'] = self.get_writing_data()
            context['review_feedback'] = self.get_review_data()
        
        return context
    
    def get_final_article(self) -> str:
        """获取最终的文章内容"""
        # 优先返回润色后的内容
        polishing_result = self.get_stage_result('polishing')
        if polishing_result and polishing_result.success:
            return polishing_result.data.get('final_article', '')
        
        # 如果润色失败，返回审核后的内容
        review_result = self.get_stage_result('review')
        if review_result and review_result.success:
            return review_result.data.get('revised_article', '')
        
        # 如果审核失败，返回初稿
        writing_result = self.get_stage_result('writing')
        if writing_result and writing_result.success:
            return writing_result.data.get('article_draft', '')
        
        return ""
    
    def _parse_user_requirements(self) -> Dict[str, Any]:
        """解析用户需求"""
        prompt = self._user_prompt.lower()
        
        # 基本需求解析
        requirements = {
            'topic': self._user_prompt,
            'length': 'medium',  # short, medium, long
            'style': 'formal',   # formal, casual, academic
            'language': 'chinese',
            'include_examples': False,
            'include_references': True
        }
        
        # 长度关键词检测
        if any(word in prompt for word in ['简短', '简单', '概述', 'brief', 'short']):
            requirements['length'] = 'short'
        elif any(word in prompt for word in ['详细', '深入', '全面', 'detailed', 'comprehensive', 'long']):
            requirements['length'] = 'long'
        
        # 风格关键词检测
        if any(word in prompt for word in ['学术', '论文', 'academic', 'research']):
            requirements['style'] = 'academic'
        elif any(word in prompt for word in ['通俗', '简单', 'casual', 'simple']):
            requirements['style'] = 'casual'
        
        # 示例需求检测
        if any(word in prompt for word in ['例子', '示例', '案例', 'example', 'case']):
            requirements['include_examples'] = True
        
        return requirements
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'user_prompt': self._user_prompt,
            'completed_stages': list(self._stage_results.keys()),
            'current_stage': self._current_stage,
            'total_conversations': len(self._conversation_history),
            'shared_data_keys': list(self._shared_data.keys()),
            'success_stages': [
                name for name, result in self._stage_results.items() 
                if result.success
            ],
            'failed_stages': [
                name for name, result in self._stage_results.items() 
                if not result.success
            ]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """将上下文转换为字典格式"""
        return {
            'user_prompt': self._user_prompt,
            'stage_results': {
                name: asdict(result) for name, result in self._stage_results.items()
            },
            'shared_data': self._shared_data,
            'conversation_history': self._conversation_history,
            'current_stage': self._current_stage,
            'execution_summary': self.get_execution_summary()
        }
    
    def clear(self):
        """清空上下文"""
        self._user_prompt = ""
        self._stage_results.clear()
        self._shared_data.clear()
        self._conversation_history.clear()
        self._current_stage = None
        self.logger.info("上下文已清空")