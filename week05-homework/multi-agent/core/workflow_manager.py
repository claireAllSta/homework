"""
工作流管理器
管理代理间的协作流程和数据传递
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class WorkflowStage(Enum):
    """工作流阶段"""
    RESEARCH = "research"
    WRITING = "writing"
    REVIEW = "review"
    POLISHING = "polishing"


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workflow_id = None
        self.current_stage = None
        self.status = WorkflowStatus.PENDING
        self.stage_dependencies = {
            WorkflowStage.RESEARCH: [],
            WorkflowStage.WRITING: [WorkflowStage.RESEARCH],
            WorkflowStage.REVIEW: [WorkflowStage.RESEARCH, WorkflowStage.WRITING],
            WorkflowStage.POLISHING: [WorkflowStage.RESEARCH, WorkflowStage.WRITING, WorkflowStage.REVIEW]
        }
        self.completed_stages = set()
        self.stage_results = {}
        self.workflow_metadata = {}
        
    def initialize_workflow(self, workflow_id: str, user_prompt: str) -> Dict[str, Any]:
        """初始化工作流"""
        self.workflow_id = workflow_id
        self.status = WorkflowStatus.RUNNING
        self.workflow_metadata = {
            'workflow_id': workflow_id,
            'user_prompt': user_prompt,
            'start_time': datetime.now().isoformat(),
            'estimated_completion_time': None,
            'total_stages': len(WorkflowStage),
            'stage_order': [stage.value for stage in WorkflowStage]
        }
        
        self.logger.info(f"工作流已初始化: {workflow_id}")
        
        return {
            'workflow_id': workflow_id,
            'status': self.status.value,
            'next_stage': WorkflowStage.RESEARCH.value,
            'metadata': self.workflow_metadata
        }
    
    def can_execute_stage(self, stage: WorkflowStage) -> bool:
        """检查是否可以执行指定阶段"""
        dependencies = self.stage_dependencies.get(stage, [])
        return all(dep in self.completed_stages for dep in dependencies)
    
    def get_next_stage(self) -> Optional[WorkflowStage]:
        """获取下一个可执行的阶段"""
        for stage in WorkflowStage:
            if stage not in self.completed_stages and self.can_execute_stage(stage):
                return stage
        return None
    
    def start_stage(self, stage: WorkflowStage) -> Dict[str, Any]:
        """开始执行阶段"""
        if not self.can_execute_stage(stage):
            missing_deps = [
                dep.value for dep in self.stage_dependencies[stage]
                if dep not in self.completed_stages
            ]
            raise ValueError(f"阶段 {stage.value} 依赖未满足，缺少: {missing_deps}")
        
        self.current_stage = stage
        start_time = datetime.now().isoformat()
        
        stage_info = {
            'stage': stage.value,
            'status': 'started',
            'start_time': start_time,
            'dependencies_met': True,
            'completed_dependencies': [dep.value for dep in self.stage_dependencies[stage]]
        }
        
        self.logger.info(f"开始执行阶段: {stage.value}")
        return stage_info
    
    def complete_stage(self, stage: WorkflowStage, result: Dict[str, Any]) -> Dict[str, Any]:
        """完成阶段执行"""
        if self.current_stage != stage:
            raise ValueError(f"当前阶段 {self.current_stage} 与完成阶段 {stage} 不匹配")
        
        self.completed_stages.add(stage)
        self.stage_results[stage] = result
        
        completion_time = datetime.now().isoformat()
        completion_info = {
            'stage': stage.value,
            'status': 'completed',
            'completion_time': completion_time,
            'result_summary': self._generate_stage_summary(stage, result)
        }
        
        # 检查是否所有阶段都已完成
        if len(self.completed_stages) == len(WorkflowStage):
            self.status = WorkflowStatus.COMPLETED
            completion_info['workflow_status'] = 'completed'
            self.workflow_metadata['end_time'] = completion_time
        else:
            next_stage = self.get_next_stage()
            completion_info['next_stage'] = next_stage.value if next_stage else None
        
        self.current_stage = None
        self.logger.info(f"阶段完成: {stage.value}")
        
        return completion_info
    
    def fail_stage(self, stage: WorkflowStage, error: str) -> Dict[str, Any]:
        """标记阶段失败"""
        self.status = WorkflowStatus.FAILED
        self.current_stage = None
        
        failure_info = {
            'stage': stage.value,
            'status': 'failed',
            'error': error,
            'failure_time': datetime.now().isoformat(),
            'workflow_status': 'failed'
        }
        
        self.logger.error(f"阶段失败: {stage.value}, 错误: {error}")
        return failure_info
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """获取工作流状态"""
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'current_stage': self.current_stage.value if self.current_stage else None,
            'completed_stages': [stage.value for stage in self.completed_stages],
            'remaining_stages': [
                stage.value for stage in WorkflowStage 
                if stage not in self.completed_stages
            ],
            'progress_percentage': len(self.completed_stages) / len(WorkflowStage) * 100,
            'metadata': self.workflow_metadata
        }
    
    def get_stage_dependencies_info(self) -> Dict[str, Any]:
        """获取阶段依赖信息"""
        return {
            stage.value: {
                'dependencies': [dep.value for dep in deps],
                'can_execute': self.can_execute_stage(stage),
                'completed': stage in self.completed_stages
            }
            for stage, deps in self.stage_dependencies.items()
        }
    
    def _generate_stage_summary(self, stage: WorkflowStage, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成阶段结果摘要"""
        summary = {
            'success': result.get('success', False),
            'data_keys': list(result.get('data', {}).keys()),
            'metadata_keys': list(result.get('metadata', {}).keys())
        }
        
        # 根据阶段类型生成特定摘要
        data = result.get('data', {})
        
        if stage == WorkflowStage.RESEARCH:
            summary['research_summary'] = {
                'sources_count': len(data.get('sources', [])),
                'key_points_count': len(data.get('key_points', [])),
                'topic': data.get('topic', 'Unknown')
            }
        
        elif stage == WorkflowStage.WRITING:
            summary['writing_summary'] = {
                'word_count': data.get('word_count', 0),
                'structure_elements': len(data.get('structure', [])),
                'has_draft': bool(data.get('article_draft'))
            }
        
        elif stage == WorkflowStage.REVIEW:
            summary['review_summary'] = {
                'overall_score': data.get('overall_score', 0),
                'suggestions_count': len(data.get('suggestions', [])),
                'has_revision': bool(data.get('revised_article'))
            }
        
        elif stage == WorkflowStage.POLISHING:
            summary['polishing_summary'] = {
                'polish_notes_count': len(data.get('polish_notes', [])),
                'has_final_article': bool(data.get('final_article'))
            }
        
        return summary
    
    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        """获取执行时间线"""
        timeline = []
        
        # 添加工作流开始事件
        if self.workflow_metadata.get('start_time'):
            timeline.append({
                'timestamp': self.workflow_metadata['start_time'],
                'event_type': 'workflow_start',
                'description': '工作流开始执行'
            })
        
        # 添加已完成阶段的事件
        for stage in WorkflowStage:
            if stage in self.completed_stages:
                result = self.stage_results.get(stage, {})
                timeline.append({
                    'timestamp': result.get('metadata', {}).get('completion_time', ''),
                    'event_type': 'stage_completion',
                    'stage': stage.value,
                    'description': f'阶段 {stage.value} 执行完成'
                })
        
        # 添加工作流结束事件
        if self.status == WorkflowStatus.COMPLETED and self.workflow_metadata.get('end_time'):
            timeline.append({
                'timestamp': self.workflow_metadata['end_time'],
                'event_type': 'workflow_completion',
                'description': '工作流执行完成'
            })
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def validate_workflow_integrity(self) -> Dict[str, Any]:
        """验证工作流完整性"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # 检查阶段依赖关系
        for stage, deps in self.stage_dependencies.items():
            if stage in self.completed_stages:
                for dep in deps:
                    if dep not in self.completed_stages:
                        validation_result['valid'] = False
                        validation_result['issues'].append(
                            f"阶段 {stage.value} 已完成，但依赖 {dep.value} 未完成"
                        )
        
        # 检查结果数据完整性
        for stage in self.completed_stages:
            result = self.stage_results.get(stage)
            if not result:
                validation_result['warnings'].append(
                    f"阶段 {stage.value} 缺少结果数据"
                )
            elif not result.get('success'):
                validation_result['warnings'].append(
                    f"阶段 {stage.value} 执行不成功"
                )
        
        return validation_result
    
    def export_workflow_report(self) -> Dict[str, Any]:
        """导出工作流报告"""
        return {
            'workflow_metadata': self.workflow_metadata,
            'execution_status': self.get_workflow_status(),
            'stage_dependencies': self.get_stage_dependencies_info(),
            'execution_timeline': self.get_execution_timeline(),
            'integrity_check': self.validate_workflow_integrity(),
            'stage_results': {
                stage.value: result for stage, result in self.stage_results.items()
            }
        }