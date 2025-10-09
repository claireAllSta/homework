"""
输出格式化工具
"""

from datetime import datetime
from typing import Dict, Any
import json


class OutputFormatter:
    """输出格式化器"""
    
    def format_article(self, result: Dict[str, Any]) -> str:
        """格式化文章输出"""
        
        if not result.get('success') or not result.get('article_content'):
            return "# 文章生成失败\n\n无法生成文章内容。"
        
        article_content = result['article_content']
        
        # 添加元信息
        header = f"""---
生成时间: {result.get('end_time', '未知')}
用户请求: {result.get('user_prompt', '未知')}
执行状态: {'成功' if result.get('success') else '失败'}
---

"""
        
        return header + article_content
    
    def format_execution_report(self, result: Dict[str, Any]) -> str:
        """格式化执行报告"""
        
        report_parts = []
        
        # 报告标题
        report_parts.append("# 多代理文章自动编写系统 - 执行报告")
        report_parts.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("")
        
        # 基本信息
        report_parts.append("## 基本信息")
        report_parts.append(f"- **用户请求**: {result.get('user_prompt', '未知')}")
        report_parts.append(f"- **执行状态**: {'✅ 成功' if result.get('success') else '❌ 失败'}")
        report_parts.append(f"- **开始时间**: {result.get('start_time', '未知')}")
        report_parts.append(f"- **结束时间**: {result.get('end_time', '未知')}")
        report_parts.append("")
        
        # 执行统计
        execution_log = result.get('execution_log', [])
        total_attempts = sum(len(stage.get('attempts', [])) for stage in execution_log)
        failed_stages = len([stage for stage in execution_log if not stage.get('success', False)])
        
        report_parts.append("## 执行统计")
        report_parts.append(f"- **总阶段数**: {len(execution_log)}")
        report_parts.append(f"- **成功阶段**: {len(execution_log) - failed_stages}")
        report_parts.append(f"- **失败阶段**: {failed_stages}")
        report_parts.append(f"- **总尝试次数**: {total_attempts}")
        report_parts.append("")
        
        # 阶段详情
        report_parts.append("## 阶段执行详情")
        for stage in execution_log:
            stage_name = stage.get('stage', '未知阶段')
            description = stage.get('description', '')
            success = stage.get('success', False)
            attempts = stage.get('attempts', [])
            
            status_icon = "✅" if success else "❌"
            report_parts.append(f"### {status_icon} {stage_name} - {description}")
            report_parts.append(f"- **状态**: {'成功' if success else '失败'}")
            report_parts.append(f"- **开始时间**: {stage.get('start_time', '未知')}")
            report_parts.append(f"- **结束时间**: {stage.get('end_time', '未知')}")
            report_parts.append(f"- **尝试次数**: {len(attempts)}")
            
            if attempts:
                report_parts.append("- **尝试详情**:")
                for attempt in attempts:
                    attempt_status = "✅" if attempt.get('success') else "❌"
                    report_parts.append(f"  - {attempt_status} 尝试 {attempt.get('attempt_number', '?')}: "
                                      f"{attempt.get('agent_name', '未知代理')} "
                                      f"({attempt.get('retry_level', '未知级别')})")
                    if attempt.get('error'):
                        report_parts.append(f"    错误: {attempt['error']}")
            
            report_parts.append("")
        
        # 错误日志
        error_log = result.get('error_log', [])
        if error_log:
            report_parts.append("## 异常处理日志")
            for error in error_log:
                report_parts.append(f"- **{error.get('timestamp', '未知时间')}** "
                                  f"[{error.get('stage', '未知阶段')}]: {error.get('error', '未知错误')}")
            report_parts.append("")
        
        # 最终结果
        if result.get('success') and result.get('article_content'):
            word_count = len(result['article_content'])
            report_parts.append("## 最终成果")
            report_parts.append(f"- **文章字数**: {word_count}")
            report_parts.append(f"- **文章质量**: 系统自动生成，经过四轮代理协作优化")
            report_parts.append("")
        
        return '\n'.join(report_parts)