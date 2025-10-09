#!/usr/bin/env python3
"""
基于 MCP 协议的多代理文章自动编写系统
主入口文件

功能：
- 协调四个专业化代理完成文章写作任务
- 研究代理 -> 撰写代理 -> 审核代理 -> 润色代理
- 支持失败重试和异常处理
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from agents.research_agent import ResearchAgent
from agents.writing_agent import WritingAgent
from agents.review_agent import ReviewAgent
from agents.polishing_agent import PolishingAgent
from core.mcp_context import MCPContext
from core.workflow_manager import WorkflowManager
from core.retry_manager import RetryManager
from utils.logger import setup_logger
from utils.output_formatter import OutputFormatter


class MultiAgentArticleWriter:
    """多代理文章自动编写系统主类"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.context = MCPContext()
        self.workflow_manager = WorkflowManager()
        self.retry_manager = RetryManager()
        self.output_formatter = OutputFormatter()
        
        # 初始化代理
        self.agents = {
            'research': ResearchAgent(self.context),
            'writing': WritingAgent(self.context),
            'review': ReviewAgent(self.context),
            'polishing': PolishingAgent(self.context)
        }
        
        self.execution_log = []
    
    async def write_article(self, user_prompt: str) -> Dict[str, Any]:
        """
        主要的文章写作流程
        
        Args:
            user_prompt: 用户输入的文章主题或要求
            
        Returns:
            包含文章内容和执行记录的字典
        """
        self.logger.info(f"开始处理用户请求: {user_prompt}")
        
        # 初始化上下文
        self.context.set_user_prompt(user_prompt)
        start_time = datetime.now()
        
        result = {
            'user_prompt': user_prompt,
            'article_content': '',
            'execution_log': [],
            'error_log': [],
            'start_time': start_time.isoformat(),
            'end_time': None,
            'success': False
        }
        
        try:
            # 阶段 1: 研究代理
            await self._execute_stage('research', "信息收集与研究", result)
            
            # 阶段 2: 撰写代理
            await self._execute_stage('writing', "文章初稿撰写", result)
            
            # 阶段 3: 审核代理
            await self._execute_stage('review', "内容审核与建议", result)
            
            # 阶段 4: 润色代理
            await self._execute_stage('polishing', "文章润色与优化", result)
            
            # 获取最终文章内容
            result['article_content'] = self.context.get_final_article()
            result['success'] = True
            
        except Exception as e:
            self.logger.error(f"文章写作过程中发生错误: {e}")
            result['error_log'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'stage': getattr(self, '_current_stage', 'unknown')
            })
        
        finally:
            result['end_time'] = datetime.now().isoformat()
            result['execution_log'] = self.execution_log
            
        return result
    
    async def _execute_stage(self, agent_name: str, stage_description: str, result: Dict[str, Any]):
        """执行单个代理阶段，包含重试逻辑"""
        self._current_stage = agent_name
        self.logger.info(f"开始执行阶段: {stage_description}")
        
        stage_start = datetime.now()
        stage_log = {
            'stage': agent_name,
            'description': stage_description,
            'start_time': stage_start.isoformat(),
            'end_time': None,
            'attempts': [],
            'success': False
        }
        
        try:
            # 使用重试管理器执行代理任务
            agent_result = await self.retry_manager.execute_with_retry(
                agent_name,
                self.agents[agent_name],
                self.context
            )
            
            # 记录执行结果
            stage_log['attempts'] = self.retry_manager.get_stage_attempts(agent_name)
            stage_log['success'] = True
            stage_log['result'] = agent_result
            
            # 更新上下文
            # 该方法在 MCP 协议中用于更新分阶段任务执行结果的核心方法，主要用于多步骤任务的进度跟踪和结果汇报。
            self.context.update_stage_result(agent_name, agent_result)
            
            print(f"✅ {stage_description} 完成")
            
        except Exception as e:
            stage_log['error'] = str(e)
            stage_log['attempts'] = self.retry_manager.get_stage_attempts(agent_name)
            result['error_log'].append({
                'timestamp': datetime.now().isoformat(),
                'stage': agent_name,
                'error': str(e)
            })
            print(f"❌ {stage_description} 失败: {e}")
            raise
        
        finally:
            stage_log['end_time'] = datetime.now().isoformat()
            self.execution_log.append(stage_log)
    
    def save_result(self, result: Dict[str, Any], output_dir: str = "output"):
        """保存执行结果到文件"""
        output_path = Path(output_dir)
        print(f"保存结果到: {output_path}")
        output_path.mkdir(exist_ok=True)
      
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果 JSON
        result_file = output_path / f"article_result_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存格式化的文章
        if result.get('article_content'):
            article_file = output_path / f"article_{timestamp}.md"
            formatted_content = self.output_formatter.format_article(result)
            with open(article_file, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
        
        # 保存执行报告
        report_file = output_path / f"execution_report_{timestamp}.md"
        report_content = self.output_formatter.format_execution_report(result)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"结果已保存到: {output_path}")
        return {
            'result_file': str(result_file),
            'article_file': str(article_file) if result.get('article_content') else None,
            'report_file': str(report_file)
        }


async def main():
    """主函数"""
    print("🤖 基于 MCP 协议的多代理文章自动编写系统")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        user_prompt = input("请输入您想要撰写的文章主题或要求: ")
    
    if not user_prompt.strip():
        print("❌ 请提供有效的文章主题或要求")
        return
    
    writer = MultiAgentArticleWriter()
    
    print(f"\n📝 开始撰写文章: {user_prompt}")
    print("-" * 50)
    
    try:
        result = await writer.write_article(user_prompt)
        
        # 保存结果
        output_files = writer.save_result(result)
        
        print("\n" + "=" * 50)
        if result['success']:
            print("✅ 文章撰写完成!")
            print(f"📄 文章文件: {output_files.get('article_file', '未生成')}")
            print(f"📊 执行报告: {output_files['report_file']}")
            print(f"📋 详细结果: {output_files['result_file']}")
        else:
            print("❌ 文章撰写失败")
            print(f"📋 错误日志: {output_files['result_file']}")
            print(f"📊 执行报告: {output_files['report_file']}")
        
        # 显示简要统计
        total_attempts = sum(len(stage.get('attempts', [])) for stage in result['execution_log'])
        failed_stages = len([stage for stage in result['execution_log'] if not stage.get('success', False)])
        
        print(f"\n📈 执行统计:")
        print(f"   总尝试次数: {total_attempts}")
        print(f"   失败阶段数: {failed_stages}")
        print(f"   执行时长: {result.get('end_time', '')} - {result.get('start_time', '')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        logging.exception("系统执行异常")


if __name__ == "__main__":
    asyncio.run(main())