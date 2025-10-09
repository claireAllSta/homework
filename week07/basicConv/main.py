"""
主程序入口
演示基础对话链的使用
"""
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

from chain import BasicConversationChain
from config import settings


console = Console()


def demo_basic_chain(current_date: Optional[datetime] = None):
    """演示基础对话链功能"""
    # 设置演示用的当前时间（2025年9月28日）
    demo_date = current_date
    

    
    # 创建对话链
    chain = BasicConversationChain(current_date=demo_date)
    
    # 测试用例
    test_queries = [
        "我昨天下的单",
        "前天买的东西什么时候到？",
        "明天的会议准备好了吗？",
        "上周的报告提交了吗？",
        "下个月发工资",
    ]
    
    for query in test_queries:
        console.print(f"\n[bold blue]用户查询：[/bold blue]{query}")
        
        try:
            # 调用链处理查询
            result = chain.invoke(query)
            
            # 显示结果
            if result.get('error'):
                console.print(f"[red]错误：{result.get('explanation')}[/red]")
            else:
                console.print(Panel(
                    JSON.from_data(result),
                    title="推断结果",
                    border_style="green"
                ))
                
                # 简化输出
                console.print(
                    f"[green]✓[/green] '{result.get('original_query')}' → "
                    f"[yellow]{result.get('inferred_date')}[/yellow]"
                )
        
        except Exception as e:
            console.print(f"[red]处理查询时出错: {str(e)}[/red]")


if __name__ == "__main__":
    import sys
    
    # 默认运行演示
    demo_basic_chain()