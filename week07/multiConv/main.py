"""
多轮对话系统主程序
演示订单查询、退款申请、发票开具等多轮交互流程
"""
import asyncio
import sys
import os
from typing import Dict, Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from loguru import logger

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from service import conversation_service
from hot_update import hot_updater
from tools import tool_manager


class ConversationDemo:
    """对话演示器"""
    
    def __init__(self):
        self.console = Console()
        self.session_id = None
        self.conversation_count = 0
    
    def show_banner(self):
        """显示欢迎横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                     多轮对话与工具调用系统                      ║
║                   基于 LangGraph 的智能客服                    ║
╠══════════════════════════════════════════════════════════════╣
║  功能特性:                                                   ║
║  • 多轮对话状态管理                                           ║
║  • 订单查询、退款申请、发票开具                                 ║
║  • 工具自动调用                                               ║
║  • 模型与插件热更新                                           ║
║  • 会话管理和健康检查                                         ║
╚══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="cyan")
    
    def show_help(self):
        """显示帮助信息"""
        help_table = Table(title="📖 使用帮助")
        help_table.add_column("命令", style="cyan", width=20)
        help_table.add_column("说明", style="white")
        
        help_table.add_row("/help", "显示此帮助信息")
        help_table.add_row("/demo", "运行预设演示场景")
        help_table.add_row("/status", "查看系统状态")
        help_table.add_row("/tools", "查看可用工具")
        help_table.add_row("/reload", "手动重载插件")
        help_table.add_row("/clear", "清屏")
        help_table.add_row("/quit", "退出程序")
        help_table.add_row("", "")
        help_table.add_row("对话示例:", "")
        help_table.add_row("", "• 查询订单ORD001")
        help_table.add_row("", "• 我要退款，订单ORD002，质量问题")
        help_table.add_row("", "• 开发票，订单ORD001，个人发票，张三")
        
        self.console.print(help_table)
    
    async def start_conversation(self) -> bool:
        """开始对话"""
        try:
            result = await conversation_service.start_conversation("demo_user")
            if result["success"]:
                self.session_id = result["session_id"]
                self.console.print(f"🤖 [bold green]{result['message']}[/bold green]")
                return True
            else:
                self.console.print(f"❌ 启动对话失败: {result.get('error')}")
                return False
        except Exception as e:
            self.console.print(f"❌ 启动对话异常: {str(e)}")
            return False
    
    async def send_message(self, message: str) -> Dict[str, Any]:
        """发送消息"""
        try:
            if not self.session_id:
                if not await self.start_conversation():
                    return {"success": False, "error": "无法启动对话"}
            
            result = await conversation_service.send_message(self.session_id, message)
            
            if result["success"]:
                # 显示AI响应
                response_panel = Panel(
                    result["response"],
                    title="🤖 AI助手回复",
                    border_style="green"
                )
                self.console.print(response_panel)
                
                # 显示状态信息
                if result.get("intent"):
                    status_info = f"意图: {result['intent']}"
                    if result.get("state"):
                        status_info += f" | 状态: {result['state']}"
                    self.console.print(f"[dim]{status_info}[/dim]")
                
                self.conversation_count += 1
                
            else:
                self.console.print(f"❌ [red]{result.get('error', '消息处理失败')}[/red]")
            
            return result
            
        except Exception as e:
            error_msg = f"消息发送异常: {str(e)}"
            self.console.print(f"❌ [red]{error_msg}[/red]")
            return {"success": False, "error": error_msg}
    
    async def show_status(self):
        """显示系统状态"""
        try:
            # 获取服务统计
            stats_result = await conversation_service.get_service_stats()
            
            # 获取热更新状态
            hot_update_status = hot_updater.get_status()
            
            # 创建状态表格
            status_table = Table(title="🔍 系统状态")
            status_table.add_column("组件", style="cyan")
            status_table.add_column("状态", style="white")
            
            if stats_result["success"]:
                stats = stats_result["stats"]
                status_table.add_row("服务运行时间", stats.get("service_uptime", "未知"))
                status_table.add_row("总对话数", str(stats.get("total_conversations", 0)))
                status_table.add_row("成功率", f"{stats.get('success_rate', 0):.1f}%")
                status_table.add_row("工具调用次数", str(stats.get("tool_calls", 0)))
                status_table.add_row("活跃会话数", str(stats.get("session_stats", {}).get("active_sessions", 0)))
            
            status_table.add_row("", "")
            status_table.add_row("热更新监视", "✅ 运行中" if hot_update_status["is_running"] else "❌ 已停止")
            status_table.add_row("模型版本", hot_update_status["model_version"])
            status_table.add_row("已载入插件", str(len(hot_update_status["loaded_plugins"])))
            status_table.add_row("可用工具", str(len(tool_manager.list_available_tools())))
            
            self.console.print(status_table)
            
        except Exception as e:
            self.console.print(f"❌ 获取状态失败: {str(e)}")
    
    def show_tools(self):
        """显示可用工具"""
        try:
            tools = tool_manager.list_available_tools()
            
            tools_table = Table(title="🔧 可用工具")
            tools_table.add_column("工具名称", style="cyan")
            tools_table.add_column("描述", style="white")
            
            for tool_name in tools:
                config = tool_manager.get_tool_config(tool_name)
                if config:
                    tools_table.add_row(tool_name, config.description)
                else:
                    tools_table.add_row(tool_name, "无描述")
            
            self.console.print(tools_table)
            
        except Exception as e:
            self.console.print(f"❌ 获取工具列表失败: {str(e)}")
    
    async def reload_plugins(self):
        """重载插件"""
        try:
            self.console.print("🔄 正在重载插件...")
            result = hot_updater.manual_reload_all()
            
            if result.get("tools_reloaded"):
                self.console.print("✅ 工具重载成功")
            else:
                self.console.print("❌ 工具重载失败")
            
            plugin_results = result.get("plugins", {})
            if plugin_results:
                success_count = sum(1 for success in plugin_results.values() if success)
                total_count = len(plugin_results)
                self.console.print(f"📦 插件重载: {success_count}/{total_count} 成功")
            
        except Exception as e:
            self.console.print(f"❌ 重载失败: {str(e)}")
    
    async def run_demo_scenarios(self):
        """运行演示场景"""
        scenarios = [
            {
                "name": "订单查询演示",
                "messages": [
                    "查订单",
                    "ORD001"
                ]
            },
            {
                "name": "退款申请演示",
                "messages": [
                    "我要申请退款",
                    "ORD002",
                    "质量问题"
                ]
            },
            {
                "name": "发票开具演示",
                "messages": [
                    "我要开发票",
                    "订单ORD001，个人发票，抬头李四"
                ]
            },
            {
                "name": "企业发票演示",
                "messages": [
                    "开企业发票，订单ORD002，公司名称测试科技有限公司，税号91110000123456789X"
                ]
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            self.console.print(f"\n[bold cyan]场景 {i}: {scenario['name']}[/bold cyan]")
            self.console.print("=" * 50)
            
            # 开始新对话
            if not await self.start_conversation():
                continue
            
            # 发送消息序列
            for j, message in enumerate(scenario['messages'], 1):
                self.console.print(f"\n[bold yellow]👤 用户 (步骤{j}):[/bold yellow] {message}")
                await self.send_message(message)
                
                # 添加延迟使演示更自然
                await asyncio.sleep(1)
            
            # 结束对话
            await conversation_service.end_conversation(self.session_id)
            self.session_id = None
            
            self.console.print("\n[dim]--- 场景演示完成 ---[/dim]")
            
            if i < len(scenarios):
                input("\n按回车键继续下一个场景...")
    
    async def interactive_mode(self):
        """交互模式"""
        self.console.print("\n[bold green]🚀 进入交互模式[/bold green]")
        self.console.print("输入 /help 查看帮助，输入 /quit 退出\n")
        
        # 启动对话
        await self.start_conversation()
        
        while True:
            try:
                # 获取用户输入
                user_input = self.console.input("[bold yellow]👤 您:[/bold yellow] ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'quit' or command == 'exit':
                        break
                    elif command == 'help':
                        self.show_help()
                    elif command == 'demo':
                        await self.run_demo_scenarios()
                        # 重新启动对话
                        await self.start_conversation()
                    elif command == 'status':
                        await self.show_status()
                    elif command == 'tools':
                        self.show_tools()
                    elif command == 'reload':
                        await self.reload_plugins()
                    elif command == 'clear':
                        os.system('clear' if os.name == 'posix' else 'cls')
                        self.show_banner()
                    else:
                        self.console.print(f"❓ 未知命令: {command}")
                else:
                    # 发送消息
                    await self.send_message(user_input)
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"❌ 处理输入异常: {str(e)}")
        
        # 结束对话
        if self.session_id:
            await conversation_service.end_conversation(self.session_id)
    
    async def run(self):
        """运行演示程序"""
        self.show_banner()
        
        # 检查系统状态
        self.console.print("🔍 正在检查系统状态...")
        
        try:
            health_result = await conversation_service.health_check()
            if health_result.get("healthy"):
                self.console.print("✅ 系统状态正常\n")
            else:
                self.console.print("⚠️  系统状态异常，但继续运行...\n")
        except Exception as e:
            self.console.print(f"⚠️  无法检查系统状态: {str(e)}\n")
        
        # 显示选项
        self.console.print("请选择运行模式:")
        self.console.print("1. 🎭 自动演示模式 - 运行预设场景")
        self.console.print("2. 💬 交互对话模式 - 自由对话")
        self.console.print("3. 📊 查看系统状态")
        self.console.print("4. 🚪 退出程序")
        
        while True:
            try:
                choice = self.console.input("\n[bold cyan]请输入选项 (1-4):[/bold cyan] ").strip()
                
                if choice == '1':
                    await self.run_demo_scenarios()
                elif choice == '2':
                    await self.interactive_mode()
                elif choice == '3':
                    await self.show_status()
                    continue
                elif choice == '4':
                    break
                else:
                    self.console.print("❓ 请输入有效选项 (1-4)")
                    continue
                
                # 询问是否继续
                if choice in ['1', '2']:
                    continue_choice = self.console.input("\n[cyan]是否返回主菜单? (y/n):[/cyan] ").strip().lower()
                    if continue_choice != 'y':
                        break
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        self.console.print("\n[bold green]👋 感谢使用多轮对话系统！[/bold green]")


async def main():
    """主函数"""
    try:
        # 配置日志
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # 创建并运行演示
        demo = ConversationDemo()
        await demo.run()
        
    except Exception as e:
        print(f"❌ 程序运行异常: {str(e)}")
        logger.exception("程序异常")


if __name__ == "__main__":
    asyncio.run(main())