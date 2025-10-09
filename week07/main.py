"""
主程序入口 - 多轮对话与工具调用演示
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dialogue_graph import DialogueManager
from tools import MOCK_ORDERS


def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("🤖 智能客服系统 - 订单查询与退款申请")
    print("=" * 60)
    print("\n功能说明：")
    print("• 支持订单查询：说'查订单'")
    print("• 支持退款申请：说'申请退款'")
    print("• 输入'quit'退出系统")
    print("• 输入'reset'重置对话")
    print("• 输入'help'查看可用订单号")
    print("\n可用测试订单号：")
    for order_id, info in MOCK_ORDERS.items():
        print(f"  {order_id}: {info['status']} - {', '.join(info['items'])}")
    print("-" * 60)


def print_help():
    """打印帮助信息"""
    print("\n📝 使用示例：")
    print("1. 查询订单：")
    print("   用户: 我要查订单")
    print("   系统: 请提供您的订单号（格式如：ORD001）")
    print("   用户: ORD001")
    print("   系统: [显示订单详细信息]")
    print("\n2. 申请退款：")
    print("   用户: 我要申请退款")
    print("   系统: 请提供要申请退款的订单号（格式如：ORD001）")
    print("   用户: ORD002")
    print("   系统: [处理退款申请]")
    print("\n可用订单号：")
    for order_id, info in MOCK_ORDERS.items():
        print(f"  {order_id}: {info['status']}")
    print()


def demo_conversation():
    """演示对话功能"""
    print("\n🎭 自动演示模式")
    print("-" * 40)
    
    dialogue_manager = DialogueManager()
    
    # 演示场景1：查询订单
    print("\n场景1：订单查询")
    print("用户: 我要查订单")
    response1 = dialogue_manager.chat("我要查订单")
    print(f"助手: {response1}")
    
    print("用户: ORD001")
    response2 = dialogue_manager.chat("ORD001")
    print(f"助手: {response2}")
    
    # 重置对话
    dialogue_manager.reset()
    
    # 演示场景2：退款申请
    print("\n" + "="*40)
    print("场景2：退款申请")
    print("用户: 申请退款")
    response3 = dialogue_manager.chat("申请退款")
    print(f"助手: {response3}")
    
    print("用户: ORD002")
    response4 = dialogue_manager.chat("ORD002")
    print(f"助手: {response4}")
    
    print("\n演示完成！")


def main():
    """主函数"""
    print_welcome()
    
    demo_conversation()

if __name__ == "__main__":
    main()