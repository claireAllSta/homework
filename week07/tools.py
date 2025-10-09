"""
工具函数模块 - 订单查询相关功能
"""
from typing import Dict, Any, Optional
import json
from config import MOCK_ORDERS


def query_order(order_id: str) -> Dict[str, Any]:
    """
    查询订单信息的工具函数
    
    Args:
        order_id: 订单号
        
    Returns:
        包含订单详细信息的字典
    """
    order_id = order_id.strip().upper()
    
    if order_id in MOCK_ORDERS:
        order_info = MOCK_ORDERS[order_id].copy()
        return {
            "success": True,
            "order_info": order_info,
            "message": f"订单 {order_id} 查询成功"
        }
    else:
        return {
            "success": False,
            "order_info": None,
            "message": f"未找到订单号为 {order_id} 的订单，请检查订单号是否正确"
        }


def format_order_info(order_data: Dict[str, Any]) -> str:
    """
    格式化订单信息用于显示
    
    Args:
        order_data: 订单查询结果数据
        
    Returns:
        格式化后的订单信息字符串
    """
    if not order_data["success"]:
        return f"❌ {order_data['message']}"
    
    order_info = order_data["order_info"]
    logistics = order_info["logistics"]
    
    formatted_info = f"""
✅ 订单查询成功！

📦 订单信息
• 订单号: {order_info['order_id']}
• 订单状态: {order_info['status']}
• 商品: {', '.join(order_info['items'])}
• 总金额: ¥{order_info['total']}

🚚 物流信息
• 快递公司: {logistics['carrier']}
• 快递单号: {order_info['tracking_number']}
• 当前位置: {logistics['current_location']}
• 预计送达: {logistics['estimated_delivery']}
    """.strip()
    
    return formatted_info


def apply_refund(order_id: str, reason: str = "") -> Dict[str, Any]:
    """
    申请退款的工具函数
    
    Args:
        order_id: 订单号
        reason: 退款原因
        
    Returns:
        退款申请结果
    """
    order_id = order_id.strip().upper()
    
    if order_id not in MOCK_ORDERS:
        return {
            "success": False,
            "message": f"未找到订单号为 {order_id} 的订单"
        }
    
    order_info = MOCK_ORDERS[order_id]
    
    # 检查订单状态是否可以退款
    if order_info["status"] == "已完成":
        return {
            "success": False,
            "message": f"订单 {order_id} 已完成配送，退款申请需要联系客服处理"
        }
    elif order_info["status"] == "已取消":
        return {
            "success": False,
            "message": f"订单 {order_id} 已取消，无需重复申请退款"
        }
    else:
        return {
            "success": True,
            "refund_id": f"REF{order_id[3:]}001",
            "message": f"退款申请提交成功！退款单号: REF{order_id[3:]}001\n" +
                      f"退款金额: ¥{order_info['total']}\n" +
                      f"预计3-5个工作日到账"
        }


# 工具函数映射表
AVAILABLE_TOOLS = {
    "query_order": {
        "function": query_order,
        "description": "查询订单状态和物流信息",
        "parameters": {
            "order_id": {
                "type": "string",
                "description": "订单号，格式如 ORD001"
            }
        }
    },
    "apply_refund": {
        "function": apply_refund,
        "description": "申请订单退款",
        "parameters": {
            "order_id": {
                "type": "string", 
                "description": "要退款的订单号"
            },
            "reason": {
                "type": "string",
                "description": "退款原因",
                "default": ""
            }
        }
    }
}