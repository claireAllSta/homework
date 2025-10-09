"""
业务工具实现模块
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from loguru import logger
import importlib
import inspect
from pathlib import Path

from .config import settings, AVAILABLE_TOOLS, ToolConfig


@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    status: str
    amount: float
    create_time: datetime
    items: List[Dict[str, Any]]
    shipping_info: Optional[Dict[str, Any]] = None


@dataclass
class RefundInfo:
    """退款信息"""
    refund_id: str
    order_id: str
    amount: float
    reason: str
    status: str
    create_time: datetime


@dataclass
class InvoiceInfo:
    """发票信息"""
    invoice_id: str
    order_id: str
    invoice_type: str
    title: str
    tax_number: Optional[str]
    amount: float
    status: str
    create_time: datetime


class MockDatabase:
    """模拟数据库，用于演示"""
    
    def __init__(self):
        self.orders = self._init_orders()
        self.refunds = {}
        self.invoices = {}
    
    def _init_orders(self) -> Dict[str, OrderInfo]:
        """初始化模拟订单数据"""
        return {
            "ORD001": OrderInfo(
                order_id="ORD001",
                status="已发货",
                amount=299.0,
                create_time=datetime.now() - timedelta(days=2),
                items=[{"name": "智能手机", "quantity": 1, "price": 299.0}],
                shipping_info={
                    "courier": "顺丰快递",
                    "tracking_number": "SF123456789",
                    "status": "运输中"
                }
            ),
            "ORD002": OrderInfo(
                order_id="ORD002", 
                status="已完成",
                amount=159.0,
                create_time=datetime.now() - timedelta(days=5),
                items=[{"name": "蓝牙耳机", "quantity": 1, "price": 159.0}],
                shipping_info={
                    "courier": "圆通快递",
                    "tracking_number": "YT987654321",
                    "status": "已签收"
                }
            ),
            "ORD003": OrderInfo(
                order_id="ORD003",
                status="待支付",
                amount=88.0,
                create_time=datetime.now() - timedelta(hours=3),
                items=[{"name": "数据线", "quantity": 2, "price": 44.0}]
            )
        }


# 全局数据库实例
db = MockDatabase()


class BusinessTools:
    """业务工具类"""
    
    @staticmethod
    def query_order(order_id: str) -> Dict[str, Any]:
        """
        查询订单信息
        
        Args:
            order_id: 订单号
            
        Returns:
            订单查询结果
        """
        try:
            logger.info(f"查询订单: {order_id}")
            
            if order_id not in db.orders:
                return {
                    "success": False,
                    "error": f"未找到订单号为 {order_id} 的订单",
                    "order_id": order_id
                }
            
            order = db.orders[order_id]
            result = {
                "success": True,
                "order_info": {
                    "order_id": order.order_id,
                    "status": order.status,
                    "amount": order.amount,
                    "create_time": order.create_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "items": order.items
                }
            }
            
            if order.shipping_info:
                result["shipping_info"] = order.shipping_info
            
            logger.info(f"订单查询成功: {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"查询订单失败 {order_id}: {str(e)}")
            return {
                "success": False,
                "error": f"查询订单时发生错误: {str(e)}",
                "order_id": order_id
            }
    
    @staticmethod 
    def apply_refund(order_id: str, reason: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """
        申请退款
        
        Args:
            order_id: 订单号
            reason: 退款原因
            amount: 退款金额，不填则全额退款
            
        Returns:
            退款申请结果
        """
        try:
            logger.info(f"申请退款: 订单{order_id}, 原因: {reason}")
            
            if order_id not in db.orders:
                return {
                    "success": False,
                    "error": f"未找到订单号为 {order_id} 的订单",
                    "order_id": order_id
                }
            
            order = db.orders[order_id]
            
            # 检查订单状态
            if order.status == "待支付":
                return {
                    "success": False,
                    "error": "待支付订单无法申请退款，请直接取消订单",
                    "order_id": order_id
                }
            
            # 确定退款金额
            refund_amount = amount if amount is not None else order.amount
            if refund_amount > order.amount:
                return {
                    "success": False,
                    "error": f"退款金额不能超过订单金额 {order.amount}",
                    "order_id": order_id
                }
            
            # 创建退款记录
            refund_id = f"REF{datetime.now().strftime('%Y%m%d%H%M%S')}"
            refund = RefundInfo(
                refund_id=refund_id,
                order_id=order_id,
                amount=refund_amount,
                reason=reason,
                status="处理中",
                create_time=datetime.now()
            )
            
            db.refunds[refund_id] = refund
            
            result = {
                "success": True,
                "refund_info": {
                    "refund_id": refund_id,
                    "order_id": order_id,
                    "amount": refund_amount,
                    "reason": reason,
                    "status": "处理中",
                    "estimated_days": "3-7个工作日"
                }
            }
            
            logger.info(f"退款申请成功: {refund_id}")
            return result
            
        except Exception as e:
            logger.error(f"申请退款失败 {order_id}: {str(e)}")
            return {
                "success": False,
                "error": f"申请退款时发生错误: {str(e)}",
                "order_id": order_id
            }
    
    @staticmethod
    def issue_invoice(order_id: str, invoice_type: str, invoice_title: str, 
                     tax_number: Optional[str] = None) -> Dict[str, Any]:
        """
        开具发票
        
        Args:
            order_id: 订单号
            invoice_type: 发票类型（个人/企业）
            invoice_title: 发票抬头
            tax_number: 税号（企业发票必填）
            
        Returns:
            发票开具结果
        """
        try:
            logger.info(f"开具发票: 订单{order_id}, 类型: {invoice_type}")
            
            if order_id not in db.orders:
                return {
                    "success": False,
                    "error": f"未找到订单号为 {order_id} 的订单",
                    "order_id": order_id
                }
            
            order = db.orders[order_id]
            
            # 检查订单状态
            if order.status in ["待支付", "已取消"]:
                return {
                    "success": False,
                    "error": f"订单状态为'{order.status}'，无法开具发票",
                    "order_id": order_id
                }
            
            # 企业发票必须提供税号
            if invoice_type == "企业" and not tax_number:
                return {
                    "success": False,
                    "error": "企业发票必须提供税号",
                    "order_id": order_id
                }
            
            # 创建发票记录
            invoice_id = f"INV{datetime.now().strftime('%Y%m%d%H%M%S')}"
            invoice = InvoiceInfo(
                invoice_id=invoice_id,
                order_id=order_id,
                invoice_type=invoice_type,
                title=invoice_title,
                tax_number=tax_number,
                amount=order.amount,
                status="开具中",
                create_time=datetime.now()
            )
            
            db.invoices[invoice_id] = invoice
            
            result = {
                "success": True,
                "invoice_info": {
                    "invoice_id": invoice_id,
                    "order_id": order_id,
                    "invoice_type": invoice_type,
                    "title": invoice_title,
                    "amount": order.amount,
                    "status": "开具中",
                    "estimated_time": "24小时内完成"
                }
            }
            
            if tax_number:
                result["invoice_info"]["tax_number"] = tax_number
            
            logger.info(f"发票开具成功: {invoice_id}")
            return result
            
        except Exception as e:
            logger.error(f"开具发票失败 {order_id}: {str(e)}")
            return {
                "success": False,
                "error": f"开具发票时发生错误: {str(e)}",
                "order_id": order_id
            }


class ToolManager:
    """工具管理器，支持动态加载和热更新"""
    
    def __init__(self):
        self.tools = {}
        self.tool_configs = AVAILABLE_TOOLS.copy()
        self._load_tools()
    
    def _load_tools(self):
        """加载工具"""
        self.tools = {
            "query_order": BusinessTools.query_order,
            "apply_refund": BusinessTools.apply_refund,
            "issue_invoice": BusinessTools.issue_invoice
        }
        logger.info(f"已加载 {len(self.tools)} 个工具")
    
    def get_tool(self, tool_name: str):
        """获取工具函数"""
        return self.tools.get(tool_name)
    
    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """获取工具配置"""
        return self.tool_configs.get(tool_name)
    
    def list_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tools.keys())
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """调用工具"""
        try:
            tool_func = self.get_tool(tool_name)
            if not tool_func:
                return {
                    "success": False,
                    "error": f"工具 {tool_name} 不存在"
                }
            
            # 调用工具函数
            result = tool_func(**kwargs)
            logger.info(f"工具调用成功: {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"工具调用失败 {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": f"工具调用错误: {str(e)}"
            }
    
    def reload_tools(self):
        """重新加载工具（热更新）"""
        try:
            self._load_tools()
            logger.info("工具热更新成功")
            return True
        except Exception as e:
            logger.error(f"工具热更新失败: {str(e)}")
            return False
    
    def add_tool(self, tool_name: str, tool_func, tool_config: ToolConfig):
        """动态添加工具"""
        self.tools[tool_name] = tool_func
        self.tool_configs[tool_name] = tool_config
        logger.info(f"已添加工具: {tool_name}")
    
    def remove_tool(self, tool_name: str):
        """移除工具"""
        if tool_name in self.tools:
            del self.tools[tool_name]
        if tool_name in self.tool_configs:
            del self.tool_configs[tool_name]
        logger.info(f"已移除工具: {tool_name}")


# 全局工具管理器实例
tool_manager = ToolManager()