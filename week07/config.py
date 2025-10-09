"""
配置文件
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 模型配置
MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

# 订单数据（模拟数据库）
MOCK_ORDERS = {
    "ORD001": {
        "order_id": "ORD001",
        "status": "已发货",
        "items": ["iPhone 15 Pro", "AirPods Pro"],
        "total": 9998.0,
        "tracking_number": "SF1234567890",
        "logistics": {
            "carrier": "顺丰快递",
            "current_location": "上海分拣中心",
            "estimated_delivery": "2024-01-15"
        }
    },
    "ORD002": {
        "order_id": "ORD002", 
        "status": "配送中",
        "items": ["MacBook Air"],
        "total": 7999.0,
        "tracking_number": "YT9876543210",
        "logistics": {
            "carrier": "圆通速递",
            "current_location": "北京朝阳区派送中",
            "estimated_delivery": "2024-01-13"
        }
    },
    "ORD003": {
        "order_id": "ORD003",
        "status": "已完成",
        "items": ["iPad Pro", "Apple Pencil"],
        "total": 6998.0,
        "tracking_number": "ZTO5555666677",
        "logistics": {
            "carrier": "中通快递",
            "current_location": "已签收",
            "estimated_delivery": "2024-01-10"
        }
    }
}