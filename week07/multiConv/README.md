# 多轮对话与工具调用系统

基于 LangGraph 的智能多轮对话系统，支持订单查询、退款申请、发票开具等业务场景，具备模型热更新和生产部署能力。

## 🌟 功能特性

### 核心功能
- **多轮对话状态管理**: 使用 LangGraph 构建状态图，支持复杂的对话流程
- **智能工具调用**: 自动识别用户意图并调用相应业务工具
- **会话管理**: 支持多用户并发对话，自动清理过期会话

### 业务工具
- **订单查询**: 查询订单状态和物流信息
- **退款申请**: 支持全额或部分退款申请
- **发票开具**: 支持个人和企业发票开具

### 高级特性
- **模型热更新**: 支持模型配置的无缝更新
- **插件热重载**: 支持业务插件的动态加载和卸载
- **健康检查**: 完善的系统健康监控
- **生产部署**: 支持高并发和容错处理

## 📁 项目结构

```
week07/multiConv/
├── __init__.py          # 包初始化
├── config.py            # 配置管理
├── tools.py             # 业务工具实现
├── graph.py             # LangGraph 状态图
├── service.py           # 多轮对话服务
├── hot_update.py        # 热更新功能
├── api.py              # FastAPI 接口
├── main.py             # 演示程序
├── test_suite.py       # 自动化测试
└── README.md           # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 进入项目根目录
cd /Users/hhhhhh/作业/homework

# 安装依赖
uv sync
```

### 2. 环境配置

创建 `.env` 文件：

```env
# OpenAI API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-ai/DeepSeek-V3.1

# 应用配置
DEBUG=false
HOST=0.0.0.0
PORT=8000

# 热更新配置
ENABLE_HOT_RELOAD=true
```

### 3. 启动方式

#### 方式一：演示模式
```bash
python -m week07.multiConv.main
```

#### 方式二：API 服务模式
```bash
python -m week07.multiConv.api
```

#### 方式三：测试模式
```bash
# 先启动 API 服务
python -m week07.multiConv.api

# 然后在另一个终端运行测试
python -m week07.multiConv.test_suite
```

## 💬 使用示例

### 多轮对话示例

#### 订单查询流程
```
用户: "查订单"
AI: "请提供您的订单号，我来帮您查询。"
用户: "ORD001"
AI: "查询到您的订单信息：
📦 订单号：ORD001
📋 订单状态：已发货
💰 订单金额：￥299.0
🕐 下单时间：2024-01-26 10:30:00
📄 商品清单：智能手机
🚚 物流信息：
   快递公司：顺丰快递
   运单号：SF123456789
   配送状态：运输中"
```

#### 退款申请流程
```
用户: "我要退款"
AI: "请提供订单号和退款原因，我来帮您申请退款。"
用户: "订单ORD002，质量问题"
AI: "退款申请已提交成功！
🆔 退款单号：REF20240126143000
📦 订单号：ORD002
💰 退款金额：￥159.0
📝 退款原因：质量问题
📊 处理状态：处理中
⏰ 预计到账：3-7个工作日"
```

#### 发票开具流程
```
用户: "开发票，订单ORD001，个人发票，抬头张三"
AI: "发票开具申请已提交！
🧾 发票号：INV20240126143030
📦 订单号：ORD001
🏷️ 发票类型：个人
🏢 发票抬头：张三
💰 开票金额：￥299.0
📊 开票状态：开具中
⏰ 预计完成：24小时内完成"
```

## 🔧 API 接口

### 健康检查
```http
GET /health
```

### 对话管理
```http
POST /conversation/start
POST /conversation/message
POST /conversation/history
POST /conversation/end
```

### 系统管理
```http
GET /stats
POST /admin/hot-update
GET /admin/hot-update/status
GET /admin/tools
POST /admin/tools/reload
```

### API 使用示例

```python
import requests

# 开始对话
response = requests.post("http://localhost:8000/conversation/start", 
                        json={"user_id": "user123"})
session_id = response.json()["session_id"]

# 发送消息
response = requests.post("http://localhost:8000/conversation/message",
                        json={"session_id": session_id, "message": "查询订单ORD001"})
print(response.json()["response"])
```

## 🔄 热更新功能

### 模型热更新
支持在不重启服务的情况下更新模型配置：

```bash
curl -X POST "http://localhost:8000/admin/hot-update" \
  -H "Content-Type: application/json" \
  -d '{"action": "reload_model", "target": "/path/to/model_config.json"}'
```

### 插件热重载
支持动态加载新的业务插件：

```bash
curl -X POST "http://localhost:8000/admin/hot-update" \
  -H "Content-Type: application/json" \
  -d '{"action": "reload_all"}'
```

### 旧会话稳定性
热更新后，现有的用户会话不会受到影响，保证服务的连续性。

## 🧪 自动化测试

系统包含完整的测试套件，覆盖以下场景：

### 发票开具测试
- ✅ 个人发票开具
- ✅ 企业发票开具  
- ✅ 缺少订单号处理
- ✅ 缺少发票信息处理
- ✅ 无效订单号处理
- ✅ 多轮对话开票流程

### 热更新测试
- ✅ 热更新状态获取
- ✅ 插件重载功能
- ✅ 工具重载功能
- ✅ 旧会话稳定性验证

### 运行测试
```bash
# 确保服务在运行
python -m week07.multiConv.api

# 运行测试套件
python -m week07.multiConv.test_suite
```

测试报告会自动保存为 `test_report_YYYYMMDD_HHMMSS.json`。

## 📊 系统监控

### 健康检查指标
- 会话管理器状态
- 对话图状态  
- 热更新器状态
- 工具管理器状态
- 活跃会话数量

### 性能统计
- 总对话数
- 成功率
- 工具调用次数
- 服务运行时间
- 会话统计信息

## 🏗️ 生产部署

### Docker 部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "week07.multiConv.api"]
```

### 环境变量配置
```env
# 生产环境配置
DEBUG=false
HOST=0.0.0.0
PORT=8000
SESSION_TIMEOUT_MINUTES=60
MAX_CONVERSATION_HISTORY=50
ENABLE_HOT_RELOAD=true
```

### 负载均衡配置
```nginx
upstream multiconv_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://multiconv_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🛠️ 扩展开发

### 添加新业务工具

1. 在 `tools.py` 中实现业务逻辑：
```python
@staticmethod
def new_business_tool(param1: str, param2: int) -> Dict[str, Any]:
    """新业务工具"""
    # 实现业务逻辑
    return {"success": True, "result": "处理结果"}
```

2. 在 `config.py` 中注册工具配置：
```python
AVAILABLE_TOOLS["new_tool"] = ToolConfig(
    name="new_tool",
    description="新业务工具描述",
    parameters={...}
)
```

3. 在 `graph.py` 中添加处理逻辑。

### 添加新对话状态

1. 在 `config.py` 中定义新状态：
```python
class ConversationState:
    NEW_STATE = "new_state"
```

2. 在 `graph.py` 中添加状态处理节点和路由逻辑。

## 📝 开发规范

### 代码风格
- 使用 Python 3.9+ 语法
- 遵循 PEP 8 代码规范
- 使用类型提示
- 完善的文档字符串

### 错误处理
- 统一的异常处理机制
- 详细的错误日志记录
- 优雅的降级处理

### 测试要求
- 单元测试覆盖率 > 80%
- 集成测试覆盖主要业务流程
- 性能测试验证并发能力

## 🐛 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口是否被占用
   - 验证环境变量配置
   - 查看错误日志

2. **对话响应慢**
   - 检查 OpenAI API 连接
   - 优化提示词长度
   - 增加超时设置

3. **热更新失败**
   - 检查文件权限
   - 验证插件语法
   - 查看更新日志

### 日志查看
```bash
# 查看服务日志
tail -f logs/multiconv.log

# 查看热更新日志
tail -f logs/hot_update.log
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送分支 (`git push origin feature/new-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [LangChain](https://langchain.com/) - 提供强大的 LLM 应用框架
- [LangGraph](https://langgraph.com/) - 提供状态图管理能力
- [FastAPI](https://fastapi.tiangolo.com/) - 提供高性能 Web 框架
- [Rich](https://rich.readthedocs.io/) - 提供美观的终端输出

---

🎉 **多轮对话与工具调用系统已完成构建！** 

该系统完整实现了您要求的所有功能：
- ✅ 多轮对话状态管理
- ✅ 订单查询、退款申请、发票开具
- ✅ 工具自动调用
- ✅ 模型与插件热更新
- ✅ 健康检查接口
- ✅ 自动化测试脚本
- ✅ 生产部署支持