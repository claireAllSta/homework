# 基于 MCP 协议的多代理文章自动编写系统

> **项目位置**: `week05-homework/multi-agent`  
> **运行方式**: 使用homework项目的统一uv环境  
> **依赖管理**: 集成在homework的pyproject.toml中

## 项目简介

这是一个基于 MCP (Model Context Protocol) 协议的多代理协作系统，能够自动完成文章写作任务。系统包含四个专业化代理，按顺序协作完成从研究到最终成稿的完整流程。

## 系统架构

### 四个专业化代理

1. **研究代理 (Research Agent)**
   - 使用搜索工具收集相关信息
   - 输出结构化的研究资料
   - 分析用户需求和研究范围

2. **撰写代理 (Writing Agent)**
   - 基于研究结果生成文章初稿
   - 支持调整文章风格和长度
   - 适应不同受众和写作要求

3. **审核代理 (Review Agent)**
   - 检查内容质量和逻辑一致性
   - 提供详细的修改建议
   - 多维度质量评估

4. **润色代理 (Polishing Agent)**
   - 优化语言表达和文章结构
   - 确保风格一致性
   - 完善格式和细节

### 核心特性

- **MCP 协议支持**: 实现代理间无缝数据传递
- **三级重试策略**: 
  - 一级：相同代理重新执行（最多 2 次）
  - 二级：切换至备用代理执行
  - 三级：向用户请求补充信息
- **实时进度显示**: 终端展示完整协作过程
- **完整执行记录**: 生成详细的过程文档

## 安装与使用

### 环境要求

- Python 3.9+
- uv (项目依赖管理工具)
- 项目使用homework的统一依赖管理

### 运行方式

本项目是homework项目的一部分，使用homework项目的统一环境和依赖管理。

#### 方式一：在homework根目录运行
```bash
# 在homework根目录下
cd /Users/hhhhhh/作业/homework
uv run python week05-homework/multi-agent/main.py "帮我写一篇关于 AI Agent 的文章"
```

#### 方式二：切换到项目目录运行
```bash
# 切换到multi-agent目录
cd /Users/hhhhhh/作业/homework/week05-homework/multi-agent
uv run python main.py "帮我写一篇关于 AI Agent 的文章"
```

#### 交互模式
```bash
# 在homework根目录
uv run python week05-homework/multi-agent/main.py
# 或在multi-agent目录
uv run python main.py
# 然后输入您的文章主题
```

### 输出文件

系统会在当前工作目录的 `output/` 目录下生成以下文件：
- `article_TIMESTAMP.md` - 最终文章
- `execution_report_TIMESTAMP.md` - 执行报告
- `article_result_TIMESTAMP.json` - 完整结果数据

### 依赖管理说明

本项目作为homework项目的一部分，使用统一的依赖管理：
- 所有依赖都定义在homework根目录的 `pyproject.toml` 中
- 使用 `uv` 作为包管理工具，自动处理虚拟环境和依赖安装
- 无需单独安装依赖，uv会自动解决

## 目录结构

```
multi-agent/
├── main.py                 # 主入口文件
├── core/                   # 核心模块
│   ├── mcp_context.py     # MCP 上下文管理
│   ├── workflow_manager.py # 工作流管理
│   └── retry_manager.py   # 重试管理
├── agents/                 # 代理模块
│   ├── research_agent.py  # 研究代理
│   ├── writing_agent.py   # 撰写代理
│   ├── review_agent.py    # 审核代理
│   └── polishing_agent.py # 润色代理
├── utils/                  # 工具模块
│   ├── logger.py          # 日志工具
│   └── output_formatter.py # 输出格式化
├── output/                 # 输出目录
├── logs/                   # 日志目录
├── requirements.txt        # 依赖文件
└── report.md              # 示例输出报告
```

## 技术特点

### MCP 协议实现
- 统一的上下文管理
- 代理间数据无缝传递
- 完整的执行状态追踪

### 智能重试机制
- 多级重试策略
- 自动错误恢复
- 详细的异常日志

### 质量保证
- 多维度内容评估
- 渐进式质量提升
- 自动化修改建议

## 示例输出

参考 `report.md` 文件查看完整的系统执行示例，包括：
- 详细的执行过程记录
- 各代理的协作表现
- 质量评估和改进建议
- 最终文章质量分析

## 扩展功能

### 已实现的扩展
- ✅ 基于 MCP 上下文的自动重试
- ✅ 三级重试策略
- ✅ 完整的异常处理日志
- ✅ 实时协作过程展示

### 未来改进
- 支持更多文章类型和风格
- 增加多语言支持
- 优化代理执行效率
- 增强用户交互功能

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 许可证

MIT License