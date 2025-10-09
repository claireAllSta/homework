"""
使用LangGraph构建多轮对话流程
"""
from typing import TypedDict, List, Dict, Any, Optional
import json
import re
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from tools import AVAILABLE_TOOLS, query_order, apply_refund, format_order_info


class ConversationState(TypedDict):
    """对话状态"""
    messages: List[BaseMessage]
    user_intent: Optional[str]
    required_params: Dict[str, Any]
    tools_called: List[str]
    current_step: str
    conversation_complete: bool


class OrderDialogueGraph:
    """订单对话图"""
    
    def __init__(self):
        """初始化对话图"""
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.1
        )
        
        # 创建状态图
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """创建对话状态图"""
        # 定义状态图
        workflow = StateGraph(ConversationState)
        
        # 添加节点
        workflow.add_node("intent_recognition", self._recognize_intent)
        workflow.add_node("collect_parameters", self._collect_parameters)
        workflow.add_node("execute_tool", self._execute_tool)
        workflow.add_node("generate_response", self._generate_response)
        
        # 设置入口点
        workflow.set_entry_point("intent_recognition")
        
        # 添加边和条件路由
        workflow.add_conditional_edges(
            "intent_recognition",
            self._should_collect_params,
            {
                "collect": "collect_parameters",
                "execute": "execute_tool",
                "respond": "generate_response"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_parameters", 
            self._params_complete,
            {
                "complete": "execute_tool",
                "incomplete": "generate_response"
            }
        )
        
        workflow.add_edge("execute_tool", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _recognize_intent(self, state: ConversationState) -> ConversationState:
        """识别用户意图"""
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # 简单的意图识别规则
        if any(keyword in last_message for keyword in ["查订单", "订单查询", "查看订单", "订单状态"]):
            intent = "query_order"
            required_params = {"order_id": None}
        elif any(keyword in last_message for keyword in ["退款", "申请退款", "要退货"]):
            intent = "apply_refund"
            required_params = {"order_id": None, "reason": None}
        else:
            intent = "general"
            required_params = {}
        
        state["user_intent"] = intent
        state["required_params"] = required_params
        state["current_step"] = "intent_recognized"
        
        return state
    
    def _should_collect_params(self, state: ConversationState) -> str:
        """判断是否需要收集参数"""
        intent = state["user_intent"]
        
        if intent == "general":
            return "respond"
        
        # 检查是否已有必要参数
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        if intent == "query_order":
            # 尝试从消息中提取订单号
            order_pattern = r'(ORD\d+)'
            match = re.search(order_pattern, last_message.upper())
            if match:
                state["required_params"]["order_id"] = match.group(1)
                return "execute"
            else:
                return "collect"
        
        elif intent == "apply_refund":
            order_pattern = r'(ORD\d+)'
            match = re.search(order_pattern, last_message.upper())
            if match:
                state["required_params"]["order_id"] = match.group(1)
                return "execute"
            else:
                return "collect"
        
        return "collect"
    
    def _collect_parameters(self, state: ConversationState) -> ConversationState:
        """收集必要参数"""
        intent = state["user_intent"]
        params = state["required_params"]
        
        if intent == "query_order" and not params.get("order_id"):
            state["current_step"] = "collecting_order_id"
        elif intent == "apply_refund" and not params.get("order_id"):
            state["current_step"] = "collecting_refund_order_id"
        
        return state
    
    def _params_complete(self, state: ConversationState) -> str:
        """检查参数是否完整"""
        intent = state["user_intent"]
        params = state["required_params"]
        
        if intent == "query_order":
            return "complete" if params.get("order_id") else "incomplete"
        elif intent == "apply_refund":
            return "complete" if params.get("order_id") else "incomplete"
        
        return "complete"
    
    def _execute_tool(self, state: ConversationState) -> ConversationState:
        """执行工具调用"""
        intent = state["user_intent"]
        params = state["required_params"]
        
        if intent == "query_order" and params.get("order_id"):
            result = query_order(params["order_id"])
            state["tool_result"] = result
            state["tools_called"].append("query_order")
            
        elif intent == "apply_refund" and params.get("order_id"):
            result = apply_refund(params["order_id"], params.get("reason", ""))
            state["tool_result"] = result
            state["tools_called"].append("apply_refund")
        
        state["current_step"] = "tool_executed"
        return state
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        """生成回复"""
        current_step = state["current_step"]
        intent = state["user_intent"]
        
        if current_step == "collecting_order_id":
            response = "请提供您的订单号（格式如：ORD001）"
            
        elif current_step == "collecting_refund_order_id":
            response = "请提供要申请退款的订单号（格式如：ORD001）"
            
        elif current_step == "tool_executed":
            tool_result = state.get("tool_result", {})
            
            if intent == "query_order":
                response = format_order_info(tool_result)
            elif intent == "apply_refund":
                if tool_result.get("success"):
                    response = f"✅ {tool_result['message']}"
                else:
                    response = f"❌ {tool_result['message']}"
            else:
                response = "操作完成"
                
        else:
            # 一般对话
            response = "您好！我可以帮您查询订单状态或申请退款。请告诉我您需要什么帮助？"
        
        # 添加AI回复到消息历史
        state["messages"].append(AIMessage(content=response))
        state["conversation_complete"] = True
        
        return state
    
    def process_message(self, user_input: str, conversation_history: List[BaseMessage] = None) -> str:
        """
        处理用户消息
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            
        Returns:
            AI回复
        """
        # 初始化状态
        if conversation_history is None:
            conversation_history = []
        
        # 添加用户消息
        conversation_history.append(HumanMessage(content=user_input))
        
        initial_state = ConversationState(
            messages=conversation_history,
            user_intent=None,
            required_params={},
            tools_called=[],
            current_step="start",
            conversation_complete=False
        )
        
        # 运行图
        result = self.graph.invoke(initial_state)
        
        # 返回最后的AI回复
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        return ai_messages[-1].content if ai_messages else "抱歉，我无法理解您的需求。"


class DialogueManager:
    """对话管理器"""
    
    def __init__(self):
        """初始化对话管理器"""
        self.graph = OrderDialogueGraph()
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        """
        进行对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI回复
        """
        response = self.graph.process_message(user_input, self.conversation_history.copy())
        
        # 更新对话历史
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=response))
        
        # 保持对话历史在合理长度内
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-8:]
        
        return response
    
    def reset(self):
        """重置对话历史"""
        self.conversation_history = []