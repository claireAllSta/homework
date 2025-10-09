"""
基于 LangGraph 的多轮对话状态管理图
"""
import json
import re
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from operator import add

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from loguru import logger

from .config import settings, ConversationState, MessageType
from .tools import tool_manager


class ConversationGraphState(TypedDict):
    """对话图状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_state: str
    session_id: str
    user_intent: Optional[str]
    extracted_info: Dict[str, Any]
    tool_calls_pending: List[Dict[str, Any]]
    conversation_context: Dict[str, Any]


class ConversationGraph:
    """多轮对话状态管理图"""
    
    def __init__(self):
        self.llm = self._create_llm()
        self.graph = self._create_graph()
    
    def _create_llm(self) -> ChatOpenAI:
        """创建LLM实例"""
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.1,
        }
        
        if settings.openai_api_key:
            llm_kwargs["openai_api_key"] = settings.openai_api_key
        
        if settings.openai_api_base:
            llm_kwargs["openai_api_base"] = settings.openai_api_base
        
        return ChatOpenAI(**llm_kwargs)
    
    def _create_graph(self) -> StateGraph:
        """创建对话状态图"""
        workflow = StateGraph(ConversationGraphState)
        
        # 添加节点
        workflow.add_node("intent_analysis", self._analyze_intent)
        workflow.add_node("extract_info", self._extract_information)
        workflow.add_node("validate_info", self._validate_information)
        workflow.add_node("call_tools", self._call_tools)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("ask_clarification", self._ask_clarification)
        workflow.add_node("handle_error", self._handle_error)
        
        # 设置入口点
        workflow.set_entry_point("intent_analysis")
        
        # 添加边
        workflow.add_conditional_edges(
            "intent_analysis",
            self._route_after_intent,
            {
                "extract_info": "extract_info",
                "generate_response": "generate_response",
                "ask_clarification": "ask_clarification"
            }
        )
        
        workflow.add_conditional_edges(
            "extract_info",
            self._route_after_extraction,
            {
                "validate_info": "validate_info",
                "ask_clarification": "ask_clarification"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_info",
            self._route_after_validation,
            {
                "call_tools": "call_tools",
                "ask_clarification": "ask_clarification",
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_edge("call_tools", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("ask_clarification", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _analyze_intent(self, state: ConversationGraphState) -> ConversationGraphState:
        """分析用户意图"""
        try:
            logger.info("分析用户意图")
            
            messages = state["messages"]
            if not messages:
                return state
            
            last_message = messages[-1]
            user_input = last_message.content
            
            # 意图分析提示词
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个智能客服助手，专门分析用户意图。请分析用户的输入，识别用户想要执行的操作。

可识别的意图类型：
1. query_order - 查询订单（用户提到"查订单"、"订单状态"、"物流信息"等）
2. apply_refund - 申请退款（用户提到"退款"、"退货"、"不满意"等）
3. issue_invoice - 开具发票（用户提到"发票"、"开票"、"报销"等）
4. general_chat - 一般聊天（问候、咨询等）

请以JSON格式输出分析结果：
{{
    "intent": "意图类型",
    "confidence": 0.8,
    "entities": {{"实体名": "实体值"}},
    "clarification_needed": false
}}

用户输入：{user_input}"""),
            ])
            
            response = self.llm.invoke(intent_prompt.format_messages(user_input=user_input))
            
            # 解析意图分析结果
            try:
                intent_result = json.loads(response.content)
                state["user_intent"] = intent_result.get("intent")
                state["extracted_info"] = intent_result.get("entities", {})
                state["conversation_context"]["confidence"] = intent_result.get("confidence", 0.0)
                
                logger.info(f"识别意图: {intent_result.get('intent')}")
                
            except json.JSONDecodeError:
                logger.warning("意图分析结果解析失败，使用默认处理")
                state["user_intent"] = "general_chat"
                state["extracted_info"] = {}
            
            return state
            
        except Exception as e:
            logger.error(f"意图分析失败: {str(e)}")
            state["user_intent"] = "general_chat"
            state["current_state"] = ConversationState.ERROR
            return state
    
    def _extract_information(self, state: ConversationGraphState) -> ConversationGraphState:
        """提取信息"""
        try:
            logger.info("提取所需信息")
            
            intent = state["user_intent"]
            messages = state["messages"]
            last_message = messages[-1].content
            
            # 根据意图提取相应信息
            if intent == "query_order":
                order_id = self._extract_order_id(last_message)
                if order_id:
                    state["extracted_info"]["order_id"] = order_id
                    state["current_state"] = ConversationState.TOOL_CALLING
                else:
                    state["current_state"] = ConversationState.WAITING_ORDER_ID
                    
            elif intent == "apply_refund":
                order_id = self._extract_order_id(last_message)
                reason = self._extract_refund_reason(last_message)
                
                state["extracted_info"]["order_id"] = order_id
                state["extracted_info"]["reason"] = reason
                
                if order_id and reason:
                    state["current_state"] = ConversationState.TOOL_CALLING
                elif order_id and not reason:
                    state["current_state"] = ConversationState.WAITING_REFUND_REASON
                else:
                    state["current_state"] = ConversationState.WAITING_ORDER_ID
                    
            elif intent == "issue_invoice":
                invoice_info = self._extract_invoice_info(last_message)
                state["extracted_info"].update(invoice_info)
                
                required_fields = ["order_id", "invoice_type", "invoice_title"]
                missing_fields = [field for field in required_fields 
                                if field not in state["extracted_info"] or not state["extracted_info"][field]]
                
                if not missing_fields:
                    state["current_state"] = ConversationState.TOOL_CALLING
                else:
                    state["current_state"] = ConversationState.WAITING_INVOICE_INFO
                    state["conversation_context"]["missing_fields"] = missing_fields
            
            return state
            
        except Exception as e:
            logger.error(f"信息提取失败: {str(e)}")
            state["current_state"] = ConversationState.ERROR
            return state
    
    def _validate_information(self, state: ConversationGraphState) -> ConversationGraphState:
        """验证信息完整性"""
        try:
            logger.info("验证信息完整性")
            
            intent = state["user_intent"]
            extracted_info = state["extracted_info"]
            
            if intent == "query_order":
                if "order_id" in extracted_info and extracted_info["order_id"]:
                    state["current_state"] = ConversationState.TOOL_CALLING
                else:
                    state["current_state"] = ConversationState.WAITING_ORDER_ID
                    
            elif intent == "apply_refund":
                if ("order_id" in extracted_info and extracted_info["order_id"] and
                    "reason" in extracted_info and extracted_info["reason"]):
                    state["current_state"] = ConversationState.TOOL_CALLING
                else:
                    missing = []
                    if not extracted_info.get("order_id"):
                        missing.append("订单号")
                    if not extracted_info.get("reason"):
                        missing.append("退款原因")
                    state["conversation_context"]["missing_info"] = missing
                    state["current_state"] = ConversationState.WAITING_REFUND_REASON
                    
            elif intent == "issue_invoice":
                required_fields = {
                    "order_id": "订单号",
                    "invoice_type": "发票类型",
                    "invoice_title": "发票抬头"
                }
                
                missing = []
                for field, desc in required_fields.items():
                    if not extracted_info.get(field):
                        missing.append(desc)
                
                # 企业发票需要税号
                if (extracted_info.get("invoice_type") == "企业" and 
                    not extracted_info.get("tax_number")):
                    missing.append("税号")
                
                if not missing:
                    state["current_state"] = ConversationState.TOOL_CALLING
                else:
                    state["conversation_context"]["missing_info"] = missing
                    state["current_state"] = ConversationState.WAITING_INVOICE_INFO
            
            return state
            
        except Exception as e:
            logger.error(f"信息验证失败: {str(e)}")
            state["current_state"] = ConversationState.ERROR
            return state
    
    def _call_tools(self, state: ConversationGraphState) -> ConversationGraphState:
        """调用工具"""
        try:
            logger.info("调用业务工具")
            
            intent = state["user_intent"]
            extracted_info = state["extracted_info"]
            
            if intent == "query_order":
                result = tool_manager.call_tool("query_order", order_id=extracted_info["order_id"])
                
            elif intent == "apply_refund":
                kwargs = {
                    "order_id": extracted_info["order_id"],
                    "reason": extracted_info["reason"]
                }
                if "amount" in extracted_info:
                    kwargs["amount"] = extracted_info["amount"]
                result = tool_manager.call_tool("apply_refund", **kwargs)
                
            elif intent == "issue_invoice":
                kwargs = {
                    "order_id": extracted_info["order_id"],
                    "invoice_type": extracted_info["invoice_type"],
                    "invoice_title": extracted_info["invoice_title"]
                }
                if "tax_number" in extracted_info:
                    kwargs["tax_number"] = extracted_info["tax_number"]
                result = tool_manager.call_tool("issue_invoice", **kwargs)
                
            else:
                result = {"success": False, "error": "未知的意图类型"}
            
            # 保存工具调用结果
            state["conversation_context"]["tool_result"] = result
            
            if result.get("success"):
                state["current_state"] = ConversationState.COMPLETED
            else:
                state["current_state"] = ConversationState.ERROR
            
            return state
            
        except Exception as e:
            logger.error(f"工具调用失败: {str(e)}")
            state["current_state"] = ConversationState.ERROR
            state["conversation_context"]["error"] = str(e)
            return state
    
    def _generate_response(self, state: ConversationGraphState) -> ConversationGraphState:
        """生成响应"""
        try:
            logger.info("生成回复")
            
            current_state = state["current_state"]
            intent = state["user_intent"]
            tool_result = state["conversation_context"].get("tool_result")
            
            if current_state == ConversationState.COMPLETED and tool_result:
                if intent == "query_order" and tool_result.get("success"):
                    order_info = tool_result["order_info"]
                    response_text = f"""查询到您的订单信息：
📦 订单号：{order_info['order_id']}
📋 订单状态：{order_info['status']}
💰 订单金额：￥{order_info['amount']}
🕐 下单时间：{order_info['create_time']}
📄 商品清单：{', '.join([item['name'] for item in order_info['items']])}"""
                    
                    if "shipping_info" in tool_result:
                        shipping = tool_result["shipping_info"]
                        response_text += f"""
🚚 物流信息：
   快递公司：{shipping['courier']}
   运单号：{shipping['tracking_number']}
   配送状态：{shipping['status']}"""
                
                elif intent == "apply_refund" and tool_result.get("success"):
                    refund_info = tool_result["refund_info"]
                    response_text = f"""退款申请已提交成功！
🆔 退款单号：{refund_info['refund_id']}
📦 订单号：{refund_info['order_id']}
💰 退款金额：￥{refund_info['amount']}
📝 退款原因：{refund_info['reason']}
📊 处理状态：{refund_info['status']}
⏰ 预计到账：{refund_info['estimated_days']}"""
                
                elif intent == "issue_invoice" and tool_result.get("success"):
                    invoice_info = tool_result["invoice_info"]
                    response_text = f"""发票开具申请已提交！
🧾 发票号：{invoice_info['invoice_id']}
📦 订单号：{invoice_info['order_id']}
🏷️ 发票类型：{invoice_info['invoice_type']}
🏢 发票抬头：{invoice_info['title']}
💰 开票金额：￥{invoice_info['amount']}
📊 开票状态：{invoice_info['status']}
⏰ 预计完成：{invoice_info['estimated_time']}"""
                    
                    if "tax_number" in invoice_info:
                        response_text += f"\n🔢 税号：{invoice_info['tax_number']}"
                
                else:
                    response_text = "操作完成！"
                    
            elif intent == "general_chat":
                response_text = "您好！我是智能客服助手，可以帮您查询订单、申请退款、开具发票等。请告诉我您需要什么帮助？"
                
            else:
                response_text = "抱歉，我没有理解您的需求，请重新描述一下？"
            
            # 添加AI响应到消息历史
            state["messages"].append(AIMessage(content=response_text))
            
            return state
            
        except Exception as e:
            logger.error(f"响应生成失败: {str(e)}")
            error_msg = "抱歉，系统出现了问题，请稍后再试。"
            state["messages"].append(AIMessage(content=error_msg))
            return state
    
    def _ask_clarification(self, state: ConversationGraphState) -> ConversationGraphState:
        """询问澄清信息"""
        try:
            logger.info("请求澄清信息")
            
            current_state = state["current_state"]
            intent = state["user_intent"]
            
            if current_state == ConversationState.WAITING_ORDER_ID:
                response_text = "请提供您的订单号，我来帮您查询。"
                
            elif current_state == ConversationState.WAITING_REFUND_REASON:
                missing_info = state["conversation_context"].get("missing_info", [])
                if "订单号" in missing_info and "退款原因" in missing_info:
                    response_text = "请提供订单号和退款原因，我来帮您申请退款。"
                elif "退款原因" in missing_info:
                    response_text = "请说明退款原因。"
                else:
                    response_text = "请提供退款相关信息。"
                    
            elif current_state == ConversationState.WAITING_INVOICE_INFO:
                missing_info = state["conversation_context"].get("missing_info", [])
                missing_str = "、".join(missing_info)
                response_text = f"请提供以下发票信息：{missing_str}"
                
            else:
                response_text = "请提供更多信息，我好为您服务。"
            
            # 添加澄清询问到消息历史
            state["messages"].append(AIMessage(content=response_text))
            
            return state
            
        except Exception as e:
            logger.error(f"澄清询问失败: {str(e)}")
            error_msg = "请提供更详细的信息。"
            state["messages"].append(AIMessage(content=error_msg))
            return state
    
    def _handle_error(self, state: ConversationGraphState) -> ConversationGraphState:
        """处理错误"""
        try:
            logger.info("处理错误状态")
            
            tool_result = state["conversation_context"].get("tool_result")
            error_msg = "抱歉，处理您的请求时出现了问题。"
            
            if tool_result and not tool_result.get("success"):
                error_msg = f"抱歉，{tool_result.get('error', '未知错误')}"
            
            state["messages"].append(AIMessage(content=error_msg))
            
            return state
            
        except Exception as e:
            logger.error(f"错误处理失败: {str(e)}")
            state["messages"].append(AIMessage(content="系统出现错误，请稍后再试。"))
            return state
    
    # 路由函数
    def _route_after_intent(self, state: ConversationGraphState) -> str:
        """意图分析后的路由"""
        intent = state["user_intent"]
        if intent in ["query_order", "apply_refund", "issue_invoice"]:
            return "extract_info"
        elif intent == "general_chat":
            return "generate_response"
        else:
            return "ask_clarification"
    
    def _route_after_extraction(self, state: ConversationGraphState) -> str:
        """信息提取后的路由"""
        current_state = state["current_state"]
        if current_state == ConversationState.TOOL_CALLING:
            return "validate_info"
        else:
            return "ask_clarification"
    
    def _route_after_validation(self, state: ConversationGraphState) -> str:
        """信息验证后的路由"""
        current_state = state["current_state"]
        if current_state == ConversationState.TOOL_CALLING:
            return "call_tools"
        elif current_state == ConversationState.ERROR:
            return "handle_error"
        else:
            return "ask_clarification"
    
    # 辅助函数
    def _extract_order_id(self, text: str) -> Optional[str]:
        """提取订单号"""
        # 匹配订单号模式
        patterns = [
            r'[A-Z]{3}\d{3}',  # ORD001格式
            r'订单号[：:]\s*([A-Za-z0-9]+)',
            r'订单[：:]\s*([A-Za-z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if pattern.startswith('[A-Z]'):
                    return match.group(0)
                else:
                    return match.group(1)
        return None
    
    def _extract_refund_reason(self, text: str) -> Optional[str]:
        """提取退款原因"""
        reason_keywords = ["质量问题", "不喜欢", "尺寸不合适", "商品描述不符", "收到货物损坏", "发错货"]
        
        for keyword in reason_keywords:
            if keyword in text:
                return keyword
        
        # 提取原因相关的句子
        reason_patterns = [
            r'原因[是：:]\s*([^。，,]+)',
            r'因为\s*([^。，,]+)',
            r'退款原因[：:]\s*([^。，,]+)'
        ]
        
        for pattern in reason_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_invoice_info(self, text: str) -> Dict[str, Any]:
        """提取发票信息"""
        info = {}
        
        # 提取订单号
        order_id = self._extract_order_id(text)
        if order_id:
            info["order_id"] = order_id
        
        # 提取发票类型
        if "个人" in text or "个人发票" in text:
            info["invoice_type"] = "个人"
        elif "企业" in text or "企业发票" in text or "公司" in text:
            info["invoice_type"] = "企业"
        
        # 提取发票抬头
        title_patterns = [
            r'抬头[：:]\s*([^。，,]+)',
            r'发票抬头[：:]\s*([^。，,]+)',
            r'公司名称[：:]\s*([^。，,]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                info["invoice_title"] = match.group(1).strip()
                break
        
        # 提取税号
        tax_patterns = [
            r'税号[：:]\s*([A-Za-z0-9]+)',
            r'纳税人识别号[：:]\s*([A-Za-z0-9]+)'
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text)
            if match:
                info["tax_number"] = match.group(1).strip()
                break
        
        return info
    
    def process_message(self, session_id: str, user_message: str, 
                       conversation_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """处理用户消息"""
        try:
            # 初始化状态
            initial_state = ConversationGraphState(
                messages=conversation_history or [],
                current_state=ConversationState.START,
                session_id=session_id,
                user_intent=None,
                extracted_info={},
                tool_calls_pending=[],
                conversation_context={}
            )
            
            # 添加用户消息
            initial_state["messages"].append(HumanMessage(content=user_message))
            
            # 运行图
            final_state = self.graph.invoke(initial_state)
            
            # 提取最后的AI响应
            ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
            response_text = ai_messages[-1].content if ai_messages else "抱歉，我无法处理您的请求。"
            
            return {
                "success": True,
                "response": response_text,
                "state": final_state["current_state"],
                "intent": final_state["user_intent"],
                "messages": final_state["messages"],
                "extracted_info": final_state["extracted_info"]
            }
            
        except Exception as e:
            logger.error(f"消息处理失败: {str(e)}")
            return {
                "success": False,
                "response": "抱歉，系统出现问题，请稍后再试。",
                "error": str(e)
            }