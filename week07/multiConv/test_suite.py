"""
自动化测试脚本
测试发票开具功能和热更新功能
"""
import asyncio
import requests
import json
import time
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import pytest
from loguru import logger

# 测试配置
TEST_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30


class TestClient:
    """测试客户端"""
    
    def __init__(self, base_url: str = TEST_BASE_URL):
        self.base_url = base_url
        self.session_id = None
    
    def start_conversation(self) -> Dict[str, Any]:
        """开始对话"""
        response = requests.post(
            f"{self.base_url}/conversation/start",
            json={"user_id": "test_user"},
            timeout=TEST_TIMEOUT
        )
        result = response.json()
        if result.get("success"):
            self.session_id = result["session_id"]
        return result
    
    def send_message(self, message: str) -> Dict[str, Any]:
        """发送消息"""
        if not self.session_id:
            raise ValueError("No active session")
        
        response = requests.post(
            f"{self.base_url}/conversation/message",
            json={"session_id": self.session_id, "message": message},
            timeout=TEST_TIMEOUT
        )
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        response = requests.get(f"{self.base_url}/health", timeout=TEST_TIMEOUT)
        return response.json()
    
    def get_hot_update_status(self) -> Dict[str, Any]:
        """获取热更新状态"""
        response = requests.get(
            f"{self.base_url}/admin/hot-update/status",
            timeout=TEST_TIMEOUT
        )
        return response.json()
    
    def trigger_hot_update(self, action: str, target: str = None) -> Dict[str, Any]:
        """触发热更新"""
        data = {"action": action}
        if target:
            data["target"] = target
        
        response = requests.post(
            f"{self.base_url}/admin/hot-update",
            json=data,
            timeout=TEST_TIMEOUT
        )
        return response.json()
    
    def end_conversation(self) -> Dict[str, Any]:
        """结束对话"""
        if not self.session_id:
            return {"success": True}
        
        response = requests.post(
            f"{self.base_url}/conversation/end",
            json={"session_id": self.session_id},
            timeout=TEST_TIMEOUT
        )
        result = response.json()
        self.session_id = None
        return result


class InvoiceTestSuite:
    """发票开具功能测试套件"""
    
    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """运行所有发票测试"""
        tests = [
            self.test_personal_invoice,
            self.test_company_invoice,
            self.test_missing_order_id,
            self.test_missing_invoice_info,
            self.test_invalid_order_id,
            self.test_multiple_step_invoice
        ]
        
        for test in tests:
            try:
                result = test()
                self.test_results.append(result)
                logger.info(f"测试 {test.__name__} 完成: {'通过' if result['passed'] else '失败'}")
            except Exception as e:
                logger.error(f"测试 {test.__name__} 异常: {str(e)}")
                self.test_results.append({
                    "test_name": test.__name__,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return self.test_results
    
    def test_personal_invoice(self) -> Dict[str, Any]:
        """测试个人发票开具"""
        logger.info("开始测试个人发票开具")
        
        # 开始新对话
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        # 发送开具个人发票请求
        message = "我要开发票，订单号ORD001，个人发票，抬头张三"
        response = self.client.send_message(message)
        
        # 验证响应
        success = (
            response["success"] and
            "发票开具申请已提交" in response.get("response", "") and
            response.get("intent") == "issue_invoice"
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_personal_invoice",
            "passed": success,
            "message": message,
            "response": response.get("response"),
            "intent": response.get("intent"),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_company_invoice(self) -> Dict[str, Any]:
        """测试企业发票开具"""
        logger.info("开始测试企业发票开具")
        
        # 开始新对话
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        # 发送开具企业发票请求
        message = "开企业发票，订单ORD002，公司名称：测试科技有限公司，税号：91110000123456789X"
        response = self.client.send_message(message)
        
        # 验证响应
        success = (
            response["success"] and
            "发票开具申请已提交" in response.get("response", "") and
            response.get("intent") == "issue_invoice"
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_company_invoice",
            "passed": success,
            "message": message,
            "response": response.get("response"),
            "intent": response.get("intent"),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_missing_order_id(self) -> Dict[str, Any]:
        """测试缺少订单号的情况"""
        logger.info("开始测试缺少订单号")
        
        # 开始新对话
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        # 发送缺少订单号的开票请求
        message = "我要开发票，个人发票，抬头李四"
        response = self.client.send_message(message)
        
        # 验证响应 - 应该询问订单号
        success = (
            response["success"] and
            ("订单号" in response.get("response", "") or "请提供" in response.get("response", ""))
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_missing_order_id",
            "passed": success,
            "message": message,
            "response": response.get("response"),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_missing_invoice_info(self) -> Dict[str, Any]:
        """测试缺少发票信息的情况"""
        logger.info("开始测试缺少发票信息")
        
        # 开始新对话
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        # 发送只有订单号的开票请求
        message = "订单ORD001开发票"
        response = self.client.send_message(message)
        
        # 验证响应 - 应该询问发票信息
        success = (
            response["success"] and
            ("发票" in response.get("response", "") and "信息" in response.get("response", ""))
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_missing_invoice_info",
            "passed": success,
            "message": message,
            "response": response.get("response"),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_invalid_order_id(self) -> Dict[str, Any]:
        """测试无效订单号"""
        logger.info("开始测试无效订单号")
        
        # 开始新对话
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        # 发送无效订单号的开票请求
        message = "订单INVALID123开发票，个人发票，抬头王五"
        response = self.client.send_message(message)
        
        # 验证响应 - 应该提示订单不存在
        success = (
            response["success"] and
            ("未找到" in response.get("response", "") or "不存在" in response.get("response", ""))
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_invalid_order_id",
            "passed": success,
            "message": message,
            "response": response.get("response"),
            "timestamp": datetime.now().isoformat()
        }
    
    def test_multiple_step_invoice(self) -> Dict[str, Any]:
        """测试多轮对话开具发票"""
        logger.info("开始测试多轮对话开具发票")
        
        # 开始新对话
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        steps = []
        
        # 第一步：提出开票需求
        message1 = "我要开发票"
        response1 = self.client.send_message(message1)
        steps.append({"message": message1, "response": response1.get("response")})
        
        # 第二步：提供订单号
        message2 = "订单号ORD001"
        response2 = self.client.send_message(message2)
        steps.append({"message": message2, "response": response2.get("response")})
        
        # 第三步：提供发票信息
        message3 = "个人发票，抬头赵六"
        response3 = self.client.send_message(message3)
        steps.append({"message": message3, "response": response3.get("response")})
        
        # 验证最终响应
        success = (
            response3["success"] and
            "发票开具申请已提交" in response3.get("response", "")
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_multiple_step_invoice",
            "passed": success,
            "steps": steps,
            "timestamp": datetime.now().isoformat()
        }


class HotUpdateTestSuite:
    """热更新功能测试套件"""
    
    def __init__(self, client: TestClient):
        self.client = client
        self.test_results = []
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """运行所有热更新测试"""
        tests = [
            self.test_hot_update_status,
            self.test_plugin_reload,
            self.test_tool_reload,
            self.test_old_session_stability
        ]
        
        for test in tests:
            try:
                result = test()
                self.test_results.append(result)
                logger.info(f"测试 {test.__name__} 完成: {'通过' if result['passed'] else '失败'}")
            except Exception as e:
                logger.error(f"测试 {test.__name__} 异常: {str(e)}")
                self.test_results.append({
                    "test_name": test.__name__,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return self.test_results
    
    def test_hot_update_status(self) -> Dict[str, Any]:
        """测试热更新状态获取"""
        logger.info("开始测试热更新状态获取")
        
        status = self.client.get_hot_update_status()
        
        # 验证状态包含必要字段
        success = (
            "status" in status and
            "recent_updates" in status and
            isinstance(status["status"], dict) and
            isinstance(status["recent_updates"], list)
        )
        
        return {
            "test_name": "test_hot_update_status",
            "passed": success,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_plugin_reload(self) -> Dict[str, Any]:
        """测试插件重载"""
        logger.info("开始测试插件重载")
        
        # 触发全量重载
        reload_result = self.client.trigger_hot_update("reload_all")
        
        # 等待重载完成
        time.sleep(2)
        
        # 检查重载后状态
        status = self.client.get_hot_update_status()
        
        success = (
            reload_result.get("success", False) and
            "status" in status
        )
        
        return {
            "test_name": "test_plugin_reload",
            "passed": success,
            "reload_result": reload_result,
            "post_status": status,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_tool_reload(self) -> Dict[str, Any]:
        """测试工具重载"""
        logger.info("开始测试工具重载")
        
        # 获取工具列表
        try:
            tools_response = requests.get(f"{TEST_BASE_URL}/admin/tools", timeout=TEST_TIMEOUT)
            tools_before = tools_response.json()
        except Exception as e:
            return {
                "test_name": "test_tool_reload",
                "passed": False,
                "error": f"获取工具列表失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # 触发工具重载
        try:
            reload_response = requests.post(
                f"{TEST_BASE_URL}/admin/tools/reload",
                timeout=TEST_TIMEOUT
            )
            reload_result = reload_response.json()
        except Exception as e:
            return {
                "test_name": "test_tool_reload",
                "passed": False,
                "error": f"工具重载失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        # 验证重载结果
        success = reload_result.get("success", False)
        
        return {
            "test_name": "test_tool_reload",
            "passed": success,
            "tools_before": tools_before,
            "reload_result": reload_result,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_old_session_stability(self) -> Dict[str, Any]:
        """测试热更新后旧会话稳定性"""
        logger.info("开始测试热更新后旧会话稳定性")
        
        # 创建会话并发送消息
        start_result = self.client.start_conversation()
        assert start_result["success"], "启动对话失败"
        
        # 发送第一条消息
        pre_update_response = self.client.send_message("你好")
        
        # 触发热更新
        reload_result = self.client.trigger_hot_update("reload_all")
        
        # 等待更新完成
        time.sleep(2)
        
        # 在同一会话中发送消息，验证会话仍然有效
        post_update_response = self.client.send_message("查询订单ORD001")
        
        # 验证会话稳定性
        success = (
            pre_update_response["success"] and
            post_update_response["success"] and
            post_update_response.get("session_id") == self.client.session_id
        )
        
        # 结束对话
        self.client.end_conversation()
        
        return {
            "test_name": "test_old_session_stability",
            "passed": success,
            "pre_update_response": pre_update_response,
            "reload_result": reload_result,
            "post_update_response": post_update_response,
            "timestamp": datetime.now().isoformat()
        }


class HealthCheckTestSuite:
    """健康检查测试套件"""
    
    def __init__(self, client: TestClient):
        self.client = client
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """测试健康检查端点"""
        logger.info("开始测试健康检查端点")
        
        health_result = self.client.get_health()
        
        # 验证健康检查响应格式
        success = (
            "healthy" in health_result and
            "timestamp" in health_result and
            "checks" in health_result and
            "version" in health_result and
            isinstance(health_result["checks"], dict)
        )
        
        return {
            "test_name": "test_health_endpoint",
            "passed": success,
            "health_result": health_result,
            "timestamp": datetime.now().isoformat()
        }


def run_comprehensive_test() -> Dict[str, Any]:
    """运行综合测试"""
    logger.info("开始运行综合测试")
    
    test_client = TestClient()
    results = {
        "start_time": datetime.now().isoformat(),
        "tests": {},
        "summary": {}
    }
    
    try:
        # 先检查服务是否运行
        health_suite = HealthCheckTestSuite(test_client)
        health_result = health_suite.test_health_endpoint()
        results["tests"]["health"] = health_result
        
        if not health_result["passed"]:
            results["summary"] = {
                "total_tests": 1,
                "passed_tests": 0,
                "failed_tests": 1,
                "error": "服务健康检查失败，跳过其他测试"
            }
            return results
        
        # 运行发票功能测试
        invoice_suite = InvoiceTestSuite(test_client)
        invoice_results = invoice_suite.run_all_tests()
        results["tests"]["invoice"] = invoice_results
        
        # 运行热更新测试
        hot_update_suite = HotUpdateTestSuite(test_client)
        hot_update_results = hot_update_suite.run_all_tests()
        results["tests"]["hot_update"] = hot_update_results
        
        # 汇总统计
        all_tests = [health_result] + invoice_results + hot_update_results
        total_tests = len(all_tests)
        passed_tests = sum(1 for test in all_tests if test.get("passed", False))
        failed_tests = total_tests - passed_tests
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0
        }
        
        results["end_time"] = datetime.now().isoformat()
        
        logger.info(f"测试完成: {passed_tests}/{total_tests} 通过")
        
    except Exception as e:
        logger.error(f"测试执行异常: {str(e)}")
        results["error"] = str(e)
        results["end_time"] = datetime.now().isoformat()
    
    return results


def save_test_report(results: Dict[str, Any], filename: str = None):
    """保存测试报告"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"测试报告已保存到: {filename}")
    except Exception as e:
        logger.error(f"保存测试报告失败: {str(e)}")


def main():
    """主函数"""
    print("=" * 60)
    print("多轮对话系统自动化测试")
    print("=" * 60)
    
    # 检查服务是否运行
    try:
        response = requests.get(f"{TEST_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ 服务未正常运行，状态码: {response.status_code}")
            print("请先启动服务: python -m week07.multiConv.api")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到服务: {str(e)}")
        print("请先启动服务: python -m week07.multiConv.api")
        return
    
    print("✅ 服务连接正常，开始执行测试...")
    print()
    
    # 运行测试
    results = run_comprehensive_test()
    
    # 显示结果
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"总测试数: {summary.get('total_tests', 0)}")
        print(f"通过测试: {summary.get('passed_tests', 0)}")
        print(f"失败测试: {summary.get('failed_tests', 0)}")
        print(f"成功率: {summary.get('success_rate', 0):.1f}%")
    
    if "error" in results:
        print(f"❌ 测试执行错误: {results['error']}")
    
    # 保存报告
    save_test_report(results)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()