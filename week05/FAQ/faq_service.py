from typing import List, Dict, Any, Optional
from loguru import logger
from faq_indexer import FAQIndexer
from faq_retriever import FAQRetriever
from file_watcher import FileWatcher
from config import settings
import asyncio
import threading

class FAQService:
    """FAQ 服务主类，整合索引构建、检索和热更新功能"""
    
    def __init__(self):
        self.indexer = FAQIndexer()
        self.retriever = FAQRetriever()
        self.file_watcher = FileWatcher()
        self._initialized = False
        self._indexing_lock = threading.Lock()
    
    async def initialize(self) -> bool:
        """异步初始化服务"""
        try:
            logger.info("正在初始化 FAQ 服务...")
            
            # 初始化索引器
            if not self.indexer.initialize():
                logger.error("索引器初始化失败")
                return False
            
            # 初始化检索器
            if not self.retriever.initialize():
                logger.error("检索器初始化失败")
                return False
            
            # 加载初始数据
            await self._load_initial_data()
            
            # 启动文件监控
            self._start_file_watching()
            
            self._initialized = True
            logger.info("FAQ 服务初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化 FAQ 服务失败: {e}")
            return False
    
    async def _load_initial_data(self):
        """加载初始 FAQ 数据"""
        try:
            import os
            if os.path.exists(settings.FAQ_DATA_PATH):
                logger.info(f"加载初始 FAQ 数据: {settings.FAQ_DATA_PATH}")
                await self._update_index(settings.FAQ_DATA_PATH)
            else:
                logger.info("未找到初始 FAQ 数据文件，创建示例数据")
                await self._create_sample_data()
        except Exception as e:
            logger.error(f"加载初始数据失败: {e}")
    
    async def _create_sample_data(self):
        """创建示例 FAQ 数据"""
        try:
            import os
            import json
            
            # 确保数据目录存在
            data_dir = os.path.dirname(settings.FAQ_DATA_PATH)
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = [
                {
                    "id": 1,
                    "question": "如何退货？",
                    "answer": "您可以在购买后7天内申请退货。请联系客服或在订单页面点击退货申请。退货商品需保持原包装完整。",
                    "category": "售后服务"
                },
                {
                    "id": 2,
                    "question": "支付方式有哪些？",
                    "answer": "我们支持多种支付方式：支付宝、微信支付、银行卡支付、信用卡支付等。所有支付都经过加密处理，确保安全。",
                    "category": "支付问题"
                },
                {
                    "id": 3,
                    "question": "配送需要多长时间？",
                    "answer": "一般情况下，订单会在1-2个工作日内发货，配送时间根据地区不同为2-7个工作日。偏远地区可能需要更长时间。",
                    "category": "配送问题"
                },
                {
                    "id": 4,
                    "question": "如何修改订单信息？",
                    "answer": "订单提交后30分钟内可以修改。请在订单详情页面点击修改，或联系在线客服协助处理。已发货订单无法修改。",
                    "category": "订单管理"
                },
                {
                    "id": 5,
                    "question": "会员有什么优惠？",
                    "answer": "会员享有多重优惠：专属折扣、积分返还、免费配送、优先客服等。不同等级会员享受不同程度的优惠政策。",
                    "category": "会员服务"
                }
            ]
            
            with open(settings.FAQ_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"示例 FAQ 数据已创建: {settings.FAQ_DATA_PATH}")
            
            # 构建索引
            await self._update_index(settings.FAQ_DATA_PATH)
            
        except Exception as e:
            logger.error(f"创建示例数据失败: {e}")
    
    def _start_file_watching(self):
        """启动文件监控"""
        try:
            def on_file_change(file_path: str):
                """文件变化回调"""
                logger.info(f"检测到文件变化，准备更新索引: {file_path}")
                # 在新线程中执行索引更新
                threading.Thread(
                    target=self._handle_file_change,
                    args=(file_path,),
                    daemon=True
                ).start()
            
            self.file_watcher.start_watching(on_file_change)
            
        except Exception as e:
            logger.error(f"启动文件监控失败: {e}")
    
    def _handle_file_change(self, file_path: str):
        """处理文件变化"""
        try:
            with self._indexing_lock:
                logger.info(f"开始更新索引: {file_path}")
                success = self.indexer.update_index(file_path)
                if success:
                    logger.info("索引更新成功")
                    # 重新初始化检索器以使用新索引
                    self.retriever.initialize()
                else:
                    logger.error("索引更新失败")
        except Exception as e:
            logger.error(f"处理文件变化失败: {e}")
    
    async def _update_index(self, file_path: str):
        """异步更新索引"""
        def update_sync():
            with self._indexing_lock:
                return self.indexer.update_index(file_path)
        
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, update_sync)
        
        if success:
            # 重新初始化检索器
            await loop.run_in_executor(None, self.retriever.initialize)
        
        return success
    
    async def search(self, query: str, top_k: int = None, category: str = None) -> List[Dict[str, Any]]:
        """
        搜索 FAQ
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            category: 类别过滤
            
        Returns:
            搜索结果
        """
        if not self._initialized:
            logger.error("服务未初始化")
            return []
        
        try:
            # 在线程池中执行搜索
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.retriever.search,
                query,
                top_k,
                category
            )
            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    async def get_similar_questions(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取相似问题"""
        if not self._initialized:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.retriever.get_similar_questions,
                question,
                top_k
            )
            return results
        except Exception as e:
            logger.error(f"获取相似问题失败: {e}")
            return []
    
    async def rebuild_index(self, file_path: str = None) -> bool:
        """重建索引"""
        if not file_path:
            file_path = settings.FAQ_DATA_PATH
        
        try:
            return await self._update_index(file_path)
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        try:
            if not self._initialized:
                return {"status": "error", "message": "服务未初始化"}
            
            # 获取检索器健康状态
            loop = asyncio.get_event_loop()
            health_info = await loop.run_in_executor(
                None,
                self.retriever.health_check
            )
            
            # 添加服务级别信息
            health_info.update({
                "service_initialized": self._initialized,
                "file_watching": self.file_watcher.is_watching(),
                "watch_directory": self.file_watcher.get_watch_directory()
            })
            
            return health_info
            
        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        try:
            # 获取当前运行的事件循环
            loop = asyncio.get_event_loop()
            # 使用循环安排任务
            # 获取索引统计
            index_stats = await loop.run_in_executor(
                None,
                self.indexer.get_index_stats
            )
            
            # 获取类别统计
            category_stats = await loop.run_in_executor(
                None,
                self.retriever.get_category_stats
            )
            
            return {
                "index_stats": index_stats,
                "category_stats": category_stats,
                "service_status": "running" if self._initialized else "stopped"
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    async def shutdown(self):
        """关闭服务"""
        try:
            logger.info("正在关闭 FAQ 服务...")
            
            # 停止文件监控
            self.file_watcher.stop_watching()
            
            # 关闭组件
            self.indexer.close()
            self.retriever.close()
            
            self._initialized = False
            logger.info("FAQ 服务已关闭")
            
        except Exception as e:
            logger.error(f"关闭服务失败: {e}")

# 全局服务实例
faq_service = FAQService()