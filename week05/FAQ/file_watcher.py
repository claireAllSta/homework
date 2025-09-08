import os
import time
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger
from config import settings

class FAQFileHandler(FileSystemEventHandler):
    """FAQ 文件变化处理器"""
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.last_modified = {}
        self.debounce_time = 2.0  # 防抖时间（秒）
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # 只处理 JSON 文件
        if not file_path.endswith('.json'):
            return
        
        # 防抖处理
        current_time = time.time()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < self.debounce_time:
                return
        
        self.last_modified[file_path] = current_time
        
        logger.info(f"检测到文件变化: {file_path}")
        
        # 调用回调函数
        try:
            self.callback(file_path)
        except Exception as e:
            logger.error(f"处理文件变化失败: {e}")
    
    def on_created(self, event):
        """文件创建事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if file_path.endswith('.json'):
            logger.info(f"检测到新文件: {file_path}")
            try:
                self.callback(file_path)
            except Exception as e:
                logger.error(f"处理新文件失败: {e}")

class FileWatcher:
    """文件监控器，用于实现热更新功能"""
    
    def __init__(self, watch_directory: str = None):
        self.watch_directory = watch_directory or settings.WATCH_DIRECTORY
        self.observer: Optional[Observer] = None
        self.callback: Optional[Callable[[str], None]] = None
        
    def start_watching(self, callback: Callable[[str], None]) -> bool:
        """
        开始监控文件变化
        
        Args:
            callback: 文件变化时的回调函数
            
        Returns:
            是否启动成功
        """
        try:
            if not os.path.exists(self.watch_directory):
                logger.warning(f"监控目录不存在，创建目录: {self.watch_directory}")
                os.makedirs(self.watch_directory, exist_ok=True)
            
            self.callback = callback
            
            # 创建事件处理器
            event_handler = FAQFileHandler(callback)
            
            # 创建观察者
            self.observer = Observer()
            self.observer.schedule(
                event_handler,
                self.watch_directory,
                recursive=True
            )
            
            # 启动观察者
            self.observer.start()
            
            logger.info(f"开始监控目录: {self.watch_directory}")
            return True
            
        except Exception as e:
            logger.error(f"启动文件监控失败: {e}")
            return False
    
    def stop_watching(self):
        """停止监控"""
        try:
            if self.observer and self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
                logger.info("文件监控已停止")
        except Exception as e:
            logger.error(f"停止文件监控失败: {e}")
    
    def is_watching(self) -> bool:
        """检查是否正在监控"""
        return self.observer is not None and self.observer.is_alive()
    
    def get_watch_directory(self) -> str:
        """获取监控目录"""
        return self.watch_directory