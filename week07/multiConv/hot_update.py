"""
模型与插件热更新功能
"""
import os
import sys
import importlib
import threading
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger

from .config import settings, ToolConfig
from .tools import tool_manager


@dataclass
class HotUpdateConfig:
    """热更新配置"""
    watch_directories: List[str] = field(default_factory=lambda: ["tools", "plugins"])
    file_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    reload_delay: float = 1.0  # 文件变化后延迟重载时间
    max_retries: int = 3
    backup_enabled: bool = True


@dataclass
class UpdateRecord:
    """更新记录"""
    timestamp: datetime
    file_path: str
    update_type: str  # "model", "plugin", "tool"
    success: bool
    error_message: Optional[str] = None
    version: Optional[str] = None


class PluginFileHandler(FileSystemEventHandler):
    """插件文件变化处理器"""
    
    def __init__(self, hot_updater):
        self.hot_updater = hot_updater
        self.pending_updates = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        if self._should_watch_file(file_path):
            # 延迟处理，避免频繁更新
            self.pending_updates[file_path] = time.time()
            threading.Timer(
                self.hot_updater.config.reload_delay,
                self._process_pending_update,
                args=[file_path]
            ).start()
    
    def _should_watch_file(self, file_path: str) -> bool:
        """检查是否应该监视该文件"""
        path = Path(file_path)
        
        # 检查文件扩展名
        for pattern in self.hot_updater.config.file_patterns:
            if path.match(pattern):
                return True
        
        # 排除临时文件和备份文件
        if path.name.startswith('.') or path.name.endswith('~'):
            return False
            
        return False
    
    def _process_pending_update(self, file_path: str):
        """处理待更新的文件"""
        if file_path in self.pending_updates:
            last_time = self.pending_updates[file_path]
            if time.time() - last_time >= self.hot_updater.config.reload_delay:
                self.hot_updater.reload_file(file_path)
                del self.pending_updates[file_path]


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self):
        self.current_version = "1.0.0"
        self.model_configs = {}
        self.backup_configs = {}
    
    def get_current_version(self) -> str:
        """获取当前模型版本"""
        return self.current_version
    
    def update_model_config(self, config: Dict[str, Any]) -> bool:
        """更新模型配置"""
        try:
            # 备份当前配置
            self.backup_configs[self.current_version] = self.model_configs.copy()
            
            # 应用新配置
            self.model_configs.update(config)
            
            # 更新版本号
            import time
            self.current_version = f"1.{int(time.time())}"
            
            logger.info(f"模型配置已更新到版本: {self.current_version}")
            return True
            
        except Exception as e:
            logger.error(f"更新模型配置失败: {str(e)}")
            return False
    
    def rollback_to_version(self, version: str) -> bool:
        """回滚到指定版本"""
        try:
            if version in self.backup_configs:
                self.model_configs = self.backup_configs[version].copy()
                self.current_version = version
                logger.info(f"已回滚到版本: {version}")
                return True
            else:
                logger.error(f"版本 {version} 不存在")
                return False
                
        except Exception as e:
            logger.error(f"版本回滚失败: {str(e)}")
            return False


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.plugin_configs: Dict[str, ToolConfig] = {}
        self.file_hashes: Dict[str, str] = {}
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def load_plugin(self, plugin_path: str) -> bool:
        """加载插件"""
        try:
            # 计算文件哈希
            current_hash = self._calculate_file_hash(plugin_path)
            plugin_name = Path(plugin_path).stem
            
            # 检查是否需要重新加载
            if (plugin_name in self.file_hashes and 
                self.file_hashes[plugin_name] == current_hash):
                logger.debug(f"插件 {plugin_name} 未发生变化，跳过加载")
                return True
            
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找插件类或函数
                plugin_class = getattr(module, 'Plugin', None)
                if plugin_class:
                    plugin_instance = plugin_class()
                    self.plugins[plugin_name] = plugin_instance
                    
                    # 注册工具
                    if hasattr(plugin_instance, 'get_tools'):
                        tools = plugin_instance.get_tools()
                        for tool_name, tool_func in tools.items():
                            if hasattr(plugin_instance, 'get_tool_config'):
                                tool_config = plugin_instance.get_tool_config(tool_name)
                                tool_manager.add_tool(tool_name, tool_func, tool_config)
                                logger.info(f"注册工具: {tool_name}")
                
                # 更新文件哈希
                self.file_hashes[plugin_name] = current_hash
                
                logger.info(f"成功加载插件: {plugin_name}")
                return True
            else:
                logger.error(f"无法创建模块规范: {plugin_path}")
                return False
                
        except Exception as e:
            logger.error(f"加载插件失败 {plugin_path}: {str(e)}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        try:
            if plugin_name in self.plugins:
                plugin = self.plugins[plugin_name]
                
                # 移除注册的工具
                if hasattr(plugin, 'get_tools'):
                    tools = plugin.get_tools()
                    for tool_name in tools.keys():
                        tool_manager.remove_tool(tool_name)
                        logger.info(f"移除工具: {tool_name}")
                
                # 清理插件
                del self.plugins[plugin_name]
                if plugin_name in self.file_hashes:
                    del self.file_hashes[plugin_name]
                
                logger.info(f"成功卸载插件: {plugin_name}")
                return True
            else:
                logger.warning(f"插件 {plugin_name} 不存在")
                return False
                
        except Exception as e:
            logger.error(f"卸载插件失败 {plugin_name}: {str(e)}")
            return False
    
    def reload_plugin(self, plugin_path: str) -> bool:
        """重新加载插件"""
        plugin_name = Path(plugin_path).stem
        
        # 先卸载现有插件
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)
        
        # 重新加载
        return self.load_plugin(plugin_path)
    
    def list_plugins(self) -> List[str]:
        """列出所有已加载的插件"""
        return list(self.plugins.keys())


class HotUpdater:
    """热更新管理器"""
    
    def __init__(self, config: Optional[HotUpdateConfig] = None):
        self.config = config or HotUpdateConfig()
        self.model_manager = ModelVersionManager()
        self.plugin_manager = PluginManager()
        self.update_history: List[UpdateRecord] = []
        self.observer = None
        self.is_running = False
        self._locks = {
            "model": threading.RLock(),
            "plugin": threading.RLock(),
            "history": threading.RLock()
        }
    
    def start_watching(self) -> bool:
        """开始监视文件变化"""
        try:
            if self.is_running:
                logger.warning("热更新监视已在运行")
                return True
            
            self.observer = Observer()
            handler = PluginFileHandler(self)
            
            # 监视指定目录
            for watch_dir in self.config.watch_directories:
                if os.path.exists(watch_dir):
                    self.observer.schedule(handler, watch_dir, recursive=True)
                    logger.info(f"开始监视目录: {watch_dir}")
                else:
                    logger.warning(f"监视目录不存在: {watch_dir}")
            
            self.observer.start()
            self.is_running = True
            
            logger.info("热更新监视已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动热更新监视失败: {str(e)}")
            return False
    
    def stop_watching(self):
        """停止监视文件变化"""
        try:
            if self.observer and self.is_running:
                self.observer.stop()
                self.observer.join()
                self.is_running = False
                logger.info("热更新监视已停止")
        except Exception as e:
            logger.error(f"停止热更新监视失败: {str(e)}")
    
    def reload_file(self, file_path: str) -> bool:
        """重新加载指定文件"""
        try:
            path = Path(file_path)
            
            if "tools" in path.parts or "plugins" in path.parts:
                return self.reload_plugin(file_path)
            elif "models" in path.parts:
                return self.reload_model_config(file_path)
            else:
                logger.debug(f"忽略文件: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"重新加载文件失败 {file_path}: {str(e)}")
            return False
    
    def reload_plugin(self, plugin_path: str) -> bool:
        """重新加载插件"""
        with self._locks["plugin"]:
            try:
                success = self.plugin_manager.reload_plugin(plugin_path)
                
                # 记录更新
                record = UpdateRecord(
                    timestamp=datetime.now(),
                    file_path=plugin_path,
                    update_type="plugin",
                    success=success,
                    error_message=None if success else "重载失败"
                )
                self._add_update_record(record)
                
                if success:
                    logger.info(f"插件热更新成功: {plugin_path}")
                
                return success
                
            except Exception as e:
                error_msg = f"插件热更新失败: {str(e)}"
                logger.error(error_msg)
                
                record = UpdateRecord(
                    timestamp=datetime.now(),
                    file_path=plugin_path,
                    update_type="plugin",
                    success=False,
                    error_message=error_msg
                )
                self._add_update_record(record)
                
                return False
    
    def reload_model_config(self, config_path: str) -> bool:
        """重新加载模型配置"""
        with self._locks["model"]:
            try:
                # 读取配置文件
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                success = self.model_manager.update_model_config(config)
                
                # 记录更新
                record = UpdateRecord(
                    timestamp=datetime.now(),
                    file_path=config_path,
                    update_type="model",
                    success=success,
                    version=self.model_manager.get_current_version()
                )
                self._add_update_record(record)
                
                if success:
                    logger.info(f"模型配置热更新成功: {config_path}")
                
                return success
                
            except Exception as e:
                error_msg = f"模型配置热更新失败: {str(e)}"
                logger.error(error_msg)
                
                record = UpdateRecord(
                    timestamp=datetime.now(),
                    file_path=config_path,
                    update_type="model",
                    success=False,
                    error_message=error_msg
                )
                self._add_update_record(record)
                
                return False
    
    def _add_update_record(self, record: UpdateRecord):
        """添加更新记录"""
        with self._locks["history"]:
            self.update_history.append(record)
            
            # 限制历史记录数量
            if len(self.update_history) > 100:
                self.update_history = self.update_history[-100:]
    
    def get_update_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取更新历史"""
        with self._locks["history"]:
            recent_records = self.update_history[-limit:] if limit > 0 else self.update_history
            
            return [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "file_path": record.file_path,
                    "update_type": record.update_type,
                    "success": record.success,
                    "error_message": record.error_message,
                    "version": record.version
                }
                for record in recent_records
            ]
    
    def get_status(self) -> Dict[str, Any]:
        """获取热更新状态"""
        return {
            "is_running": self.is_running,
            "model_version": self.model_manager.get_current_version(),
            "loaded_plugins": self.plugin_manager.list_plugins(),
            "watched_directories": self.config.watch_directories,
            "total_updates": len(self.update_history),
            "successful_updates": sum(1 for r in self.update_history if r.success),
            "failed_updates": sum(1 for r in self.update_history if not r.success)
        }
    
    def manual_reload_all(self) -> Dict[str, Any]:
        """手动重新加载所有插件"""
        results = {
            "plugins": {},
            "tools_reloaded": False
        }
        
        try:
            # 重新加载工具管理器
            results["tools_reloaded"] = tool_manager.reload_tools()
            
            # 重新加载所有插件
            for watch_dir in self.config.watch_directories:
                if os.path.exists(watch_dir):
                    for file_path in Path(watch_dir).rglob("*.py"):
                        if file_path.name != "__init__.py":
                            plugin_name = file_path.stem
                            success = self.plugin_manager.reload_plugin(str(file_path))
                            results["plugins"][plugin_name] = success
            
            logger.info("手动重新加载完成")
            
        except Exception as e:
            logger.error(f"手动重新加载失败: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def shutdown(self):
        """关闭热更新器"""
        try:
            self.stop_watching()
            logger.info("热更新器已关闭")
        except Exception as e:
            logger.error(f"关闭热更新器失败: {str(e)}")


# 全局热更新器实例
hot_updater = HotUpdater()

# 启动热更新监视（如果启用）
if settings.enable_hot_reload:
    hot_updater.start_watching()