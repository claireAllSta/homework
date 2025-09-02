"""
PP OCR Reader for LlamaIndex - 标准Reader实现
继承BaseReader，可以直接集成到LlamaIndex的文档处理流程中
仿照https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/json.py 
"""

import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError(
        "请安装PaddleOCR: uv add paddleocr paddlepaddle"
    )

logger = logging.getLogger(__name__)


class PPOCRReader(BaseReader):
    """
    PP OCR Reader for LlamaIndex
    
    标准的LlamaIndex Reader实现，可以直接集成到LlamaIndex的文档处理流程中
    支持SimpleDirectoryReader等标准加载器使用
    """
    
    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = "ch",
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        初始化PP OCR Reader
        
        Args:
            use_angle_cls: 是否使用角度分类器
            lang: 语言类型，支持 'ch', 'en', 'fr', 'german', 'korean', 'japan'
            confidence_threshold: 置信度阈值，低于此值的文本将被过滤
            **kwargs: 其他PaddleOCR参数
        """
        super().__init__()
        # 使用最简单的PaddleOCR初始化，避免参数问题
        self.ocr = PaddleOCR(lang=lang)
        self.confidence_threshold = confidence_threshold
        
    def load_data(
        self, 
        file: Union[str, Path], 
        extra_info: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        从图片文件加载数据
        
        Args:
            file: 图片文件路径
            extra_info: 额外的元数据信息
            
        Returns:
            包含OCR识别文本的Document列表
        """
        file_path = str(file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 检查文件扩展名
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_extensions:
            logger.warning(f"不支持的文件格式: {file_ext}")
            return []
            
        # 执行OCR识别
        try:
            result = self.ocr.ocr(file_path)
            
            if not result or not result[0]:
                logger.warning(f"未能从 {file_path} 中识别到文本")
                return []
                
            # 提取文本内容
            texts = []
            positions = []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    bbox = line[0]  # 边界框坐标
                    text_info = line[1]
                    
                    # 安全地提取文本和置信度
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]  # 获取识别的文本
                        confidence = text_info[1]  # 获取置信度
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 1.0  # 默认置信度
                    else:
                        continue
                    
                    # 过滤低置信度的文本
                    if confidence > self.confidence_threshold:
                        texts.append(text)
                        positions.append({
                            "text": text,
                            "bbox": bbox,
                            "confidence": confidence
                        })
            
            # 合并所有文本
            full_text = "\n".join(texts)
            
            if not full_text.strip():
                logger.warning(f"从 {file_path} 中未提取到有效文本")
                return []
            
            # 创建元数据
            metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "image",
                "ocr_engine": "PaddleOCR_v5",
                "total_lines": len(texts),
                "confidence_threshold": self.confidence_threshold,
                "positions": positions  # 包含位置信息
            }
            
            if extra_info:
                metadata.update(extra_info)
                
            # 创建Document对象
            document = Document(
                text=full_text,
                metadata=metadata
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"OCR识别失败 {file_path}: {str(e)}")
            raise
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名，用于LlamaIndex识别"""
        return "PPOCRReader"