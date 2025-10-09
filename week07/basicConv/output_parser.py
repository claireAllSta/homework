"""
时间推断输出解析器
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import re
from langchain_core.output_parsers import BaseOutputParser
from langchain.schema import OutputParserException


class TimeInferenceOutputParser(BaseOutputParser[Dict[str, Any]]):
    """时间推断输出解析器，将相对时间转换为具体日期"""
    
    def __init__(self, current_date: Optional[datetime] = None):
        """
        初始化解析器
        
        Args:
            current_date: 当前日期，默认为今天
        """
        super().__init__()
        # 直接使用内置属性设置
        object.__setattr__(self, '_current_date', current_date or datetime.now())
    
    @property
    def current_date(self) -> datetime:
        return getattr(self, '_current_date')
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析LLM输出，提取时间推断结果
        
        Args:
            text: LLM的输出文本
            
        Returns:
            包含原始查询、推断日期等信息的字典
        """
        try:
            # 尝试从结构化输出中提取信息
            result = self._parse_structured_output(text)
            if result:
                return result
            
            # 如果不是结构化输出，尝试从自然语言中提取
            return self._parse_natural_language(text)
            
        except Exception as e:
            raise OutputParserException(f"解析输出时出错: {str(e)}")
    
    def _parse_structured_output(self, text: str) -> Optional[Dict[str, Any]]:
        """解析结构化输出"""
        # 查找JSON格式的输出
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        
        if matches:
            try:
                import json
                result = json.loads(matches[0])
                if 'inferred_date' in result:
                    return result
            except:
                pass
        
        return None
    
    def _parse_natural_language(self, text: str) -> Dict[str, Any]:
        """从自然语言输出中解析时间信息"""
        # 查找日期模式
        date_patterns = [
            r'(\d{4}-\d{1,2}-\d{1,2})',  # YYYY-MM-DD
            r'(\d{4}年\d{1,2}月\d{1,2}日)',  # YYYY年MM月DD日
        ]
        
        inferred_date = None
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                date_str = matches[0]
                # 尝试解析日期
                try:
                    if '年' in date_str:
                        # 处理中文日期格式
                        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                    inferred_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                    break
                except:
                    continue
        
        # 如果没有找到具体日期，尝试推断相对时间
        if not inferred_date:
            inferred_date = self._infer_relative_time(text)
        
        return {
            'original_query': self._extract_original_query(text),
            'inferred_date': inferred_date,
            'explanation': text.strip(),
            'current_date': self.current_date.strftime('%Y-%m-%d')
        }
    
    def _extract_original_query(self, text: str) -> str:
        """从输出中提取原始查询"""
        # 查找引号中的内容，通常是原始查询
        quote_patterns = [
            r'"([^"]*)"',
            r'"([^"]*)"',
            r"'([^']*)'",
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        return "未知查询"
    
    def _infer_relative_time(self, text: str) -> str:
        """推断相对时间"""
        text_lower = text.lower()
        
        if '昨天' in text or 'yesterday' in text_lower:
            target_date = self.current_date - timedelta(days=1)
            return target_date.strftime('%Y-%m-%d')
        elif '前天' in text or 'day before yesterday' in text_lower:
            target_date = self.current_date - timedelta(days=2)
            return target_date.strftime('%Y-%m-%d')
        elif '明天' in text or 'tomorrow' in text_lower:
            target_date = self.current_date + timedelta(days=1)
            return target_date.strftime('%Y-%m-%d')
        elif '后天' in text or 'day after tomorrow' in text_lower:
            target_date = self.current_date + timedelta(days=2)
            return target_date.strftime('%Y-%m-%d')
        elif '上周' in text or 'last week' in text_lower:
            target_date = self.current_date - timedelta(weeks=1)
            return target_date.strftime('%Y-%m-%d')
        elif '下周' in text or 'next week' in text_lower:
            target_date = self.current_date + timedelta(weeks=1)
            return target_date.strftime('%Y-%m-%d')
        
        # 默认返回今天
        return self.current_date.strftime('%Y-%m-%d')
    
    @property
    def _type(self) -> str:
        return "time_inference_output_parser"