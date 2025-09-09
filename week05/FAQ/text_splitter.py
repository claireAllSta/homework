import re
import jieba
from typing import List, Dict, Any
from loguru import logger

class SemanticTextSplitter:
    """语义文本切分器，支持中英文混合文本的智能切分"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        将文本按语义进行切分
        
        Args:
            text: 输入文本
            metadata: 元数据信息
            
        Returns:
            切分后的文本块列表，每个块包含文本和元数据
        """
        if not text.strip():
            return []
            
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 按段落分割
        paragraphs = self._split_by_paragraphs(text)
        
        # 进一步切分长段落
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
            else:
                sub_chunks = self._split_long_paragraph(paragraph)
                chunks.extend(sub_chunks)
        
        # 添加重叠处理
        overlapped_chunks = self._add_overlap(chunks)
        
        # 构建结果
        result = []
        for i, chunk in enumerate(overlapped_chunks):
            chunk_metadata = {
                "chunk_id": i,
                "chunk_size": len(chunk),
                "original_length": len(text)
            }
            if metadata:
                chunk_metadata.update(metadata)
                
            result.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        logger.info(f"文本切分完成: 原文长度={len(text)}, 切分块数={len(result)}")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 统一换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        # 去除多余空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按双换行分割段落
        paragraphs = text.split('\n\n')
        
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """切分长段落"""
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        
        # 尝试按句子分割
        sentences = self._split_by_sentences(paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果单个句子就超过限制，强制切分
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 强制按字符切分长句子
                sub_chunks = self._force_split(sentence)
                chunks.extend(sub_chunks)
            else:
                # 检查添加句子后是否超过限制
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 中文句号、问号、感叹号
        chinese_endings = r'[。！？；]'
        # 英文句号、问号、感叹号
        english_endings = r'[.!?;]'
        
        # 组合正则表达式
        sentence_pattern = f'({chinese_endings}|{english_endings})'
        
        sentences = re.split(sentence_pattern, text)
        
        # 重新组合句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        return result
    
    def _force_split(self, text: str) -> List[str]:
        """强制按字符数切分文本"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠处理"""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 获取前一个块的末尾部分作为重叠
            if len(prev_chunk) > self.chunk_overlap:
                overlap_text = prev_chunk[-self.chunk_overlap:]
                # 尝试在词边界处切分重叠部分
                overlap_text = self._find_word_boundary(overlap_text, reverse=True)
                current_chunk = overlap_text + " " + current_chunk
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _find_word_boundary(self, text: str, reverse: bool = False) -> str:
        """在词边界处切分文本"""
        if not text:
            return text
        
        # 对于中文，使用jieba分词
        words = list(jieba.cut(text))
        
        if reverse:
            # 从后往前找合适的切分点
            total_len = 0
            result_words = []
            for word in reversed(words):
                if total_len + len(word) <= len(text) * 0.8:  # 保留80%的重叠内容
                    result_words.insert(0, word)
                    total_len += len(word)
                else:
                    break
            return ''.join(result_words)
        else:
            # 从前往后找合适的切分点
            total_len = 0
            result_words = []
            for word in words:
                if total_len + len(word) <= len(text) * 0.8:
                    result_words.append(word)
                    total_len += len(word)
                else:
                    break
            return ''.join(result_words)
