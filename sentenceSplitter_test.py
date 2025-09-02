from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

def main():
    # 示例文本
    sample_text = """
    时政微纪录丨习主席的天津上合时间：上合组织开启新航程 2025-09-02 14:12·央视新闻
    2025年上合组织峰会于8月31日至9月1日在天津市举行。习近平主席同20多位外国领导人和10位国际组织负责人聚首海河之滨，总结上合组织成功经验，擘画上合组织发展蓝图，凝聚“上合组织大家庭”合作共识，推动组织朝着构建更加紧密命运共同体的目标阔步迈进。
    这是中国第五次主办上合组织峰会，也是上合组织成立以来规模最大的一次峰会。本次峰会上，习主席首次提出全球治理倡议。这是习主席继全球发展倡议、全球安全倡议、全球文明倡议之后提出的又一重大全球倡议，这也是新时代中国向世界提供的又一重要公共产品。在“上海精神”指引下，从天津再出发，向着更加美好的未来，上合组织必将开启充满希望的新航程。
    """
    
    # 创建Document对象
    document = Document(text=sample_text)
    

    # 节点1和节点2直接没有重叠
    """
    正常情况下（不使用元数据）：

    文本被按句子边界正确分割
    但由于SentenceSplitter优先保持句子完整性，所以即使设置了chunk_overlap=10，如果句子边界不允许重叠，就不会强制创建重叠

    """

    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=30,
    )
    
    # 对文档进行分割
    nodes = splitter.get_nodes_from_documents([document])
    
    print("=== 文本分割结果 ===")
    print(f"原始文本长度: {len(sample_text)} 字符")
    print(f"分割后节点数量: {len(nodes)}")
    print()
    
    # 打印每个分割后的节点
    for i, node in enumerate(nodes, 1):
        print(f"节点 {i}:")
        print(f"内容: {node.text.strip()}")
        print(f"长度: {len(node.text)} 字符")
        print("-" * 50)

def demo_different_splitters():
    """演示不同的分割器"""
    print("\n=== 不同分割器演示 ===")
    
    text = "这是第一句话。这是第二句话！这是第三句话？这是第四句话。"
    document = Document(text=text)
    
    # 1. 按句子分割
    sentence_splitter = SentenceSplitter(
        chunk_size=10,
        chunk_overlap=0
    )
    
    # 2. 按段落分割（使用不同的分隔符）
    paragraph_splitter = SentenceSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separator="\n\n"
    )
    
    print("按句子分割:")
    sentence_nodes = sentence_splitter.get_nodes_from_documents([document])
    for i, node in enumerate(sentence_nodes, 1):
        print(f"  {i}: {node.text.strip()}\n")
    
    print("\n按段落分割:")
    paragraph_nodes = paragraph_splitter.get_nodes_from_documents([document])
    for i, node in enumerate(paragraph_nodes, 1):
        print(f"  {i}: {node.text.strip()}")

def demo_with_metadata():
    """演示带元数据的文本分割"""
    print("\n=== 带元数据的文本分割 ===")
    """
    使用元数据时：

    元数据长度为25字符，接近chunk_size=30
    这导致SentenceSplitter退化为字符级滑动窗口分割
    在这种模式下，重叠是存在的！看节点1和节点2：
    节点1: "人工智能" (4字符)
    节点2: "人工智能技" (5字符)
    重叠部分: "人工智能" (4字符)
    """
    # Metadata length (25) is close to chunk size (30). Resulting chunks are less than 50 tokens. Consider increasing the chunk size or decreasing the size of your metadata to avoid this.
    
    # 创建带元数据的文档
    document = Document(
        text="人工智能技术正在快速发展。机器学习算法越来越先进。深度学习在各个领域都有应用。",
        metadata={
            "source": "AI技术文档",
            "author": "技术专家",
            "category": "技术"
        }
    )
    
    splitter = SentenceSplitter(chunk_size=30, chunk_overlap=5)
    nodes = splitter.get_nodes_from_documents([document])
    
    for i, node in enumerate(nodes, 1):
        print(f"节点 {i}:")
        print(f"  内容: {node.text.strip()}")
        print(f"  元数据: {node.metadata}")
        print()

if __name__ == "__main__":
    main()
    demo_different_splitters()
    demo_with_metadata()