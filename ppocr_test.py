"""
PP OCR与LlamaIndex标准集成使用示例
演示如何将PP OCR集成到LlamaIndex的标准文档处理流程中
"""

import os
from pathlib import Path

# 导入LlamaIndex核心组件
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# 导入PP OCR Reader
from ppOcr import PPOCRReader


def setup_test_environment():
    """设置测试环境"""
    print("=== 设置测试环境 ===")
    
    # 创建测试目录
    test_dirs = ["test_images"]
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ 创建目录: {dir_name}")
    
    print("\n请将测试文件放入以下目录:")
    print("- test_images/: 放入图片文件 (.jpg, .png, .bmp等)")
    print()


def example_basic_ocr_reader():
    """示例1: 基础PP OCR Reader使用"""
    print("=== 示例1: 基础PP OCR Reader使用 ===")
    
    # 初始化PP OCR Reader
    ocr_reader = PPOCRReader(
        lang="ch",              # 中文识别
        confidence_threshold=0.6 # 置信度阈值
    )
    
    # 检查是否有测试图片
    test_image = "test_images"
    if os.path.exists(test_image) and os.listdir(test_image):
        image_files = [f for f in os.listdir(test_image) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            image_path = os.path.join(test_image, image_files[0])
            print(f"处理图片: {image_path}")
            
            try:
                # 使用OCR Reader加载文档
                documents = ocr_reader.load_data(image_path)
                
                print(f"✓ 成功识别 {len(documents)} 个文档")
                
                for i, doc in enumerate(documents):
                    print(f"\n文档 {i+1}:")
                    print(f"  内容预览: {doc.text[:100]}...")
                    print(f"  文本长度: {len(doc.text)} 字符")
                    print(f"  识别行数: {doc.metadata.get('total_lines', 0)}")
                    print(f"  置信度阈值: {doc.metadata.get('confidence_threshold')}")
                
                return documents
                
            except Exception as e:
                print(f"✗ 处理失败: {e}")
                return []
        else:
            print("✗ test_images目录中没有图片文件")
            return []
    else:
        print("✗ test_images目录不存在或为空")
        return []


def main():
    """主函数"""
    print("PP OCR与LlamaIndex标准集成示例")
    print("=" * 60)
    
    # 设置测试环境
    setup_test_environment()
    
    # 运行示例
    example_basic_ocr_reader()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("\n集成要点总结:")
    print("1. PPOCRReader继承BaseReader，符合LlamaIndex标准")
    print("2. 可以与SimpleDirectoryReader无缝集成")
    print("3. 测试PPOCRReader的基本功能")
    
    print("\n使用建议:")
    print("- 将图片文件放入test_images目录测试基础功能")


if __name__ == "__main__":
    main()