#!/usr/bin/env python3
"""
从HotpotQA数据创建文档块索引文件
这个脚本会生成 hotpot_chunks_index.pkl 文件，用于Hybrid RAG系统
"""

import os
import json
import pickle
from typing import Dict, Any
from tqdm import tqdm

def create_chunks_index(data_path: str, output_path: str):
    """
    从HotpotQA数据创建文档块索引
    
    Args:
        data_path: HotpotQA数据文件路径
        output_path: 输出的chunks索引文件路径
    """
    print(f"正在从 {data_path} 加载数据...")
    
    # 加载HotpotQA数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 个样本")
    
    # 构建chunks索引
    chunks_index = {}
    
    print("正在构建文档块索引...")
    for sample in tqdm(data, desc="处理样本"):
        # 获取上下文信息
        contexts = sample.get('context', [])
        
        for ctx in contexts:
            if len(ctx) < 2:
                continue
                
            entity = ctx[0]  # 实体名称
            sentences = ctx[1]  # 句子列表
            
            # 为每个实体创建chunks索引
            if entity not in chunks_index:
                chunks_index[entity] = {}
            
            # 为每个句子创建索引
            for i, sentence in enumerate(sentences):
                chunk_text = f"{entity}: {sentence}"
                chunks_index[entity][str(i)] = chunk_text
    
    print(f"构建了 {len(chunks_index)} 个实体的文档块索引")
    
    # 统计总的文档块数量
    total_chunks = sum(len(chunks) for chunks in chunks_index.values())
    print(f"总共有 {total_chunks} 个文档块")
    
    # 保存到pickle文件
    print(f"正在保存到 {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(chunks_index, f)
    
    print(f"文档块索引已保存到 {output_path}")
    
    # 显示一些示例
    print("\n=== 示例文档块 ===")
    for i, (entity, chunks) in enumerate(list(chunks_index.items())[:3]):
        print(f"\n实体 {i+1}: {entity}")
        for j, (chunk_id, chunk_text) in enumerate(list(chunks.items())[:2]):
            print(f"  块 {chunk_id}: {chunk_text[:100]}...")
    
    return chunks_index

def create_empty_kg(output_path: str):
    """
    创建一个空的KG文件（Hybrid RAG实际不使用KG，但保留接口兼容性）
    
    Args:
        output_path: 输出的KG文件路径
    """
    print(f"正在创建空的KG文件: {output_path}")
    
    # 创建空的KG字典
    empty_kg = {}
    
    # 保存到pickle文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(empty_kg, f)
    
    print(f"空KG文件已保存到 {output_path}")

def main():
    """主函数"""
    # 配置路径
    base_dir = "../.."
    data_dir = os.path.join(base_dir, "data", "hotpotqa")
    
    # 输入文件路径
    input_files = {
        "50examples": os.path.join(data_dir, "hotpot_dev_distractor_v1_50examples.json"),
        "100examples": os.path.join(data_dir, "hotpot_dev_distractor_v1_100examples.json"),
        "full": os.path.join(data_dir, "hotpot_dev_distractor_v1.json")
    }
    
    # 输出文件路径
    output_dir = os.path.join(base_dir, "data")
    
    print("=== 创建HotpotQA文档块索引文件 ===\n")
    
    # 检查哪些输入文件存在
    available_files = {}
    for name, path in input_files.items():
        if os.path.exists(path):
            available_files[name] = path
            print(f"✓ 找到文件: {name} -> {path}")
        else:
            print(f"✗ 文件不存在: {name} -> {path}")
    
    if not available_files:
        print("\n错误: 没有找到任何HotpotQA数据文件!")
        print("请确保以下文件中至少有一个存在:")
        for name, path in input_files.items():
            print(f"  - {path}")
        return
    
    # 选择要处理的文件
    if len(available_files) == 1:
        selected_name, selected_path = list(available_files.items())[0]
        print(f"\n自动选择文件: {selected_name}")
    else:
        print(f"\n找到多个文件，请选择:")
        for i, (name, path) in enumerate(available_files.items(), 1):
            print(f"  {i}. {name}")
        
        while True:
            try:
                choice = int(input("请输入选择 (1-{}): ".format(len(available_files))))
                if 1 <= choice <= len(available_files):
                    selected_name, selected_path = list(available_files.items())[choice-1]
                    break
                else:
                    print("无效选择，请重新输入")
            except ValueError:
                print("请输入数字")
    
    print(f"\n处理文件: {selected_path}")
    
    # 创建输出文件路径
    chunks_output = os.path.join(output_dir, "hotpot_chunks_index.pkl")
    kg_output = os.path.join(output_dir, "hotpot_kg.pkl")
    
    # 创建文档块索引
    chunks_index = create_chunks_index(selected_path, chunks_output)
    
    # 创建空的KG文件
    create_empty_kg(kg_output)
    
    print(f"\n=== 完成 ===")
    print(f"已创建以下文件:")
    print(f"  - 文档块索引: {chunks_output}")
    print(f"  - 空KG文件: {kg_output}")
    print(f"\n现在您可以运行Hybrid RAG了:")
    print(f"  python code/hybrid_rag_baseline.py")

if __name__ == "__main__":
    main() 