import os
from sentence_transformers import SentenceTransformer
import argparse

def download_model(model_path):
    print(f"开始下载模型到: {model_path}")
    
    # 确保目录存在
    os.makedirs(model_path, exist_ok=True)
    
    # 下载模型
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    
    # 保存到指定路径
    model.save(model_path)
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/bge-large-zh-v1.5',
                      help='自定义模型保存路径')
    args = parser.parse_args()
    
    download_model(args.model_path) 