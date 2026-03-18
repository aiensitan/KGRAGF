import os
import ujson as json
from tqdm import tqdm
from llama_index.llms.ollama import Ollama
import torch
from multiprocessing import Pool, cpu_count
import time

def extract_triplets(llm, ctx):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split('$$')
    
    for triplet_text in triplet_texts:
        # 清理提示文本
        triplet_text = triplet_text.replace('There are the triplets extracted from the text:\n\n', '')
        triplet_text = triplet_text.replace('There are the extracted triplets:\n\n', '')
        triplet_text = triplet_text.replace('Text:', '')
        
        if len(triplet_text) <= 6:
            continue
            
        # 移除开头的<和结尾的>
        if triplet_text.startswith('<'):
            triplet_text = triplet_text[1:]
        if triplet_text.endswith('>'):
            triplet_text = triplet_text[:-1]
            
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
            
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        
        # 清理每个部分中可能存在的尖括号
        h = h.replace('<', '').replace('>', '').strip()
        r = r.replace('<', '').replace('>', '').strip()
        t = t.replace('<', '').replace('>', '').strip()
        
        # 基本过滤规则
        if ('no ' in h.lower()) or ('no ' in t.lower()) or \
           ('unknown' in h.lower()) or ('unknown' in t.lower()) or \
           ('null' in h.lower()) or ('null' in t.lower()) or \
           (len(h) < 2) or (len(t) < 2):
            continue
            
        # 移除引号，但保留电影名称等特殊表达
        if not (h.startswith('"') and h.endswith('"')):
            h = h.strip('"\'')
        if not (r.startswith('"') and r.endswith('"')):
            r = r.strip('"\'')
        if not (t.startswith('"') and t.endswith('"')):
            t = t.strip('"\'')
        
        # 确保关系词或尾实体在原文中出现，但允许部分匹配
        if not any(word.lower() in ctx.lower() for word in r.split()):
            if not any(word.lower() in ctx.lower() for word in t.split()):
                continue
            
        # 允许头尾相同的情况，但确保关系词有意义
        if h.lower() == t.lower() and len(r.split()) < 2:
            continue

        triplets.add((h, r, t))
        
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets

def process_batch(batch_data):
    # 获取当前进程ID
    process_id = os.getpid()
    # 获取批次ID
    batch_id = batch_data[0]['_id'] if batch_data else "unknown"
    
    # 在每个进程中重新初始化LLM
    llm = Ollama(model='phi', device="cuda", request_timeout=120)
    batch_entities = set()
    processed_entities = 0
    
    for sample in batch_data:
        question = sample['question']
        answer = sample['answer']
        ctxs = sample['context']
        for ctx in ctxs:
            ent = ctx[0]
            batch_entities.add(ent)
    
    total_entities = len(batch_entities)
    print(f"\n[Process {process_id}] Starting batch {batch_id} with {total_entities} unique entities")
    
    for sample in batch_data:
        question = sample['question']
        answer = sample['answer']
        ctxs = sample['context']
        for ctx in ctxs:
            ent = ctx[0]
            out_path = os.path.join(out_dir, f'{ent.replace("/","_")}.json')
            if os.path.exists(out_path):
                processed_entities += 1
                print(f"[Process {process_id}] Entity {ent} already processed ({processed_entities}/{total_entities})")
                continue
                
            entity_triplets = {}
            for i in range(len(ctx[1])):
                if not i==0:
                    ctx_text = f'{ent}: {ctx[1][i]}'
                else:
                    ctx_text = ctx[1][i]
                ext_triplets = extract_triplets(llm, ctx_text)
                if len(ext_triplets)==0:
                    continue
                entity_triplets[i] = ext_triplets
            
            # 立即保存当前实体的结果，确保正确处理Unicode字符
            if entity_triplets:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(entity_triplets, f, ensure_ascii=False, indent=2)
                processed_entities += 1
                print(f"[Process {process_id}] Processed entity {ent} ({processed_entities}/{total_entities}), extracted {sum(len(v) for v in entity_triplets.values())} triplets")
            else:
                processed_entities += 1
                print(f"[Process {process_id}] No triplets extracted for entity {ent} ({processed_entities}/{total_entities})")
    
    print(f"[Process {process_id}] Batch {batch_id} completed. Processed {processed_entities} entities")
    return processed_entities  # 只返回处理数量，因为结果已经保存

if __name__ == '__main__':
    data_path = '../../data/hotpotqa/hotpot_dev_distractor_v1.json'
    with open(data_path) as f:
        data = json.load(f)

    out_dir = '../../data/hotpotqa/kgs/extract_subkgs_new'
    os.makedirs(out_dir, exist_ok=True)

    # 设置进程数
    num_processes = min(cpu_count(), 4)  # 限制最大进程数为4，避免GPU内存不足
    batch_size = 10  # 每个进程处理的样本数
    
    # 将数据分批
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batches.append(batch)
    
    print(f"\n=== Processing Configuration ===")
    print(f"Total samples: {len(data)}")
    print(f"Number of batches: {len(batches)}")
    print(f"Processes: {num_processes}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {out_dir}")
    print(f"==============================\n")
    
    # 使用进程池处理数据
    with Pool(num_processes) as pool:
        processed_counts = list(tqdm(pool.imap(process_batch, batches), total=len(batches), desc="Processing batches"))
    
    print(f'\n=== Processing Summary ===')
    print(f'Total batches processed: {len(batches)}')
    print(f'Total entities processed: {sum(processed_counts)}')
    print(f'=========================')