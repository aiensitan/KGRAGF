import os
import ujson as json
from tqdm import tqdm
from llama_index.llms.ollama import Ollama
import torch
from multiprocessing import Pool, cpu_count
import time
import re


# 改进的三元组提取函数，使用原始模型的高质量提示词
def extract_high_quality_triplets(llm, ctx):
    """使用原始模型的高质量提示词提取三元组，增加严格的质量控制"""
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-" * 20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-" * 20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-" * 20}\nText: {ctx}\nTriplets:'

    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split('$$')

    # 预处理：提取原文中的关键词，用于验证
    ctx_lower = ctx.lower()

    for triplet_text in triplet_texts:
        # 清理提示文本 - 增强版本
        triplet_text = triplet_text.replace('There are the triplets extracted from the text:\n\n', '')
        triplet_text = triplet_text.replace('There are the extracted triplets:\n\n', '')
        triplet_text = triplet_text.replace('Here are the triplets extracted from the text:\n\n', '')
        triplet_text = triplet_text.replace('Here are the extracted triplets:\n\n', '')
        triplet_text = triplet_text.replace('Triplets:', '')
        triplet_text = triplet_text.replace('Text:', '')
        triplet_text = triplet_text.strip()

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

        # 新增：过滤无意义的词汇
        meaningless_words = {'it', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 'we', 'you', 'i'}
        if h.lower() in meaningless_words or t.lower() in meaningless_words:
            continue

        # 移除引号，但保留电影名称等特殊表达
        if not (h.startswith('"') and h.endswith('"')):
            h = h.strip('"\'')
        if not (r.startswith('"') and r.endswith('"')):
            r = r.strip('"\'')
        if not (t.startswith('"') and t.endswith('"')):
            t = t.strip('"\'')

        # 确保关系词或尾实体在原文中出现，但允许部分匹配
        if not any(word.lower() in ctx_lower for word in r.split()):
            if not any(word.lower() in ctx_lower for word in t.split()):
                continue

        # 允许头尾相同的情况，但确保关系词有意义
        if h.lower() == t.lower() and len(r.split()) < 2:
            continue

        # 新增：长度限制，避免过长的实体名
        if len(h) > 100 or len(t) > 100 or len(r) > 50:
            continue

        # 新增：检查是否包含重复信息
        if h.lower() in t.lower() and len(h) > 5:
            continue
        if t.lower() in h.lower() and len(t) > 5:
            continue

        triplets.add((h, r, t))

    triplets = [[h, r, t] for (h, r, t) in triplets]
    return triplets


def generate_entity_summary(llm, entity_name, sentences):
    """生成实体摘要"""
    context = "\n".join([f"{i}: {sent}" for i, sent in enumerate(sentences)])
    query = f"""Based on the following sentences about {entity_name}, generate a comprehensive summary (2-3 sentences) that captures the key information:

{context}

Please provide a concise summary of {entity_name}:"""

    try:
        resp = llm.complete(query)
        summary = resp.text.strip()
        # 简单清理
        if summary.startswith("Summary:"):
            summary = summary[8:].strip()
        if summary.startswith(f"{entity_name}"):
            return summary
        return f"{entity_name} {summary}" if len(summary) > 10 else None
    except:
        return None


def extract_entity_relations(llm, context_entities):
    """基于同一个context中的实体提取实体间关系三元组"""
    if len(context_entities) < 2:
        return []

    # 构建实体和其句子的上下文
    entity_names = list(context_entities.keys())
    context_text = ""
    for entity, sentences in context_entities.items():
        context_text += f"\n{entity}:\n"
        for i, sentence in enumerate(sentences):
            context_text += f"  {i + 1}. {sentence}\n"

    # 使用一阶段方法，让llama3直接生成三元组
    query = f"""Based on the following context, identify relationships between different entities and generate triplets in the format <Entity1##relationship##Entity2>.

Context:
{context_text}

Available entities: {', '.join(entity_names)}

Instructions:
1. Look for ANY meaningful relationships between these entities (teammates, competitors, contemporaries, from same organization, worked together, etc.)
2. The relationship can be any real connection - don't limit to specific types
3. Generate triplets in the exact format: <Entity1##relationship##Entity2>$$<Entity1##relationship##Entity2>$$...
4. Only use entities from the available entities list
5. Only include relationships that are clearly supported by the context
6. Make relationships natural and descriptive (e.g., "teammate of", "competed with", "contemporary of", "coached by", etc.)

Examples based on context:
- If two gymnasts were on the same team: <Simone Biles##teammate of##Aly Raisman>
- If someone coached a team: <Romania team##coached by##Mariana Bitang>  
- If they competed in same era: <Tim Daggett##contemporary gymnast##Aly Raisman>
- If they're from same country/organization: <Yelena Grudneva##competed for##Unified Team>

Generate the triplets based on the context above:"""

    try:
        # 直接用llama3生成三元组
        resp = llm.complete(query)
        triplet_text = resp.text.strip()
        # print(f"Generated triplets: {triplet_text}")

        # 解析三元组
        relations = parse_entity_relations_triplets(triplet_text, entity_names)

        return relations
    except Exception as e:
        print(f"Error extracting entity relations: {e}")
        return []


def parse_entity_relations_triplets(triplet_text, valid_entities):
    """解析实体间关系三元组"""
    triplets = []

    # 清理响应文本
    triplet_text = triplet_text.replace('There are the triplets extracted from the text:\n\n', '')
    triplet_text = triplet_text.replace('There are the extracted triplets:\n\n', '')
    triplet_text = triplet_text.replace('Triplets:', '').strip()

    # 处理多个三元组（用$$分隔）
    triplet_candidates = []
    if '$$' in triplet_text:
        triplet_candidates = triplet_text.split('$$')
    else:
        # 如果没有$$分隔符，尝试按行分割
        lines = triplet_text.split('\n')
        for line in lines:
            if '<' in line and '##' in line and '>' in line:
                triplet_candidates.append(line)

    # 创建实体名称的小写版本用于匹配
    valid_entities_lower = [entity.lower() for entity in valid_entities]

    for candidate in triplet_candidates:
        candidate = candidate.strip()

        if len(candidate) <= 6:
            continue

        # 移除开头的<和结尾的>
        if candidate.startswith('<'):
            candidate = candidate[1:]
        if candidate.endswith('>'):
            candidate = candidate[:-1]

        # 分割三元组
        tokens = candidate.split('##')
        if len(tokens) != 3:
            continue

        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()

        # 清理每个部分中可能存在的尖括号
        h = h.replace('<', '').replace('>', '').strip()
        r = r.replace('<', '').replace('>', '').strip()
        t = t.replace('<', '').replace('>', '').strip()

        # 验证头实体和尾实体是否在有效实体列表中
        h_valid = False
        t_valid = False

        for valid_entity in valid_entities:
            # 检查头实体
            if (h.lower() == valid_entity.lower() or
                    valid_entity.lower() in h.lower() or
                    h.lower() in valid_entity.lower()):
                h = valid_entity  # 使用标准实体名
                h_valid = True

            # 检查尾实体
            if (t.lower() == valid_entity.lower() or
                    valid_entity.lower() in t.lower() or
                    t.lower() in valid_entity.lower()):
                t = valid_entity  # 使用标准实体名
                t_valid = True

        # 只有当头实体和尾实体都有效时才添加三元组
        if not (h_valid and t_valid):
            continue

        # 确保头实体和尾实体不同
        if h.lower() == t.lower():
            continue

        # 验证关系词的有效性
        if (len(r) < 2 or
                r.lower() in ['no', 'unknown', 'null', 'the', 'and', 'or']):
            continue

        # 移除引号但保留有意义的内容
        if not (r.startswith('"') and r.endswith('"')):
            r = r.strip('"\'')

        # 长度限制
        if (len(h) > 100 or len(r) > 100 or len(t) > 100):
            continue

        # 最终验证
        if len(h) > 1 and len(r) > 1 and len(t) > 1:
            triplets.append([h, r, t])

    return triplets


def extract_entity_sentence_relations(llm, entity_name, sentences, light_llm=None):
    """提取实体与每个句子的关系 - 生成三元组形式"""
    relations = {}

    # 使用更轻量级的模型进行三元组生成
    if light_llm is None:
        light_llm = Ollama(model='phi', device="cuda", request_timeout=60)  # 使用phi轻量级模型

    for i, sentence in enumerate(sentences):
        # 第一步：让主模型分析句子与实体的关系类型
        analysis_query = f"""Analyze the relationship between the entity and this sentence. What type of information does this sentence provide about the entity?

Entity: {entity_name}
Sentence: {sentence}

Question: What aspect or type of information about {entity_name} is described in this sentence?
Answer with a brief phrase (e.g., "biographical information", "career details", "educational background", "achievements", "personal life", etc.):"""

        try:
            # 第一步：关系分析
            analysis_resp = llm.complete(analysis_query)
            analysis_text = analysis_resp.text.strip()
            # print(f"Analysis for sentence {i}: {analysis_text}")

            # 第二步：使用轻量级模型生成标准三元组
            triplet_query = f"""Generate a triplet showing the relationship between the entity and the sentence content. Follow the exact format:

Entity: {entity_name}
Sentence: {sentence}
Relationship type: {analysis_text}

Create ONE triplet in this exact format:
<{entity_name}##[relationship]##[content_type]>

Where:
- [relationship] should be "describes", "contains", "mentions", or "includes"
- [content_type] should be the type of information (like "career information", "biographical details", "educational background")

Examples:
<Harold Miner##describes##career achievements>
<University of Southern California##contains##educational information>
<NBA Slam Dunk Contest##mentions##competition details>

Generate the triplet:"""

            # 使用轻量级模型生成三元组
            triplet_resp = light_llm.complete(triplet_query)
            triplet_text = triplet_resp.text.strip()
            # print(f"Triplet response for sentence {i}: {triplet_text}")

            # 解析三元组，参考extract_high_quality_triplets的解析逻辑
            triplets = parse_entity_sentence_triplets(triplet_text, entity_name)

            if triplets:
                relations[str(i)] = triplets
                # print(f"✓ Extracted triplets for sentence {i}: {triplets}")
            # else:
            #     print(f"✗ No valid triplets extracted for sentence {i}")

        except Exception as e:
            print(f"Error processing sentence {i}: {e}")
            continue

    return relations


def parse_entity_sentence_triplets(triplet_text, entity_name):
    """解析实体-句子关系三元组，参考extract_high_quality_triplets的解析逻辑"""
    triplets = []

    # 清理响应文本
    triplet_text = triplet_text.replace('There are the triplets extracted from the text:\n\n', '')
    triplet_text = triplet_text.replace('There are the extracted triplets:\n\n', '')
    triplet_text = triplet_text.replace('Triplet:', '').strip()

    # 处理单个三元组或多个三元组（用$$分隔）
    triplet_candidates = []
    if '$$' in triplet_text:
        triplet_candidates = triplet_text.split('$$')
    else:
        triplet_candidates = [triplet_text]

    for candidate in triplet_candidates:
        candidate = candidate.strip()

        if len(candidate) <= 6:
            continue

        # 移除开头的<和结尾的>
        if candidate.startswith('<'):
            candidate = candidate[1:]
        if candidate.endswith('>'):
            candidate = candidate[:-1]

        # 分割三元组
        tokens = candidate.split('##')
        if len(tokens) != 3:
            continue

        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()

        # 清理每个部分中可能存在的尖括号
        h = h.replace('<', '').replace('>', '').strip()
        r = r.replace('<', '').replace('>', '').strip()
        t = t.replace('<', '').replace('>', '').strip()

        # 验证三元组的有效性
        # 1. 头实体应该是目标实体或其变体
        if not (h.lower() == entity_name.lower() or
                entity_name.lower() in h.lower() or
                h.lower() in entity_name.lower()):
            # 如果头实体不匹配，尝试用正确的实体名替换
            h = entity_name

        # 2. 关系词应该有意义
        if (len(r) < 2 or
                r.lower() in ['no', 'unknown', 'null', 'the', 'and', 'or']):
            continue

        # 3. 尾实体应该描述某种方面或内容
        if (len(t) < 2 or
                t.lower() in ['no', 'unknown', 'null', 'the', 'and', 'or']):
            continue

        # 4. 移除引号但保留有意义的内容
        if not (t.startswith('"') and t.endswith('"')):
            t = t.strip('"\'')

        # 5. 确保关系词合理（通常是describes, contains, mentions等）
        valid_relations = ['describes', 'contains', 'mentions', 'includes', 'shows',
                           'presents', 'details', 'covers', 'discusses', 'highlights',
                           'explains', 'provides', 'reveals', 'indicates', 'states']

        if not any(rel in r.lower() for rel in valid_relations):
            # 如果关系词不在预期列表中，但其他部分有效，仍然接受
            if len(r.split()) > 3:  # 如果关系词太长，拒绝
                continue

        # 6. 最终验证
        if (len(h) > 1 and len(r) > 1 and len(t) > 1 and
                len(h) < 100 and len(r) < 50 and len(t) < 100):
            triplets.append([h, r, t])

    return triplets


def process_batch_improved(batch_data):
    # 获取当前进程ID
    process_id = os.getpid()
    # 获取批次ID
    batch_id = batch_data[0]['_id'] if batch_data else "unknown"
    llm = Ollama(model='llama3:8b', device="cuda", request_timeout=120)
    light_llm = Ollama(model='phi', device="cuda", request_timeout=60)
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

    # 首先收集所有实体摘要
    all_entity_summaries = {}

    for sample in batch_data:
        question = sample['question']
        answer = sample['answer']
        ctxs = sample['context']
        for ctx in ctxs:
            ent = ctx[0]
            out_path = os.path.join(out_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(out_path):
                processed_entities += 1
                print(f"[Process {process_id}] Entity {ent} already processed ({processed_entities}/{total_entities})")
                continue

            sentences = ctx[1]

            # 1. 生成实体摘要
            t_summary_start = time.perf_counter()
            entity_summary = generate_entity_summary(llm, ent, sentences)
            t_summary_end = time.perf_counter()
            if entity_summary:
                all_entity_summaries[ent] = entity_summary
            print(f"[Process {process_id}] Entity {ent} summary time: {t_summary_end - t_summary_start:.2f}s")

    # 现在为每个实体生成完整的知识图谱
    for sample in batch_data:
        question = sample['question']
        answer = sample['answer']
        ctxs = sample['context']

        # 每个样本只生成一次实体-实体关系，避免重复调用
        sample_context_entities = {}
        for sample_ctx in ctxs:
            sample_ent = sample_ctx[0]
            sample_sentences = sample_ctx[1]
            sample_context_entities[sample_ent] = sample_sentences
        t_entity_rel_start = time.perf_counter()
        all_entity_relations = extract_entity_relations(llm, sample_context_entities)
        t_entity_rel_end = time.perf_counter()
        print(f"[Process {process_id}] Sample {batch_id} entity-rel time: {t_entity_rel_end - t_entity_rel_start:.2f}s")

        for ctx in ctxs:
            ent = ctx[0]
            out_path = os.path.join(out_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(out_path):
                continue
            t_entity_start = time.perf_counter()

            # 构建层次化的知识图谱结构
            entity_kg = {
                'entity_summary': None,
                'entity_sentence_relations': {},
                'sentence_relations': {},  # 句子内部三元组，类似原模型
                'relations': [],  # 基于实体摘要的实体间关系
                'original_sentences': ctx[1]
            }

            sentences = ctx[1]

            # 1. 使用已生成的实体摘要
            if ent in all_entity_summaries:
                entity_kg['entity_summary'] = all_entity_summaries[ent]

            # 2. 提取实体与句子的关系（描述每个句子说明了实体的什么方面）
            t_entity_sent_start = time.perf_counter()
            entity_sentence_rels = extract_entity_sentence_relations(llm, ent, sentences, light_llm=light_llm)
            t_entity_sent_end = time.perf_counter()
            if entity_sentence_rels:
                entity_kg['entity_sentence_relations'] = entity_sentence_rels
            print(f"[Process {process_id}] Entity {ent} entity-sentence time: {t_entity_sent_end - t_entity_sent_start:.2f}s")

            # 3. 提取句子内部关系（使用原模型的方法）
            sentence_relations = {}
            for i, sentence in enumerate(sentences):
                if i == 0:
                    ctx_text = sentence  # 第一个句子直接使用
                else:
                    ctx_text = f'{ent}: {sentence}'  # 其他句子加上实体前缀

                # 使用原模型的三元组提取方法
                t_sentence_rel_start = time.perf_counter()
                triplets = extract_high_quality_triplets(llm, ctx_text)
                t_sentence_rel_end = time.perf_counter()
                if triplets:
                    sentence_relations[i] = triplets
                print(f"[Process {process_id}] Entity {ent} sentence {i} triplets time: {t_sentence_rel_end - t_sentence_rel_start:.2f}s")

            if sentence_relations:
                entity_kg['sentence_relations'] = sentence_relations

            # 4. 只保留与当前实体相关的关系
            entity_specific_relations = []
            if all_entity_relations:
                for relation in all_entity_relations:
                    if len(relation) == 3:
                        h, r, t = relation
                        # 只保留头实体或尾实体是当前实体的关系
                        if (h.lower() == ent.lower() or
                                ent.lower() in h.lower() or
                                h.lower() in ent.lower() or
                                t.lower() == ent.lower() or
                                ent.lower() in t.lower() or
                                t.lower() in ent.lower()):
                            entity_specific_relations.append(relation)

            if entity_specific_relations:
                entity_kg['relations'] = entity_specific_relations

            # 保存结果
            if any([entity_kg['entity_summary'], entity_kg['relations'],
                    entity_kg['entity_sentence_relations'], entity_kg['sentence_relations']]):
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(entity_kg, f, ensure_ascii=False, indent=2)
                processed_entities += 1

                # 统计信息
                summary_count = 1 if entity_kg['entity_summary'] else 0
                entity_rel_count = len(entity_kg['relations'])
                entity_sent_rel_count = len(entity_kg['entity_sentence_relations'])
                sent_rel_count = sum(len(v) for v in entity_kg['sentence_relations'].values())

                t_entity_end = time.perf_counter()
                print(f"[Process {process_id}] Processed entity {ent} ({processed_entities}/{total_entities}), "
                      f"entity total time: {t_entity_end - t_entity_start:.2f}s")
                # print(f"  - Entity summary: {summary_count}")
                # print(f"  - Entity-entity relations: {entity_rel_count}")
                # print(f"  - Entity-sentence relations: {entity_sent_rel_count}")
                # print(f"  - Sentence internal triplets: {sent_rel_count}")
            else:
                processed_entities += 1
                t_entity_end = time.perf_counter()
                print(
                    f"[Process {process_id}] No useful information extracted for entity {ent} "
                    f"({processed_entities}/{total_entities}), entity total time: {t_entity_end - t_entity_start:.2f}s")

    print(f"[Process {process_id}] Batch {batch_id} completed. Processed {processed_entities} entities")
    return processed_entities


if __name__ == '__main__':
    data_path = '../../data/hotpotqa/hotpot_dev_distractor_v1_dev_1000.json'
    with open(data_path) as f:
        data = json.load(f)

    global out_dir
    out_dir = '../../data/hotpotqa/kgs/extract_subkgs_llama3_three_relations_improved_dev_1000'
    os.makedirs(out_dir, exist_ok=True)

    # 使用与原始模型相同的配置
    num_processes = min(cpu_count(), 2)  # 限制进程数，避免GPU内存不足
    batch_size = 10  # 每个进程处理的样本数

    # 将数据分批
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)

    print(f"\n=== Improved High-Quality Triplet Extraction ===")
    print(f"Total samples: {len(data)}")
    print(f"Number of batches: {len(batches)}")
    print(f"Processes: {num_processes}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {out_dir}")
    print(f"Key improvements:")
    print(f"- Using proven high-quality prompts from original model")
    print(f"- Enhanced quality control with strict filtering")
    print(f"- Simplified data structure (same as original)")
    print(f"- Focused on basic triplets only (no complex relations)")
    print(f"- Same processing flow as original word model")
    print(f"===============================================\n")

    # 使用进程池处理数据
    with Pool(num_processes) as pool:
        processed_counts = list(
            tqdm(pool.imap(process_batch_improved, batches), total=len(batches), desc="Processing improved batches"))

    print(f'\n=== High-Quality Triplet Extraction Summary ===')
    print(f'Total batches processed: {len(batches)}')
    print(f'Total entities processed: {sum(processed_counts)}')
    print(f'Expected quality improvements:')
    print(f'- Higher precision triplets with strict validation')
    print(f'- Better entity-relation relevance')
    print(f'- Reduced noise from invalid relationships')
    print(f'- Same simple structure as original high-performing model')
    print(f'==============================================')
