import os
import sys
import string
import argparse
import pandas as pd
import ujson as json
from tqdm import tqdm
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler


def ngram_overlap(span, sent, n=3):
    while (len(span) < n) or (len(sent) < n):
        n -= 1
    if n <= 0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = ''.join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = ''.join(sent_tokens)
    span_tokens = set([span_tokens[i:i + n] for i in range(len(span_tokens) - n + 1)])
    sent_tokens = set([sent_tokens[i:i + n] for i in range(len(sent_tokens) - n + 1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap) + 0.01) / (len(span_tokens) + 0.01))


def extract_high_quality_triplets(llm, ctx):
    """使用高质量提示词提取三元组，增加严格的质量控制"""
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-" * 20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-" * 20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-" * 20}\nText: {ctx}\nTriplets:'

    resp = llm.complete(query)
    resp = resp.text
    triplets = set()
    triplet_texts = resp.split('$$')

    # 预处理：提取原文中的关键词，用于验证
    ctx_lower = ctx.lower()

    for triplet_text in triplet_texts:
        # 清理提示文本
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

        # 过滤无意义的词汇
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

        # 长度限制，避免过长的实体名
        if len(h) > 100 or len(t) > 100 or len(r) > 50:
            continue

        # 检查是否包含重复信息
        if h.lower() in t.lower() and len(h) > 5:
            continue
        if t.lower() in h.lower() and len(t) > 5:
            continue

        triplets.add((h, r, t))

    triplets = [[h, r, t] for (h, r, t) in triplets]
    return triplets


def generate_entity_summary(llm, entity_name, all_paragraphs):
    """基于实体的所有段落生成实体摘要"""
    # 将所有段落合并
    combined_text = "\n\n".join(all_paragraphs)
    
    query = f"""Based on the following paragraphs about {entity_name}, generate a comprehensive summary (2-3 sentences) that captures the key information across all paragraphs:

Paragraphs about {entity_name}:
{combined_text}

Please provide a concise summary of {entity_name} that synthesizes information from all the paragraphs:"""

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
    """基于同一个sample中的实体提取实体间关系三元组"""
    if len(context_entities) < 2:
        return []

    # 构建实体和其段落的上下文
    entity_names = list(context_entities.keys())
    context_text = ""
    for entity, paragraphs in context_entities.items():
        context_text += f"\n{entity}:\n"
        for i, paragraph in enumerate(paragraphs):
            context_text += f"  {i+1}. {paragraph}\n"
    
    # 使用一阶段方法，让llama3直接生成三元组
    query = f"""Based on the following context, identify relationships between different entities and generate triplets in the format <Entity1##relationship##Entity2>.

Context:
{context_text}

Available entities: {', '.join(entity_names)}

Instructions:
1. Look for ANY meaningful relationships between these entities (related to, connected with, mentioned together, etc.)
2. The relationship can be any real connection - don't limit to specific types
3. Generate triplets in the exact format: <Entity1##relationship##Entity2>$$<Entity1##relationship##Entity2>$$...
4. Only use entities from the available entities list
5. Only include relationships that are clearly supported by the context
6. Make relationships natural and descriptive

Generate the triplets based on the context above:"""

    try:
        # 直接用llama3生成三元组
        resp = llm.complete(query)
        triplet_text = resp.text.strip()
        # print(f"Generated entity relations: {triplet_text}")
        
        # 解析三元组
        relations = parse_entity_relations_triplets(triplet_text, entity_names)
        
        return relations
    except Exception as e:
        # print(f"Error extracting entity relations: {e}")
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
    valid_entities_lower = [ent.lower() for ent in valid_entities]
    
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
        h_valid = any(h.lower() == ent_lower or h.lower() in ent_lower or ent_lower in h.lower() 
                     for ent_lower in valid_entities_lower)
        t_valid = any(t.lower() == ent_lower or t.lower() in ent_lower or ent_lower in t.lower() 
                     for ent_lower in valid_entities_lower)
        
        if not (h_valid and t_valid):
            continue
        
        # --- 新增清洗：去除LLM常见前缀、换行等噪声 --------------
        unwanted_prefixes = [
            'here are the generated triplets:',
            'here is the generated triplet:',
            'here are the extracted triplets:',
            'here are the triplets extracted from the text:',
            'here is the triplet:'
        ]

        def _clean_head_tail(text):
            txt = text.strip()
            # 移除常见前缀
            lower_txt = txt.lower()
            for pref in unwanted_prefixes:
                if lower_txt.startswith(pref):
                    txt = txt[len(pref):].strip()
                    break
            # 若仍包含换行，取最后一行（通常是真实体名）
            if '\n' in txt:
                txt = txt.split('\n')[-1].strip()
            return txt

        h = _clean_head_tail(h)
        t = _clean_head_tail(t)
        # --------------------------------------------------------
        
        # 基本验证
        if (len(h) > 1 and len(r) > 1 and len(t) > 1 and 
            len(h) < 100 and len(r) < 50 and len(t) < 100):
            triplets.append([h, r, t])
    
    return triplets


def extract_entity_paragraph_relations(llm, entity_name, paragraph_text):
    """提取实体与段落的关系 - 生成三元组形式"""
    # 分析段落与实体的关系类型
    analysis_query = f"""Analyze the relationship between the entity and this paragraph. What type of information does this paragraph provide about the entity?

Entity: {entity_name}
Paragraph: {paragraph_text}

Question: What aspect or type of information about {entity_name} is described in this paragraph?
Answer with a brief phrase (e.g., "biographical information", "career details", "educational background", "achievements", "description", etc.):"""
    
    try:
        # 第一步：关系分析
        analysis_resp = llm.complete(analysis_query)
        analysis_text = analysis_resp.text.strip()
        
        # 第二步：生成标准三元组
        triplet_query = f"""Generate a triplet showing the relationship between the entity and the paragraph content. Follow the exact format:

Entity: {entity_name}
Paragraph: {paragraph_text}
Relationship type: {analysis_text}

Create ONE triplet in this exact format:
<{entity_name}##[relationship]##[content_type]>

Where:
- [relationship] should be "describes", "contains", "mentions", or "includes"
- [content_type] should be the type of information (like "career information", "biographical details", "educational background")

Examples:
<Steve Hillage##describes##career achievements>
<Miquette Giraudy##contains##biographical information>
<University of Oxford##mentions##educational details>

Generate the triplet:"""

        # 生成三元组
        triplet_resp = llm.complete(triplet_query)
        triplet_text = triplet_resp.text.strip()
        
        # 解析三元组
        triplets = parse_entity_paragraph_triplets(triplet_text, entity_name)
        
        return triplets
            
    except Exception as e:
        # print(f"Error processing entity-paragraph relation: {e}")
        return []


def parse_entity_paragraph_triplets(triplet_text, entity_name):
    """解析实体-段落关系三元组"""
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
        
        # --- 新增清洗：剔除 LLM 说明性前缀，处理换行 --------------
        unwanted_prefixes = [
            'here are the generated triplets:',
            'here is the generated triplet:',
            'here are the extracted triplets:',
            'here are the triplets extracted from the text:',
            'here is the triplet:'
        ]

        def _clean_text(txt):
            txt = txt.strip()
            lower_txt = txt.lower()
            for pref in unwanted_prefixes:
                if lower_txt.startswith(pref):
                    txt = txt[len(pref):].strip()
                    break
            if '\n' in txt:
                txt = txt.split('\n')[-1].strip()
            return txt

        h = _clean_text(h)
        t = _clean_text(t)
        # --------------------------------------------------------
        
        # 5. 确保关系词合理
        valid_relations = ['describes', 'contains', 'mentions', 'includes', 'shows', 
                          'presents', 'details', 'covers', 'discusses', 'highlights',
                          'explains', 'provides', 'reveals', 'indicates', 'states']
        
        if not any(rel in r.lower() for rel in valid_relations):
            if len(r.split()) > 3:  # 如果关系词太长，拒绝
                continue
        
        # 6. 最终验证
        if (len(h) > 1 and len(r) > 1 and len(t) > 1 and 
            len(h) < 100 and len(r) < 50 and len(t) < 100):
            triplets.append([h, r, t])
    
    return triplets


def extract_triplets_from_musique(data, llm):
    """从MuSiQue数据中提取多种类型的三元组"""
    ents = set()
    ent2kg = dict()  # 存储完整的知识图谱结构
    text2seq = dict()  # 记录每个实体下段落到seq的映射（当原数据无 seq 字段）
    textcount = 0
    unique_textcount = 0
    
    for index, row in tqdm(data.iterrows()):
        question = row['question']
        answer = row['answer']
        ctxs = row['paragraphs']
        
        # 为每个段落构建知识图谱
        for ctx in ctxs:
            ent = ctx['title']
            text = ctx['paragraph_text']
            # --- 按 word 版逻辑为每个实体分配稳定的 seq ---------------
            if ent not in text2seq:
                text2seq[ent] = {}
            if text not in text2seq[ent]:
                text2seq[ent][text] = len(text2seq[ent])
            seq_val = text2seq[ent][text]
            ctx['seq'] = seq_val  # 记录回原 ctx 供后续可能使用

            seq = str(seq_val)  # KG 键统一用字符串
            # ------------------------------------------------------
            
            ents.add(ent)
            textcount += 1
            
            # 如果实体还没有知识图谱，创建一个并生成实体摘要
            if ent not in ent2kg:
                ent2kg[ent] = {}
                unique_textcount += 1
                
                # 收集当前样本中的所有实体和段落（为生成实体摘要）
                sample_entities = {}
                for sample_ctx in ctxs:
                    sample_ent = sample_ctx['title']
                    sample_text = sample_ctx['paragraph_text']
                    if sample_ent not in sample_entities:
                        sample_entities[sample_ent] = []
                    sample_entities[sample_ent].append(sample_text)
                
                # 生成实体摘要（基于该实体的所有段落）
                entity_summary = generate_entity_summary(llm, ent, sample_entities[ent])
                if entity_summary:
                    ent2kg[ent]['entity_summary'] = entity_summary
            
            # 构建这个段落的知识图谱条目（只包含三种关系）
            if seq not in ent2kg[ent]:
                ent2kg[ent][seq] = {
                    'entity_paragraph_relations': [],
                    'paragraph_internal_triplets': [],
                    'entity_relations': []
                }
            
            # 1. 提取实体与段落的关系
            entity_paragraph_rels = extract_entity_paragraph_relations(llm, ent, text)
            if entity_paragraph_rels:
                ent2kg[ent][seq]['entity_paragraph_relations'] = entity_paragraph_rels
            
            # 2. 提取段落内部的三元组（使用原始方法）
            paragraph_triplets = extract_high_quality_triplets(llm, f'{ent}: {text}')
            if paragraph_triplets:
                ent2kg[ent][seq]['paragraph_internal_triplets'] = paragraph_triplets
            
            # 3. 基于当前样本中的所有实体生成实体间关系（每个实体都调用一次）
            sample_context_entities = {}
            for sample_ctx in ctxs:
                sample_ent = sample_ctx['title']
                sample_text = sample_ctx['paragraph_text']
                if sample_ent not in sample_context_entities:
                    sample_context_entities[sample_ent] = []
                sample_context_entities[sample_ent].append(sample_text)
            
            # 为每个实体都调用一次实体间关系提取（与HotpotQA保持一致）
            all_entity_relations = extract_entity_relations(llm, sample_context_entities)
            
            # 只保留与当前实体相关的关系
            if all_entity_relations:
                entity_specific_relations = []
                for relation in all_entity_relations:
                    if len(relation) == 3:
                        h, r, t = relation
                        # 只保留与当前实体相关的关系
                        if (h.lower() == ent.lower() or ent.lower() in h.lower() or h.lower() in ent.lower() or
                            t.lower() == ent.lower() or ent.lower() in t.lower() or t.lower() in ent.lower()):
                            entity_specific_relations.append(relation)
                
                if entity_specific_relations:
                    ent2kg[ent][seq]['entity_relations'] = entity_specific_relations
    
    print(f'#ents: {len(ents)}')
    print(f'#total text: {textcount}')
    print(f'#unique entities: {unique_textcount}')
    return data, ent2kg


def main(args):
    model_name = 'llama3:8b'
    token_counter = TokenCountingHandler()
    Settings.llm = Ollama(model=model_name, request_timeout=120)
    Settings.callback_manager = CallbackManager([token_counter])

    data_dir = '../../data/MuSiQue'
    data_path = os.path.join(data_dir, 'musique_ans_v1.0_dev_10examples.jsonl')
    if not os.path.exists(data_path):
        print(f'Data file not found: {data_path}')
        return
    data = pd.read_json(data_path, lines=True)

    out_dir = '../../data/MuSiQue/kgs/extract_subkgs_10examples_llama3_three_relations'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mapped_data, ent2kg = extract_triplets_from_musique(data, Settings.llm)

    print(f'Completion token count: {token_counter.completion_llm_token_count}')
    print(f'Prompt token count: {token_counter.prompt_llm_token_count}')

    kg_path = os.path.join(out_dir, 'musique_kg_three_relations.json')
    print(f'Saving extracted knowledge graphs to {kg_path}')
    with open(kg_path, 'w') as f:
        json.dump(ent2kg, f, ensure_ascii=False, indent=2)
    
    mapped_data_path = os.path.join(out_dir, 'musique_ans_v1.0_dev_mapped_three_relations.jsonl')
    print(f'Saving mapped data to {mapped_data_path}')
    mapped_data.to_json(mapped_data_path, orient='records', lines=True)

    # 输出统计信息
    total_summaries = sum(
        1 for ent_kg in ent2kg.values()
        if ent_kg.get('entity_summary')
    )

    total_paragraph_relations = 0
    total_internal_triplets = 0
    total_entity_relations = 0

    for ent_kg in ent2kg.values():
        for seq_key, entry in ent_kg.items():
            if seq_key == 'entity_summary':
                continue  # 跳过摘要键
            total_paragraph_relations += len(entry.get('entity_paragraph_relations', []))
            total_internal_triplets += len(entry.get('paragraph_internal_triplets', []))
            total_entity_relations += len(entry.get('entity_relations', []))

    print(f'\n=== Three Relations Extraction Summary ===')
    print(f'Entity summaries: {total_summaries}')
    print(f'Entity-paragraph relations: {total_paragraph_relations}')
    print(f'Paragraph internal triplets: {total_internal_triplets}')
    print(f'Entity-entity relations: {total_entity_relations}')
    print(f'==========================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)