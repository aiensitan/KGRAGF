import os
import copy
import ujson as json
import argparse
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor_three_summary_improved import NaivePostprocessor, KGRetrievePostProcessor, ngram_overlap, GraphFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('KGRAG')
logging.getLogger('KGRAG').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

from builtins import print as _print
from sys import _getframe
def print(*arg, **kw):
    s = f'Line {_getframe(1).f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)

import random
import numpy as np
import torch
import time

# ---------------- Deterministic behaviour ----------------
# Set python hash seed before anything else (must be str)
os.environ.setdefault("PYTHONHASHSEED", "42")

# Python / numpy / torch seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Make torch deterministic (warn_only True to avoid crash if unsupported op)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except AttributeError:
    # Torch <1.8 may not have this API
    pass
# ----------------------------------------------------------

# Cache embeddings/nodes to avoid recomputing embeddings across samples.
NODE_CACHE = {}
EMBEDDING_CACHE = {}

def get_cached_node(text, node_id):
    node = NODE_CACHE.get(node_id)
    if node is None or node.text != text:
        node = TextNode(text=text, id_=node_id)
        emb = EMBEDDING_CACHE.get(text)
        if emb is None:
            emb = Settings.embed_model.get_text_embedding(text)
            EMBEDDING_CACHE[text] = emb
        node.embedding = emb
        NODE_CACHE[node_id] = node
    return node

def read_data(args):
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')
    if args.dataset == 'hotpotqa':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif args.dataset == '2wiki':
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif args.dataset == 'musique':
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return data


def init_model(args):
    # Use greedy decoding to eliminate LLM sampling randomness
    Settings.llm = Ollama(
        model=args.model_name,
        request_timeout=200,
        temperature=0,
        top_p=0,
        top_k=1,
        repeat_penalty=1.0,
    )
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)
def read_kg(args, data):
    if args.dataset == 'hotpotqa':
        ents = set()
        for sample in data:
            for ctx in sample['context']:
                ents.add(ctx[0])
        kg_dir = args.kg_dir
        doc2kg = dict()
        print('Loading KGs')
        for ent in tqdm(ents):
            subkg_path = os.path.join(kg_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path, 'r', encoding='utf-8') as f:
                    subkg = json.load(f)
                    # 保持原有的复杂数据结构，不转换为原模型格式
                    if subkg and len(subkg.keys()) > 0:
                        # 清理空的sentence_relations条目
                        if 'sentence_relations' in subkg:
                            cleaned_sentence_relations = {}
                            for seq, triplets in subkg['sentence_relations'].items():
                                if triplets and len(triplets) > 0:
                                    cleaned_sentence_relations[seq] = triplets
                            subkg['sentence_relations'] = cleaned_sentence_relations

                        # 只有在有有效数据时才添加
                        if (subkg.get('sentence_relations') or
                                subkg.get('entity_sentence_relations') or
                                subkg.get('relations') or
                                subkg.get('entity_summary')):
                            doc2kg[ent] = subkg
    elif args.dataset == '2wiki':
        ents = set()
        for sample in data:
            ctxs = sample.get('context')
            if isinstance(ctxs, str):
                ctxs = json.loads(ctxs)
            for ctx in ctxs:
                ents.add(ctx[0])
        kg_dir = args.kg_dir
        doc2kg = dict()
        print('Loading KGs')
        for ent in tqdm(ents):
            subkg_path = os.path.join(kg_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(subkg_path):
                with open(subkg_path, 'r', encoding='utf-8') as f:
                    subkg = json.load(f)
                    if subkg and len(subkg.keys()) > 0:
                        if 'sentence_relations' in subkg:
                            cleaned_sentence_relations = {}
                            for seq, triplets in subkg['sentence_relations'].items():
                                if triplets and len(triplets) > 0:
                                    cleaned_sentence_relations[seq] = triplets
                            subkg['sentence_relations'] = cleaned_sentence_relations

                        if (subkg.get('sentence_relations') or
                                subkg.get('entity_sentence_relations') or
                                subkg.get('relations') or
                                subkg.get('entity_summary')):
                            doc2kg[ent] = subkg
    elif args.dataset == 'musique':
        kg_dir = args.kg_dir
        kg_path = os.path.join(kg_dir, 'musique_kg_three_relations.json')
        with open(kg_path, 'r', encoding='utf-8') as f:
            doc2kg = json.load(f)
    print(f'Loaded kg for {len(doc2kg.keys())} entities from {args.dataset}')
    return doc2kg

def write_prediction(args, data, prediction):
    result_path = args.result_path
    # 获取目标目录路径
    output_dir = os.path.dirname(result_path)

    # 判断目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        print(f"Created directory: {output_dir}")

    if args.dataset == 'hotpotqa' or args.dataset == '2wiki':
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(prediction, f)
    elif args.dataset == 'musique':
        with open(result_path, 'w', encoding='utf-8') as f:
            for sample in data:
                sample_id = sample['id']
                sample['predicted_answer'] = prediction['answer'][sample_id]
                sample['predicted_support_idxs'] = prediction['sp'][sample_id]
                sample['predicted_answerable'] = sample['answerable']
                f.write(json.dumps(sample) + '\n')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    print(f'Prediction written to {result_path}')

def process_sample(args, sample, kg):
    t_start = time.perf_counter()
    if args.dataset == 'hotpotqa':
        sample_id = sample['_id']
    elif args.dataset == '2wiki':
        sample_id = sample['_id']
    elif args.dataset == 'musique':
        sample_id = sample['id']
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    sample_question = sample['question']
    sample_answer = sample['answer']

    ents = set()
    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    if args.dataset == 'hotpotqa':
        ctxs = sample['context']  # 获取上下文
        ents = [ctx[0] for ctx in ctxs]  # 获取实体
        for ctx in ctxs:
            ent = ctx[0]
            chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                # ------------- add original sentence nodes -------------
                # 检查知识图谱中是否存在该实体的信息
                if ent in kg:
                    subkg[ent] = kg[ent]  # 直接复制整个实体的KG数据
                text = f'{ent}: {ctx[1][i]}'  # 构建文本节点
                node_id = f'{ent}##{str(i)}'
                doc_chunk = get_cached_node(text, node_id)  # 创建文本节点
                doc_chunks.append(doc_chunk)  # 将文本节点添加到文档块列表中
                chunks_index[ent][str(i)] = text  # 将文本节点存储到索引中

                # (MuSiQue only) summary nodes are not added for HotpotQA to avoid seq parsing issues
    elif args.dataset == '2wiki':
        ctxs = sample.get('context')
        if isinstance(ctxs, str):
            ctxs = json.loads(ctxs)
        ents = [ctx[0] for ctx in ctxs]
        for ctx in ctxs:
            ent = ctx[0]
            chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                # ------------- add original sentence nodes -------------
                # 妫€鏌ョ煡璇嗗浘璋变腑鏄惁瀛樺湪璇ュ疄浣撶殑淇℃伅
                if ent in kg:
                    subkg[ent] = kg[ent]  # 鐩存帴澶嶅埗鏁翠釜瀹炰綋鐨凨G鏁版嵁
                text = f'{ent}: {ctx[1][i]}'  # 鏋勫缓鏂囨湰鑺傜偣
                node_id = f'{ent}##{str(i)}'
                doc_chunk = get_cached_node(text, node_id)  # 鍒涘缓鏂囨湰鑺傜偣
                doc_chunks.append(doc_chunk)  # 灏嗘枃鏈妭鐐规坊鍔犲埌鏂囨。鍧楀垪琛ㄤ腑
                chunks_index[ent][str(i)] = text  # 灏嗘枃鏈妭鐐瑰瓨鍌ㄥ埌绱㈠索涓?
                # (MuSiQue only) summary nodes are not added for HotpotQA/2Wiki to avoid seq parsing issues
    elif args.dataset == 'musique':
        ctxs = sample['paragraphs']
        for ctx in ctxs:
            idx = ctx['idx']
            ent = ctx['title']
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            seq = ctx['seq']
            text = f'{ent}: {ctx["paragraph_text"]}'
            # 直接将实体完整 KG 注入，KG 内部已按 seq 键细分
            if ent in kg:
                subkg[ent] = kg[ent]
            node_id = f'{str(idx)}##{ent}##{str(seq)}'
            doc_chunk = get_cached_node(text, node_id)
            doc_chunks.append(doc_chunk)
            chunks_index[ent][f'{str(idx)}##{str(seq)}'] = text

        # -------- after looping through ctxs, add entity summary nodes --------
        for ent in ents:
            if ent in kg and isinstance(kg[ent].get('entity_summary'), str):
                summary_text = f'{ent}: {kg[ent]["entity_summary"]}'
                summary_idx_seq = '-1##summary'
                node_id = f'-1##{ent}##summary'
                if ent not in chunks_index:
                    chunks_index[ent] = {}
                if summary_idx_seq not in chunks_index[ent]:
                    chunks_index[ent][summary_idx_seq] = summary_text
                    doc_chunks.append(get_cached_node(summary_text, node_id))

    t_index_start = time.perf_counter()
    index = VectorStoreIndex(doc_chunks)  # 创建向量存储索引
    t_index_end = time.perf_counter()
    # 初次检索放宽：多取 5 条候选供后续过滤，但最终仍截至 args.top_k
    retrieval_k = args.top_k + args.extra_k
    retriever = VectorIndexRetriever(index=index, similarity_top_k=retrieval_k)  # 创建向量索引检索器
    qa_rag_template_str = 'Context information is below.\n{context_str}\nThink step by step but give a short factoid answer (as few words as possible) based on the context and your own knowledge.\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)

    expansion_pp = KGRetrievePostProcessor(dataset=args.dataset, ents=ents, doc2kg=subkg, chunks_index=chunks_index)
    bge_reranker = None if args.no_reranker else FlagReranker(model_name_or_path=args.reranker, device=3)
    filter_pp = GraphFilterPostProcessor(dataset=args.dataset, use_tpt=args.use_tpt, topk=args.top_k, ents=ents, doc2kg=subkg, chunks_index=chunks_index, reranker=bge_reranker)
    naive_pp = NaivePostprocessor(dataset=args.dataset)

    postprocessors = []
    if not args.no_kg_retrieve:
        postprocessors.append(expansion_pp)
    if not args.no_graph_filter:
        postprocessors.append(filter_pp)
    postprocessors.append(naive_pp)

    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=postprocessors)

    try:
        t_query_start = time.perf_counter()
        response = query_engine.query(sample_question)  # 查询引擎查询
        t_query_end = time.perf_counter()
        prediction = response.response  # 获取预测答案
        if args.dataset == 'hotpotqa' or args.dataset == '2wiki':
            sps = []
            for source_node in response.source_nodes:
                parts = source_node.node.id_.split('##')
                if len(parts) < 2:
                    continue
                ent, seq_str = parts[0], parts[1]
                if seq_str.isdigit():
                    sps.append([ent, int(seq_str)])
        elif args.dataset == 'musique':
            sps = [int(source_node.node.id_.split('##')[0]) for source_node in response.source_nodes]
            sps = [idx for idx in sps if (idx >= 0)]
    except Exception as e:
        print(f'Sample {sample_id}, Error: {e}')
        prediction = ''
        sps = []
        t_query_end = time.perf_counter()

    t_end = time.perf_counter()
    timing = {
        'index_s': t_index_end - t_index_start,
        'query_s': t_query_end - t_query_start,
        'total_s': t_end - t_start,
    }
    logger.info(f"Sample {sample_id} timing: index={timing['index_s']:.2f}s, "
                f"query={timing['query_s']:.2f}s, total={timing['total_s']:.2f}s")
    return sample_id, prediction, sps, timing

def kgrag_distractor_predict(args, data, kg):
    prediction = {'answer': {}, 'sp': {}}
    sps_count = 0
    total_index = 0.0
    total_query = 0.0
    total_total = 0.0
    for sample in tqdm(data):
        sample_id, sample_prediction, sample_sps, timing = process_sample(args, sample, kg)
        prediction['answer'][sample_id] = sample_prediction
        prediction['sp'][sample_id] = sample_sps
        sps_count += len(sample_sps)
        total_index += timing['index_s']
        total_query += timing['query_s']
        total_total += timing['total_s']
    print(f'Average number of supporting facts: {sps_count / len(data)}')
    if len(data) > 0:
        logger.info(
            f"Average timing per sample: index={total_index / len(data):.2f}s, "
            f"query={total_query / len(data):.2f}s, total={total_total / len(data):.2f}s"
        )
    return prediction

def main(args):
    data = read_data(args)
    init_model(args)
    kg = read_kg(args, data)
    prediction = kgrag_distractor_predict(args, data, kg)
    write_prediction(args, data, prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # # hotpotqa distractor
    # parser.add_argument('--dataset', type=str, default='hotpotqa', help='Dataset name')
    # parser.add_argument('--data_path', type=str, default='../data/hotpotqa/hotpot_dev_distractor_v1_dev_1000.json', help='Path to the data file')
    # parser.add_argument('--kg_dir', type=str, default='../data/hotpotqa/kgs/extract_subkgs_llama3_three_relations_improved', help='Directory of the KGs')
    # parser.add_argument('--use_tpt', type=bool, default=False, help='Whether to use triplet representation')
    # parser.add_argument('--result_path', type=str,default='../output/hotpotqa/hotpot_dev_distractor_v1_dev_1000-k-25-no_kg_retrieve.json',help='Path to the result file')
    # parser.add_argument('--extra_k', type=int, default=0, help='Additional docs to retrieve before filtering')

    # # pu-hotpotqa distractor
    # parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/pu-hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/pu-hotpotqa/kgs/extract_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=False,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/pu-hotpot/pu-hotpot_dev_distractor_v1_kgrag.json',help='Path to the result file')

    # musique distractor
    parser.add_argument('--dataset',type=str,default='musique',help='Dataset name')
    parser.add_argument('--data_path',type=str,default='../data/MuSiQue/musique_ans_v1.0_dev_mapped_dev_1000.jsonl',help='Path to the mapped data file with seq field')
    parser.add_argument('--kg_dir',type=str,default='../data/MuSiQue/kgs/extract_subkgs_llama3_three_relations',help='Directory of the KGs')
    parser.add_argument('--use_tpt',type=bool,default=True,help='Whether to use triplet representation')
    parser.add_argument('--result_path',type=str,default='../output/MuSiQue/musique_dev_1000_k-15.jsonl',help='Path to the result file')
    parser.add_argument('--extra_k', type=int, default=0, help='Additional docs to retrieve before filtering')

    # # 2wiki distractor
    # parser.add_argument('--dataset', type=str, default='2wiki', help='Dataset name')
    # parser.add_argument('--data_path', type=str, default='../data/2wikimultihopQA/train_200.jsonl', help='Path to the data file')
    # parser.add_argument('--kg_dir', type=str, default='../data/2wikimultihopQA/kgs/extract_subkgs_train_200_llama3_three_relations_improved', help='Directory of the KGs')
    # parser.add_argument('--use_tpt', type=bool, default=False, help='Whether to use triplet representation')
    # parser.add_argument('--result_path', type=str, default='../output/2wikimultihopQA/train_200_kgrag.json', help='Path to the result file')
    # parser.add_argument('--extra_k', type=int, default=5, help='Additional docs to retrieve before filtering')

    # # trivia
    # parser.add_argument('--dataset',type=str,default='hotpotqa',help='Dataset name')
    # parser.add_argument('--data_path',type=str,default='../data/trivia_qa/trivia.json',help='Path to the data file')
    # parser.add_argument('--kg_dir',type=str,default='../data/trivia_qa/kgs/extracted_subkgs',help='Directory of the KGs')
    # parser.add_argument('--use_tpt',type=bool,default=True,help='Whether to use triplet representation')
    # parser.add_argument('--result_path',type=str,default='../output/trivia_qa/trivia_kgrag.json',help='Path to the result file')

    parser.add_argument('--embed_model_name', type=str, default='nomic-embed-text', help='Ollama embedding model name for indexing')
    parser.add_argument('--model_name', type=str, default='llama3:8b', help='Ollama model name')
    parser.add_argument('--reranker', type=str, default='../model/bge-reranker-large', help='Path of the reranker model')
    parser.add_argument('--top_k', type=int, default=15, help='Top k similar documents')

    # Ablation flags
    parser.add_argument('--no_kg_retrieve', action='store_true', default=False, help='Disable KGRetrievePostProcessor for ablation')
    parser.add_argument('--no_graph_filter', action='store_true', default=False, help='Disable GraphFilterPostProcessor for ablation')
    parser.add_argument('--no_reranker', action='store_true', default=False, help='Disable BGE reranker inside GraphFilter')

    args = parser.parse_args()

    main(args)
