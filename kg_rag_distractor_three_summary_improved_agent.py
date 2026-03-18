import argparse
import importlib.util
import os
from collections import Counter

from FlagEmbedding import FlagReranker
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode

from util.kg_post_processor_three_summary_improved import (
    NaivePostprocessor,
    KGRetrievePostProcessor,
    ngram_overlap,
    GraphFilterPostProcessor,
)
from util.kg_response_synthesizer import get_response_synthesizer

# Load the base module without modifying it.
_BASE_PATH = os.path.join(os.path.dirname(__file__), "kg_rag_distractor_three_summary_improved.py")
_spec = importlib.util.spec_from_file_location("kgrag_base", _BASE_PATH)
kgrag_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kgrag_base)


def normalize_answer(s):
    if s is None:
        return ""
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).split()


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction)
    gt_tokens = normalize_answer(ground_truth)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class EpsilonGreedyBandit:
    def __init__(self, actions, epsilon=0.1):
        self.actions = list(actions)
        self.epsilon = epsilon
        self.counts = {a: 0 for a in self.actions}
        self.values = {a: 0.0 for a in self.actions}

    def choose(self):
        import random

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.values[a])

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = value + (reward - value) / n


def build_query_engine(args, ents, subkg, chunks_index, retriever, action):
    qa_rag_template_str = (
        "Context information is below.\n{context_str}\n"
        "Think step by step but give a short factoid answer (as few words as possible) "
        "based on the context and your own knowledge.\n"
        "Q: Were Scott Derrickson and Ed Wood of the same nationality?\n"
        "A: Yes.\n"
        "Q: Who was born earlier, Emma Bull or Virginia Woolf?\n"
        "A: Adeline Virginia Woolf.\n"
        "Q: The arena where the Lewiston Maineiacs played their home games can seat how many people?\n"
        "A: 3,677 seated.\n"
        "Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\n"
        "A: Chief of Protocol.\n"
        "---------------------\nQ: {query_str}\nA: "
    )
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template
    )

    expansion_pp = KGRetrievePostProcessor(
        dataset=args.dataset, ents=ents, doc2kg=subkg, chunks_index=chunks_index
    )
    bge_reranker = None if args.no_reranker else FlagReranker(
        model_name_or_path=args.reranker, device=3
    )
    filter_pp = GraphFilterPostProcessor(
        dataset=args.dataset,
        use_tpt=args.use_tpt,
        topk=args.top_k,
        ents=ents,
        doc2kg=subkg,
        chunks_index=chunks_index,
        reranker=bge_reranker,
    )
    naive_pp = NaivePostprocessor(dataset=args.dataset)

    postprocessors = []
    if action == "A2":
        postprocessors.append(expansion_pp)
        postprocessors.append(filter_pp)
    elif action == "A1":
        postprocessors.append(filter_pp)
    postprocessors.append(naive_pp)

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )


def compute_confidence(sample_question, nodes, prediction):
    scores = [n.score for n in nodes if n.score is not None]
    max_score = max(scores) if scores else 0.0
    score_retr = max_score / (max_score + 1.0)

    ent_ids = []
    for n in nodes:
        parts = n.node.id_.split("##")
        if parts:
            ent_ids.append(parts[0])
    if ent_ids:
        most_common = Counter(ent_ids).most_common(1)[0][1]
        score_cons = most_common / len(ent_ids)
    else:
        score_cons = 0.0

    answer_hit = 0.0
    if prediction:
        pred_lower = prediction.strip().lower()
        for n in nodes:
            if pred_lower and pred_lower in n.node.text.lower():
                answer_hit = 1.0
                break

    confidence = 0.5 * score_retr + 0.3 * score_cons + 0.2 * answer_hit
    return confidence, score_retr, score_cons, answer_hit


def process_sample_agent(args, sample, kg, bandit):
    if args.dataset == "hotpotqa":
        sample_id = sample["_id"]
    elif args.dataset == "musique":
        sample_id = sample["id"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    sample_question = sample["question"]
    sample_answer = sample["answer"]

    ents = set()
    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    if args.dataset == "hotpotqa":
        ctxs = sample["context"]
        ents = [ctx[0] for ctx in ctxs]
        for ctx in ctxs:
            ent = ctx[0]
            chunks_index[ent] = {}
            for i in range(len(ctx[1])):
                if ent in kg:
                    subkg[ent] = kg[ent]
                text = f"{ent}: {ctx[1][i]}"
                node_id = f"{ent}##{str(i)}"
                doc_chunk = kgrag_base.get_cached_node(text, node_id)
                doc_chunks.append(doc_chunk)
                chunks_index[ent][str(i)] = text
    elif args.dataset == "musique":
        ctxs = sample["paragraphs"]
        for ctx in ctxs:
            idx = ctx["idx"]
            ent = ctx["title"]
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            seq = ctx["seq"]
            text = f"{ent}: {ctx['paragraph_text']}"
            if ent in kg:
                subkg[ent] = kg[ent]
            node_id = f"{str(idx)}##{ent}##{str(seq)}"
            doc_chunk = kgrag_base.get_cached_node(text, node_id)
            doc_chunks.append(doc_chunk)
            chunks_index[ent][f"{str(idx)}##{str(seq)}"] = text

        for ent in ents:
            if ent in kg and isinstance(kg[ent].get("entity_summary"), str):
                summary_text = f"{ent}: {kg[ent]['entity_summary']}"
                summary_idx_seq = "-1##summary"
                node_id = f"-1##{ent}##summary"
                if ent not in chunks_index:
                    chunks_index[ent] = {}
                if summary_idx_seq not in chunks_index[ent]:
                    chunks_index[ent][summary_idx_seq] = summary_text
                    doc_chunks.append(kgrag_base.get_cached_node(summary_text, node_id))

    index = VectorStoreIndex(doc_chunks)
    retrieval_k = args.top_k + args.extra_k
    retriever = VectorIndexRetriever(index=index, similarity_top_k=retrieval_k)

    # Step 1: fast path (A0)
    qe_a0 = build_query_engine(args, ents, subkg, chunks_index, retriever, "A0")
    response_a0 = qe_a0.query(sample_question)
    prediction_a0 = response_a0.response
    nodes_a0 = response_a0.source_nodes
    conf, score_retr, score_cons, answer_hit = compute_confidence(
        sample_question, nodes_a0, prediction_a0
    )

    if conf >= args.conf_high:
        return sample_id, prediction_a0, nodes_a0, "A0"

    # Step 2: reflection rule + bandit chooses between A1/A2
    if answer_hit == 0.0:
        action = "A2"
    else:
        action = bandit.choose()
    qe = build_query_engine(args, ents, subkg, chunks_index, retriever, action)
    response = qe.query(sample_question)
    prediction = response.response

    # Update bandit reward using self-supervised signals
    _, _, score_cons_b, answer_hit_b = compute_confidence(
        sample_question, response.source_nodes, prediction
    )
    reward = 0.6 * answer_hit_b + 0.4 * score_cons_b
    if action in bandit.actions:
        bandit.update(action, reward)

    return sample_id, prediction, response.source_nodes, action


def kgrag_agent_predict(args, data, kg):
    prediction = {"answer": {}, "sp": {}, "action": {}}
    bandit = EpsilonGreedyBandit(actions=["A1", "A2"], epsilon=args.epsilon)

    for sample in data:
        sample_id, sample_prediction, source_nodes, action = process_sample_agent(
            args, sample, kg, bandit
        )
        prediction["answer"][sample_id] = sample_prediction
        prediction["action"][sample_id] = action

        if args.dataset == "hotpotqa":
            sps = []
            for source_node in source_nodes:
                parts = source_node.node.id_.split("##")
                if len(parts) < 2:
                    continue
                ent, seq_str = parts[0], parts[1]
                if seq_str.isdigit():
                    sps.append([ent, int(seq_str)])
        elif args.dataset == "musique":
            sps = [int(source_node.node.id_.split("##")[0]) for source_node in source_nodes]
            sps = [idx for idx in sps if (idx >= 0)]
        else:
            sps = []
        prediction["sp"][sample_id] = sps

    return prediction


def main(args):
    data = kgrag_base.read_data(args)
    kgrag_base.init_model(args)
    kg = kgrag_base.read_kg(args, data)
    prediction = kgrag_agent_predict(args, data, kg)
    kgrag_base.write_prediction(args, data, prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/hotpotqa/hotpot_dev_distractor_v1_50examples.json",
        help="Path to the data file",
    )
    parser.add_argument(
        "--kg_dir",
        type=str,
        default="../data/hotpotqa/kgs/extract_subkgs_50examples_llama3_three_relations_improved",
        help="Directory of the KGs",
    )
    parser.add_argument(
        "--use_tpt", type=bool, default=False, help="Whether to use triplet representation"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../output/hotpotqa/hotpot_dev_distractor_v1_50examples_llama3_three_summary_improved_agent.json",
        help="Path to the result file",
    )
    parser.add_argument("--extra_k", type=int, default=5, help="Additional docs to retrieve before filtering")

    parser.add_argument("--embed_model_name", type=str, default="nomic-embed-text")
    parser.add_argument("--model_name", type=str, default="llama3:8b")
    parser.add_argument("--reranker", type=str, default="../model/bge-reranker-large")
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--no_reranker", action="store_true", default=False)
    parser.add_argument("--conf_high", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
