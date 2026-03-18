import random
import re
import time
import networkx as nx

from typing import List, Dict, Optional, Set
from FlagEmbedding import FlagReranker
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher

dispatcher = get_dispatcher(__name__)

from builtins import print as _print
import inspect


def print(*arg, **kw):
    s = f'Line {inspect.currentframe().f_back.f_lineno}'
    return _print(f"Func {__name__} - {s}", *arg, **kw)


import string


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


class NaivePostprocessor(BaseNodePostprocessor):
    """Naive Node Postprocessor."""

    dataset: str = Field

    @classmethod
    def class_name(cls) -> str:
        return "NaivePostprocessor"

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        entity_order = {}
        sorted_nodes = []
        for i, node in enumerate(nodes):
            node_id = node.node.id_
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                ent, seq_str = node_id.split('##')
                idx_seq_str = seq_str
                ctx_seq = int(seq_str)
            elif self.dataset == 'musique':
                idx_str, ent, seq_str = node_id.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
                ctx_seq = i
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent, ctx_seq, node))
        sorted_nodes.sort(key=lambda x: (entity_order[x[0]], x[1]))
        sorted_nodes = [node for _, _, node in sorted_nodes]

        prev_ent = ''
        for i in range(0, len(sorted_nodes)):
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                temp_ent = sorted_nodes[i].node.id_.split('##')[0]
            elif self.dataset == 'musique':
                temp_ent = sorted_nodes[i].node.id_.split('##')[1]
            if (prev_ent == temp_ent):
                sorted_nodes[i].node.text = sorted_nodes[i].node.text[len(temp_ent + ': '):]
            if i < len(sorted_nodes) - 1:
                if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                    next_ent = sorted_nodes[i + 1].node.id_.split('##')[0]
                elif self.dataset == 'musique':
                    next_ent = sorted_nodes[i + 1].node.id_.split('##')[1]
                if next_ent != temp_ent:
                    sorted_nodes[i].node.text += '\n'
            prev_ent = temp_ent

        return sorted_nodes


class KGRetrievePostProcessor(BaseNodePostprocessor):
    """KnowledgeGraph-based Node processor for complex KG structure."""

    dataset: str = Field
    ents: Set[str] = Field
    doc2kg: Dict[str, Dict] = Field  # 修改类型以支持复杂结构
    chunks_index: Dict[str, Dict[str, str]] = Field

    @classmethod
    def class_name(cls) -> str:
        return "KGRetrievePostprocessor"

    def _extract_relevant_triplets(self, ent: str, seq: str = None) -> List[List[str]]:
        """从复杂KG结构中提取相关的三元组"""
        triplets = []

        if ent not in self.doc2kg:
            return triplets

        kg_data = self.doc2kg[ent]

        # 如果 KG 采用 ent → seq → {...} 结构，则先下钻到对应 seq
        if seq is not None and seq in kg_data:
            kg_data = kg_data[seq]

        # 1. 提取句子关系 (sentence_relations)
        if 'sentence_relations' in kg_data:
            if seq and seq in kg_data['sentence_relations']:
                # 特定句子的三元组
                sentence_triplets = kg_data['sentence_relations'][seq]
                if isinstance(sentence_triplets, list):
                    for triplet in sentence_triplets:
                        if isinstance(triplet, list) and len(triplet) == 3:
                            triplets.append([str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()])
            else:
                # 所有句子的三元组
                for seq_key, sentence_triplets in kg_data['sentence_relations'].items():
                    if isinstance(sentence_triplets, list):
                        for triplet in sentence_triplets:
                            if isinstance(triplet, list) and len(triplet) == 3:
                                triplets.append(
                                    [str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()])

        # 2. 提取实体-句子关系 (entity_sentence_relations)
        if 'entity_sentence_relations' in kg_data:
            if seq and seq in kg_data['entity_sentence_relations']:
                # 特定句子的实体-句子关系
                ent_sent_relations = kg_data['entity_sentence_relations'][seq]
                if isinstance(ent_sent_relations, list):
                    for relation in ent_sent_relations:
                        if isinstance(relation, list) and len(relation) == 3:
                            triplets.append(
                                [str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()])
            else:
                # 所有句子的实体-句子关系
                for seq_key, ent_sent_relations in kg_data['entity_sentence_relations'].items():
                    if isinstance(ent_sent_relations, list):
                        for relation in ent_sent_relations:
                            if isinstance(relation, list) and len(relation) == 3:
                                triplets.append(
                                    [str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()])

        # 3. 提取本段落级新增关系键
        for key in ['entity_paragraph_relations', 'paragraph_internal_triplets', 'entity_relations']:
            if key in kg_data and isinstance(kg_data[key], list):
                for triplet in kg_data[key]:
                    if isinstance(triplet, list) and len(triplet) == 3:
                        triplets.append([str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()])

        # 4. 提取实体间关系 (relations)
        if 'relations' in kg_data and isinstance(kg_data['relations'], list):
            for relation in kg_data['relations']:
                if isinstance(relation, list) and len(relation) == 3:
                    triplets.append([str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()])

        return triplets

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes with complex KG structure support"""
        top_k = len(nodes)

        retrieved_ids = set()
        retrieved_ents = set()
        related_ents = set()
        highly_related_ents = set()

        textid2score = dict()
        ent_count = dict()
        ent_score = dict()

        for i, node in enumerate(nodes):
            node_id = node.node.id_
            retrieved_ids.add(node_id)
            textid2score[node_id] = node.score

            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                entity, seq_str = node_id.split('##')
            elif self.dataset == 'musique':
                idx_str, entity, seq_str = node_id.split('##')
            else:
                continue

            if (i < (top_k // 2)) and (entity in retrieved_ents):
                highly_related_ents.add(entity)
            retrieved_ents.add(entity)

            if entity not in ent_count:
                ent_count[entity] = 0
                ent_score[entity] = 0.0
            ent_count[entity] += 1
            ent_score[entity] += node.score

        # 识别高度相关实体
        sorted_ents = sorted(ent_count.keys(), key=lambda x: (ent_score[x] / ent_count[x]), reverse=True)
        for i in range(min(2, len(sorted_ents))):
            highly_related_ents.add(sorted_ents[i])

        # 从KG中发现相关实体
        for node in nodes:
            node_id = node.node.id_
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                entity, seq_str = node_id.split('##')
            elif self.dataset == 'musique':
                idx_str, entity, seq_str = node_id.split('##')
            else:
                continue

            # 提取该节点相关的三元组
            triplets = self._extract_relevant_triplets(entity, seq_str)

            for triplet in triplets:
                h, r, t = triplet
                # 应用ngram重叠规范化
                if ngram_overlap(h, entity) >= 0.90 or ngram_overlap(entity, h) >= 0.90:
                    h = entity
                if ngram_overlap(t, entity) >= 0.90 or ngram_overlap(entity, t) >= 0.90:
                    t = entity

                # 发现新实体
                if (h in self.ents) and (h not in retrieved_ents):
                    related_ents.add(h)
                    if h not in ent_count:
                        ent_count[h] = 0
                        ent_score[h] = 0.0
                    ent_count[h] += 1
                    ent_score[h] += node.score
                if (t in self.ents) and (t not in retrieved_ents):
                    related_ents.add(t)
                    if t not in ent_count:
                        ent_count[t] = 0
                        ent_score[t] = 0.0
                    ent_count[t] += 1
                    ent_score[t] += node.score

        # 为新发现的实体生成节点
        additional_ids = set()
        avg_score = float(sum([node.score for node in nodes]) / len(nodes))
        retrieved_ents = retrieved_ents - highly_related_ents

        for ent in (related_ents - retrieved_ents):
            if (ent not in self.chunks_index) or (len(self.chunks_index[ent]) == 0):
                continue
            for idx_seq_str in self.chunks_index[ent]:
                if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                    ctx_id = f'{ent}##{idx_seq_str}'
                elif self.dataset == 'musique':
                    idx_str, seq_str = idx_seq_str.split('##')
                    ctx_id = f'{idx_str}##{ent}##{seq_str}'
                else:
                    continue
                if ctx_id in retrieved_ids:
                    continue
                additional_ids.add(ctx_id)
                textid2score[ctx_id] = 0.0
                if ent in ent_score:
                    textid2score[ctx_id] += (ent_score[ent] + avg_score) / (ent_count[ent] + 1)

        # 创建新节点
        added_nodes = []
        for ctx_id in additional_ids:
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                ent, seq_str = ctx_id.split('##')
                idx_seq_str = seq_str
            elif self.dataset == 'musique':
                idx_str, ent, seq_str = ctx_id.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            else:
                continue
            if ent in self.chunks_index:
                if idx_seq_str in self.chunks_index[ent]:
                    ctx_text = self.chunks_index[ent][idx_seq_str]
                    from llama_index.core.schema import TextNode, NodeWithScore
                    node = TextNode(id_=ctx_id, text=ctx_text)
                    node = NodeWithScore(node=node, score=textid2score[ctx_id])
                    added_nodes.append(node)

        added_nodes = sorted(added_nodes, key=lambda x: x.score, reverse=True)
        return nodes + added_nodes


class GraphFilterPostProcessor(BaseNodePostprocessor):
    """KnowledgeGraph-based Node processor for complex KG structure."""
    dataset: str = Field
    topk: int = Field
    use_tpt: bool = Field
    ents: Set[str] = Field
    doc2kg: Dict[str, Dict] = Field  # 修改类型以支持复杂结构
    chunks_index: Dict[str, Dict[str, str]] = Field
    reranker: Optional[FlagReranker] = Field(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "GraphFilterPostprocessor"

    def _extract_triplets_from_kg(self, ent: str, seq: str = None) -> List[List[str]]:
        """从复杂KG结构中提取三元组"""
        triplets = []

        if ent not in self.doc2kg:
            return triplets

        kg_data = self.doc2kg[ent]

        # 下钻到 seq 子图（MuSiQue 三关系结构）
        if seq is not None and seq in kg_data:
            kg_data = kg_data[seq]

        # 1. 优先使用句子关系 (sentence_relations)
        if 'sentence_relations' in kg_data:
            if seq and seq in kg_data['sentence_relations']:
                sentence_triplets = kg_data['sentence_relations'][seq]
                if isinstance(sentence_triplets, list):
                    for triplet in sentence_triplets:
                        if isinstance(triplet, list) and len(triplet) == 3:
                            h, r, t = str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()
                            triplets.append([h, r, t])
            else:
                # 如果没有指定序列，使用所有句子关系
                for seq_key, sentence_triplets in kg_data['sentence_relations'].items():
                    if isinstance(sentence_triplets, list):
                        for triplet in sentence_triplets:
                            if isinstance(triplet, list) and len(triplet) == 3:
                                h, r, t = str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()
                                triplets.append([h, r, t])

        # 2. 补充使用实体-句子关系 (entity_sentence_relations)
        if 'entity_sentence_relations' in kg_data:
            if seq and seq in kg_data['entity_sentence_relations']:
                ent_sent_relations = kg_data['entity_sentence_relations'][seq]
                if isinstance(ent_sent_relations, list):
                    for relation in ent_sent_relations:
                        if isinstance(relation, list) and len(relation) == 3:
                            h, r, t = str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()
                            triplets.append([h, r, t])

        # 3. 额外加入三关系键
        for key in ['entity_paragraph_relations', 'paragraph_internal_triplets', 'entity_relations']:
            if key in kg_data and isinstance(kg_data[key], list):
                for triplet in kg_data[key]:
                    if isinstance(triplet, list) and len(triplet) == 3:
                        h, r, t = str(triplet[0]).strip(), str(triplet[1]).strip(), str(triplet[2]).strip()
                        triplets.append([h, r, t])

        # 4. 补充使用实体间关系 (relations)
        if 'relations' in kg_data and isinstance(kg_data['relations'], list):
            for relation in kg_data['relations']:
                if isinstance(relation, list) and len(relation) == 3:
                    h, r, t = str(relation[0]).strip(), str(relation[1]).strip(), str(relation[2]).strip()
                    triplets.append([h, r, t])

        return triplets

    def _compute_scores(self, query_text_pairs):
        """Compute relevance scores; fallback to simple n-gram when reranker is None"""
        if self.reranker is None:
            # simple overlap proxy; return list of overlap scores
            return [ngram_overlap(q, d, n=3) for q, d in query_text_pairs]
        return self.reranker.compute_score(query_text_pairs)

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes with complex KG structure support"""
        ents = set()
        rels = set()

        g = nx.MultiGraph()

        for node in nodes:
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                ent, seq_str = node.node.id_.split('##')
            elif self.dataset == 'musique':
                idx_str, ent, seq_str = node.node.id_.split('##')
            else:
                continue
            ents.add(ent)

        for node in nodes:
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                ent, seq_str = node.node.id_.split('##')
                idx_seq_str = seq_str
            elif self.dataset == 'musique':
                idx_str, ent, seq_str = node.node.id_.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            else:
                continue

            # 从复杂KG结构中提取三元组
            triplets = self._extract_triplets_from_kg(ent, seq_str)

            for triplet in triplets:
                h, r, t = triplet
                h = h.strip()
                r = r.strip()
                t = t.strip()
                ents.add(h)
                ents.add(t)
                rels.add(r)
                g.add_edge(h, t, rel=r, source=node.node.id_, weight=node.score)

        mentioned_ents = set()
        mentioned_rels = set()

        # dataset-specific n-gram threshold
        ngram_thr = 0.90 if self.dataset == 'musique' else 0.90

        for ent in ents:
            overlap_score = ngram_overlap(ent, query_bundle.query_str)
            if overlap_score >= ngram_thr:
                mentioned_ents.add(ent)
        for rel in rels:
            overlap_score = ngram_overlap(rel, query_bundle.query_str)
            if overlap_score >= ngram_thr:
                mentioned_rels.add(rel)

        for node in nodes:
            if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                ent, seq_str = node.node.id_.split('##')
                idx_seq_str = seq_str
            elif self.dataset == 'musique':
                idx_str, ent, seq_str = node.node.id_.split('##')
                idx_seq_str = f'{idx_str}##{seq_str}'
            else:
                continue

            # 检查是否存在对应的KG数据
            if (ent not in self.doc2kg):
                continue

            # 从复杂KG结构中提取三元组进行处理
            triplets = self._extract_triplets_from_kg(ent, seq_str)

            for triplet in triplets:
                h, r, t = triplet
                if (h in mentioned_ents) and (r in mentioned_rels) and (not t in mentioned_ents):
                    mentioned_ents.add(t)
                if (t in mentioned_ents) and (r in mentioned_rels) and (not h in mentioned_ents):
                    mentioned_ents.add(h)

        mentioned_ents_list = sorted(mentioned_ents)
        for i in range(len(mentioned_ents_list)):
            for j in range(i + 1, len(mentioned_ents_list)):
                if (g.has_edge(mentioned_ents_list[i], mentioned_ents_list[j])) or (
                        g.has_edge(mentioned_ents_list[j], mentioned_ents_list[i])):
                    continue
                g.add_edge(mentioned_ents_list[i], mentioned_ents_list[j], rel='cooccurrence', source='query',
                           weight=0.0)

        wanted_ents = set(mentioned_ents)
        wanted_rels = set(mentioned_rels)

        early_exit = False
        if (len(wanted_ents) > 3 and len(wanted_rels) > 0 and self.dataset == 'musique'):
            early_exit = True
        early_exit = False

        wccs = list(nx.connected_components(g))
        sorted_wccs = sorted(wccs, key=lambda c: (-len(c), sorted(list(c))[0]))
        cand_ctxs_lists = list()

        for i in range(len(sorted_wccs)):
            if (early_exit) and (len(wanted_ents) == 0):
                break
            cand_ctxs_list = []
            wcc = sorted_wccs[i]
            for ent in sorted(wcc):
                if ent in wanted_ents:
                    wanted_ents.remove(ent)
            if len(wcc) > 1:
                subgraph = g.subgraph(wcc)
                mst = nx.maximum_spanning_tree(subgraph, weight='weight')
                cand_ctx_list = []
                for cand_edge in sorted(mst.edges(data=True), key=lambda e: (-e[2]['weight'], e[0], e[1])):
                    if cand_edge[2]['source'] == 'query':
                        continue
                    else:
                        cand_ctx_list.append(cand_edge[2]['source'])
                cand_ctxs_list.append(cand_ctx_list)
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
                for edge in sorted_edges:
                    if edge[2]['source'] == 'query':
                        continue
                    else:
                        cand_ctxs_list.append([edge[2]['source'], ])
                        break
            cand_ctxs_lists.append(cand_ctxs_list)

        cand_ids_lists = list()
        for cand_ctxs_list in cand_ctxs_lists:
            cand_ids_lists.extend(cand_ctxs_list)

        cand_tpts = []
        cand_strs = []
        for cand_ids_list in cand_ids_lists:
            ctx_str = ''
            tpt_str = ''
            for cand_id in cand_ids_list:
                if self.dataset == 'hotpotqa' or self.dataset == '2wiki':
                    cand_ent, seq_str = cand_id.split('##')
                    idx_seq_str = seq_str
                elif self.dataset == 'musique':
                    idx_str, cand_ent, seq_str = cand_id.split('##')
                    idx_seq_str = f'{idx_str}##{seq_str}'
                else:
                    continue

                # 安全地访问chunks_index
                if cand_ent in self.chunks_index and idx_seq_str in self.chunks_index[cand_ent]:
                    ctx_str += self.chunks_index[cand_ent][idx_seq_str]

                if self.use_tpt:
                    # 从复杂KG结构中提取三元组用于展示
                    triplets = self._extract_triplets_from_kg(cand_ent, seq_str)
                    if triplets:
                        tpt_str += ', '.join([f'{h} has/is {r} {t}' for h, r, t in triplets[:min(len(triplets), 3)]])
                        if len(tpt_str) > 0:
                            ctx_str = f'{ctx_str} Relational facts: {tpt_str}.'
            cand_strs.append(ctx_str)
            cand_tpts.append(tpt_str)

        if len(cand_strs) == 0:
            scores = self._compute_scores([(query_bundle.query_str, node.node.text) for node in nodes])
            sorted_seqs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
            wanted_nodes = [nodes[sorted_seqs[i]] for i in range(min(self.topk, len(sorted_seqs)))]
            return wanted_nodes

        wanted_ctxs = []
        scores = self._compute_scores([(query_bundle.query_str, cand_str) for cand_str in cand_strs])
        # scores = self.reranker.compute_score([(query_bundle.query_str,cand_tpt) for cand_tpt in cand_tpts])
        sorted_seqs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        for seq in sorted_seqs:
            if len(set(wanted_ctxs) | set(cand_ids_lists[seq])) > self.topk:
                break
            wanted_ctxs.extend(cand_ids_lists[seq])
            wanted_ctxs = list(set(wanted_ctxs))

        if len(wanted_ctxs) < self.topk // 2:
            cands = [(query_bundle.query_str, node.node.text,) for node in nodes if node.node.id_ not in wanted_ctxs]
            if len(cands) > 0:
                scores = self._compute_scores(cands)
                sorted_seqs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
                for seq in sorted_seqs[:self.topk]:
                    if nodes[seq].node.id_ not in wanted_ctxs:
                        wanted_ctxs.append(nodes[seq].node.id_)
                        if len(wanted_ctxs) >= self.topk:
                            break

        # -------- ALWAYS include the original top-2 nodes --------
        orig_sorted = sorted(nodes, key=lambda n: n.score, reverse=True)

        # 保底加入原检索前 N 个节点（HotpotQA:2, MuSiQue:5）
        min_orig = 2 if self.dataset == 'musique' else 2

        for n in orig_sorted[:min_orig]:
            if n.node.id_ not in wanted_ctxs:
                wanted_ctxs.append(n.node.id_)

        # 2) 如仍未达到 top_k，再按分数高低补足
        if len(wanted_ctxs) < self.topk:
            for n in orig_sorted:
                if n.node.id_ not in wanted_ctxs:
                    wanted_ctxs.append(n.node.id_)
                if len(wanted_ctxs) >= self.topk:
                    break

        wanted_nodes = []
        for node in nodes:
            if node.node.id_ in wanted_ctxs:
                wanted_nodes.append(node)
        return wanted_nodes
