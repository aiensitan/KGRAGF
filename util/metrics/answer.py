"""
Answer metric -- mostly taken directly from squad_tools of allennlp.
"""
import re
import string
import collections
from typing import Tuple, List

from .metric import Metric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1_precision_recall(a_gold, a_pred):
    """计算F1, Precision, Recall - 返回三个值的元组"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        if gold_toks == pred_toks:
            return 1.0, 1.0, 1.0  # f1, precision, recall
        else:
            return 0.0, 0.0, 0.0
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def metric_max_over_ground_truths_detailed(prediction, ground_truths):
    """为多个ground truth计算最大的F1, Precision, Recall"""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        f1, precision, recall = compute_f1_precision_recall(prediction, ground_truth)
        scores_for_ground_truths.append((f1, precision, recall))
    
    # 选择F1最高的结果
    best_score = max(scores_for_ground_truths, key=lambda x: x[0])
    return best_score


class AnswerMetric(Metric):
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._count = 0

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):

        exact_scores = metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        
        # 使用detailed函数获取F1, Precision, Recall
        f1, precision, recall = metric_max_over_ground_truths_detailed(
            predicted_answer, ground_truth_answers
        )

        self._total_em += int(exact_scores)
        self._total_f1 += f1
        self._total_precision += precision
        self._total_recall += recall
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float, float, float]:
        """返回 (EM, F1, Precision, Recall)"""
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        precision_score = self._total_precision / self._count if self._count > 0 else 0
        recall_score = self._total_recall / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score, precision_score, recall_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._count = 0
