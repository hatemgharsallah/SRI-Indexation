import json
import math
from typing import Dict, List, Sequence, Tuple

from corpus import Corpus, Query
from model_language import LanguageModel
from retrieval_model import RetrievalModel


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """
    DCG_k = rel_1 + sum_{i=2..k} rel_i / log2(i)
    """
    if not relevances or k == 0:
        return 0.0

    dcg = relevances[0]  # rel_1 (no discount)
    for i in range(2, min(k, len(relevances)) + 1):
        rel = relevances[i - 1]
        dcg += rel / math.log2(i)

    return dcg



def ndcg_at_k(ground_truth_scores: Sequence[int], ranked_doc_ids: Sequence[int], k: int) -> float:
    """
    Compute nDCG@k for a single query.

    ground_truth_scores: list where index = doc_id and value = relevance grade.
    ranked_doc_ids: model output list of doc_ids in ranked order (best first).
    """
    ranked_rels = [ground_truth_scores[doc_id] for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(ranked_rels, k)

    ideal_rels = sorted(ground_truth_scores, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_ndcg(
    model: RetrievalModel,
    queries_with_scores: Dict[str, List[int]],
    k: int = 10,
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate a retrieval model with nDCG@k.

    Returns:
        per_query_scores: {query_text: ndcg}
        avg_ndcg: average nDCG@k over all queries
    """
    per_query_scores: Dict[str, float] = {}

    for query_text, scores in queries_with_scores.items():
        ranked_docs = model.rank(Query(query_text), limit=k)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

        ndcg = ndcg_at_k(scores, ranked_doc_ids, k)
        per_query_scores[query_text] = ndcg

    avg_ndcg = sum(per_query_scores.values()) / len(per_query_scores) if per_query_scores else 0.0
    return per_query_scores, avg_ndcg


if __name__ == "__main__":
    with open("evaluation_queries_with_scores.json", "r", encoding="utf-8") as f:
        queries_with_scores = json.load(f)

    model = LanguageModel(Corpus(), 0.2)
    per_query, average = evaluate_ndcg(model, queries_with_scores, k=10)

    print("nDCG@10 per query:")
    for query, score in per_query.items():
        print(f"- {query}: {score:.4f}")

    print(f"Average nDCG@10: {average:.4f}")
