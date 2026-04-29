from __future__ import annotations

import math


def dcg(relevances, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_score(ranked_ids, correct_ids, k):
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in ranked_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(relevances, k) / idcg if idcg > 0 else 0.0
