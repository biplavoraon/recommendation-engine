import math


def recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    if not relevant:
        return 0.0

    hits = len(set(recommended) & set(relevant))

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    relevant = set(relevant)

    dcg = 0.0

    for rank, item_id in enumerate(recommended):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 2)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg
