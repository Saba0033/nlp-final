def recall_at_k(relevant_id, ranked_ids, k):
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def mrr(relevant_id, ranked_ids):
    for i, doc_id in enumerate(ranked_ids):
        if doc_id == relevant_id:
            return 1.0 / (i + 1)
    return 0.0
