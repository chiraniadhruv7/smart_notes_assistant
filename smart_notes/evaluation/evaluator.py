from smart_notes.evaluation.metrics import recall_at_k, mean_reciprocal_rank


class RAGEvaluator:
    def evaluate(self, relevant_chunks, retrieved_chunks, k=5):
        retrieved_texts = [r.text for r in retrieved_chunks]

        return {
            "recall@k": recall_at_k(relevant_chunks, retrieved_texts, k),
            "mrr": mean_reciprocal_rank(relevant_chunks, retrieved_texts),
        }
