import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel


class ExtendedBooleanModel(RetrievalModel):
    def __init__(self, p: float = 2.0) -> None:
        self.p = p
        self.doc_term_counts: List[Counter] = []
        self.max_tf: List[int] = []
        self.num_docs: int = 0

    def build(self, preprocessed_corpus: List[List[str]]) -> None:
        self.num_docs = len(preprocessed_corpus)
        self.doc_term_counts = [Counter(doc) for doc in preprocessed_corpus]
        self.max_tf = [max(counter.values(), default=1) for counter in self.doc_term_counts]

    def _weight(self, term: str, doc_id: int) -> float:
        tf = self.doc_term_counts[doc_id].get(term, 0)
        if tf == 0:
            return 0.0
        return tf / self.max_tf[doc_id]

    def rank(self, preprocessed_query: List[str]) -> List[Tuple[int, float]]:
        if not preprocessed_query:
            return [(doc_id, 0.0) for doc_id in range(self.num_docs)]

        query_terms = list(dict.fromkeys(preprocessed_query))
        m = len(query_terms)

        scores: List[float] = []
        for doc_id in range(self.num_docs):
            sum_weights = 0.0
            for term in query_terms:
                w_ij = self._weight(term, doc_id)
                sum_weights += math.pow(w_ij, self.p)
            rsv = math.pow(sum_weights / m, 1.0 / self.p)
            scores.append(rsv)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return ranked

    def rank_query(self, query: str) -> List[Tuple[int, float]]:
        preprocessed_query = preprocess_document(query)
        return self.rank(preprocessed_query)


PREPROCESSED_CORPUS = preprocess_corpus(CORPUS)
BOOLEAN_MODEL = ExtendedBooleanModel()
BOOLEAN_MODEL.build(PREPROCESSED_CORPUS)


if __name__ == "__main__":
    results = BOOLEAN_MODEL.rank_query("dopamine memoire cerveau")
    for doc_id, score in results[:5]:
        print(doc_id, score)
