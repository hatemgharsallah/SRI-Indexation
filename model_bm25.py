import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel


class BM25Model(RetrievalModel):
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_term_counts: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.num_docs: int = 0

    def build(self, preprocessed_corpus: List[List[str]]) -> None:
        self.num_docs = len(preprocessed_corpus)
        self.doc_term_counts = [Counter(doc) for doc in preprocessed_corpus]
        self.doc_lengths = [len(doc) for doc in preprocessed_corpus]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        )

        doc_freq: Dict[str, int] = {}
        for doc_counter in self.doc_term_counts:
            for term in doc_counter.keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self.doc_freq = doc_freq

    def rank(self, preprocessed_query: List[str]) -> List[Tuple[int, float]]:
        scores = [0.0 for _ in range(self.num_docs)]

        for term in preprocessed_query:
            if term not in self.doc_freq:
                continue
            df = self.doc_freq[term]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5))
            for doc_id, doc_counter in enumerate(self.doc_term_counts):
                tf = doc_counter.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (self.doc_lengths[doc_id] / self.avg_doc_length)
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1) / denom)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return ranked

    def rank_query(self, query: str) -> List[Tuple[int, float]]:
        preprocessed_query = preprocess_document(query)
        return self.rank(preprocessed_query)


PREPROCESSED_CORPUS = preprocess_corpus(CORPUS)
BM25 = BM25Model()
BM25.build(PREPROCESSED_CORPUS)


if __name__ == "__main__":
    results = BM25.rank_query("dopamine memoire cerveau")
    for doc_id, score in results[:5]:
        print(doc_id, score)
