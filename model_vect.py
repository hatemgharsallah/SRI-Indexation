import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel


class VectorSpaceModel(RetrievalModel):
    def __init__(self) -> None:
        self.doc_term_counts: List[Counter] = []
        self.doc_lengths: List[float] = []
        self.doc_freq: Dict[str, int] = {}
        self.num_docs: int = 0

    def build(self, preprocessed_corpus: List[List[str]]) -> None:
        self.num_docs = len(preprocessed_corpus)
        self.doc_term_counts = [Counter(doc) for doc in preprocessed_corpus]

        doc_freq: Dict[str, int] = {}
        for doc_counter in self.doc_term_counts:
            for term in doc_counter.keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self.doc_freq = doc_freq

        self.doc_lengths = []
        for doc_counter in self.doc_term_counts:
            length_sq = 0.0
            for term, tf in doc_counter.items():
                df = self.doc_freq.get(term, 0)
                idf = math.log(self.num_docs / df) if df else 0.0
                w_td = tf * idf
                length_sq += w_td * w_td
            self.doc_lengths.append(math.sqrt(length_sq) if length_sq else 1.0)

    def rank(self, preprocessed_query: List[str]) -> List[Tuple[int, float]]:
        scores = [0.0 for _ in range(self.num_docs)]
        query_counts = Counter(preprocessed_query)

        for term, tf_q in query_counts.items():
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(self.num_docs / df)
            w_tq = tf_q * idf
            for doc_id, doc_counter in enumerate(self.doc_term_counts):
                tf_d = doc_counter.get(term, 0)
                if tf_d == 0:
                    continue
                w_td = tf_d * idf
                scores[doc_id] += w_td * w_tq

        for doc_id, length in enumerate(self.doc_lengths):
            scores[doc_id] = scores[doc_id] / length if length else 0.0

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return ranked

    def rank_query(self, query: str) -> List[Tuple[int, float]]:
        preprocessed_query = preprocess_document(query)
        return self.rank(preprocessed_query)


PREPROCESSED_CORPUS = preprocess_corpus(CORPUS)
VECTOR_MODEL = VectorSpaceModel()
VECTOR_MODEL.build(PREPROCESSED_CORPUS)


if __name__ == "__main__":
    results = VECTOR_MODEL.rank_query("dopamine memoire cerveau")
    for doc_id, score in results[:5]:
        print(doc_id, score)
