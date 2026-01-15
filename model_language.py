import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel


class LanguageModel(RetrievalModel):
    def __init__(self, lambd: float = 0.2) -> None:
        self.lambd = lambd
        self.doc_term_counts: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.collection_counts: Counter = Counter()
        self.collection_length: int = 0
        self.num_docs: int = 0

    def build(self, preprocessed_corpus: List[List[str]]) -> None:
        self.num_docs = len(preprocessed_corpus)
        self.doc_term_counts = [Counter(doc) for doc in preprocessed_corpus]
        self.doc_lengths = [len(doc) for doc in preprocessed_corpus]
        self.collection_counts = Counter(
            term for doc in preprocessed_corpus for term in doc
        )
        self.collection_length = sum(self.doc_lengths)

    def rank(self, preprocessed_query: List[str]) -> List[Tuple[int, float]]:
        scores: List[float] = []
        for doc_id, doc_counter in enumerate(self.doc_term_counts):
            doc_length = self.doc_lengths[doc_id]
            log_score = 0.0
            for term in preprocessed_query:
                tf_d = doc_counter.get(term, 0)
                p_mle_d = tf_d / doc_length if doc_length else 0.0
                ctf = self.collection_counts.get(term, 0)
                p_mle_c = ctf / self.collection_length if self.collection_length else 0.0
                p_term = (1 - self.lambd) * p_mle_d + self.lambd * p_mle_c
                if p_term > 0:
                    log_score += math.log(p_term)
                else:
                    log_score += float("-inf")
            scores.append(log_score)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return ranked

    def rank_query(self, query: str) -> List[Tuple[int, float]]:
        preprocessed_query = preprocess_document(query)
        return self.rank(preprocessed_query)


PREPROCESSED_CORPUS = preprocess_corpus(CORPUS)
LANGUAGE_MODEL = LanguageModel()
LANGUAGE_MODEL.build(PREPROCESSED_CORPUS)


if __name__ == "__main__":
    results = LANGUAGE_MODEL.rank_query("dopamine memoire cerveau")
    for doc_id, score in results[:5]:
        print(doc_id, score)
