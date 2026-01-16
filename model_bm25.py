import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS, Corpus, Query
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel


class BM25Model(RetrievalModel):
    def __init__(self, corpus:Corpus, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(corpus)
        self.k1 = k1
        self.b = b
    

    def rsv(self, query: Query, doc_id: int) -> float:
        for term in query.get_preprocessed_query():
            term_freq_in_doc = self.corpus.termOccurenceInDocument(term, doc_id)
            doc_length = self.corpus.get_doc_length(doc_id)
            avg_doc_length = self.corpus.get_average_doc_length()
            term_doc_freq = self.corpus.termOccurence(term)
            num_docs = self.corpus.documents_count()

            idf = math.log((num_docs - term_doc_freq + 0.5) / (term_doc_freq + 0.5) + 1)
            denom = term_freq_in_doc + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
            score = idf * (term_freq_in_doc * (self.k1 + 1) / denom) if denom > 0 else 0.0
            return score
    # def rank(self, preprocessed_query: List[str]) -> List[Tuple[int, float]]:
    #     scores = [0.0 for _ in range(self.num_docs)]

    #     for term in preprocessed_query:
    #         if term not in self.doc_freq:
    #             continue
    #         df = self.doc_freq[term]
    #         idf = math.log((self.num_docs - df + 0.5) / (df + 0.5))
    #         for doc_id, doc_counter in enumerate(self.doc_term_counts):
    #             tf = doc_counter.get(term, 0)
    #             if tf == 0:
    #                 continue
    #             denom = tf + self.k1 * (
    #                 1 - self.b + self.b * (self.doc_lengths[doc_id] / self.avg_doc_length)
    #             )
    #             scores[doc_id] += idf * (tf * (self.k1 + 1) / denom)

    #     ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    #     return ranked



if __name__ == "__main__":
    results = BM25Model(Corpus(), k1=1.5, b=0.75).rank(Query("dopamine memoire cerveau"))
    for doc_id, score in results:
        print(doc_id, score)
