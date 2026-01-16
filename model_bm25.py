import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS, Query
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

    def rsv(self, query: Query, doc_id: int) -> float:
        return self.rank([query])[doc_id][1] if doc_id < len(self.rank([query])) else 0.0





if __name__ == "__main__":
    results = BM25.rank_query("dopamine memoire cerveau")
    for doc_id, score in results[:5]:
        print(doc_id, score)
