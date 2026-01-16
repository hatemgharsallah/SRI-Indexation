from typing import List, Tuple

from corpus import Corpus, Query


class RetrievalModel:
    def __init__(self,corpus:Corpus):
        self.corpus=corpus

    def rank(self, query:Query,limit: int=-1) -> List[Tuple[int, float]]:
        ranked = []
        for doc_id in range(self.corpus.documents_count()):
            score = self.rsv(query, doc_id)
            ranked.append((doc_id, score))
        ranked = sorted(ranked, key=lambda item: item[1], reverse=True)
        return ranked[:limit] if limit > 0 else ranked

    def rsv(self, query:Query, doc_id: int) -> List[Tuple[int, float]]:
        raise NotImplementedError