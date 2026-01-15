from typing import List, Tuple


class RetrievalModel:
    def build(self, preprocessed_corpus: List[List[str]]) -> None:
        raise NotImplementedError

    def rank(self, preprocessed_query: List[str]) -> List[Tuple[int, float]]:
        raise NotImplementedError
