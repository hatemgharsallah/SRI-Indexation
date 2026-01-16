import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS, Corpus, Query
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel


class LanguageModel(RetrievalModel):
    def __init__(self, corpus, lambd: float = 0.2) -> None:
        super().__init__(corpus)
        self.lambd = lambd


    def rsv (self, query:Query, doc_id: int) -> float:
        """PLME score for a given query and document ID"""
        doc_terms = self.corpus.get_corpus()[doc_id]
        doc_length = len(doc_terms)
        collection_term_count = sum([len(doc) for doc in self.corpus.get_corpus()])
        

        score = 1.0
        
        for term in query.get_preprocessed_query():
            term_freq_in_doc = self.corpus.termOccurenceInDocument(term,doc_id)
            term_freq_in_collection = self.corpus.termOccurence(term)

            p_td = term_freq_in_doc / doc_length if doc_length > 0 else 0.0
            p_tc = term_freq_in_collection / collection_term_count if collection_term_count > 0 else 0.0

            p_t = (1 - self.lambd) * p_td + self.lambd * p_tc
            score *= p_t if p_t > 0 else 1e-10  # Avoid multiplying by zero

        return score        
        




if __name__ == "__main__":
    results = LanguageModel(Corpus(), 0.2).rank(Query("neurone"))
    for doc_id, score in results:
        print(doc_id, score)
