import math
from collections import Counter
from typing import Dict, List, Tuple

from corpus import CORPUS, Corpus, Query
from preprocessing import preprocess_corpus, preprocess_document
from retrieval_model import RetrievalModel
from vectorial import SmartNotation, VecotrialModel


class ExtendedBooleanModel(RetrievalModel):
    def __init__(self,corpus:Corpus, p: float = 2.0) -> None:
        super().__init__(corpus)
        self.p = p
        self.vectorialModel = VecotrialModel(corpusSmart=SmartNotation.N().T().Max(), querySmart=SmartNotation.B().N().N(), corpus=self.corpus)

    def rsv(self, query:Query, doc_id: int) -> float:
        query_weights, terms_weight = self.vectorialModel.getSmartWeights(query, doc_id)
        
        termsNumber= sum(query_weights)
        sum_weights = 0.0
        for term_index in range(len(terms_weight)):
            if query_weights[term_index]>0:
                sum_weights += math.pow(1-terms_weight[term_index], self.p)
        return 1 - math.pow(sum_weights / termsNumber, 1.0 / self.p)



if __name__ == "__main__":
    results = ExtendedBooleanModel(Corpus(),2).rank(Query("dopamine memoire cerveau"))
    for doc_id, score in results:
        print(doc_id, score)
