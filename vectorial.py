
import math
from typing import Callable, Dict, List

from corpus import Corpus, Query


class SmartConfig:
    """Encapsulates all weighting functions for SMART notation"""
    tf_func: Callable[[str, Dict[str, int]], float]
    df_func: Callable[[str, Corpus], float]
    norm_func: Callable[[str, Dict[str, int]], float]
    

    def __init__(self, tf_func: Callable, df_func: Callable, norm_func: Callable) -> None:
        self.tf_func = tf_func
        self.df_func = df_func
        self.norm_func = norm_func

    def __str__(self):
        return f"SmartConfig(tf={self.tf_func.__name__}, df={self.df_func.__name__}, norm={self.norm_func.__name__})"


class WeightingFunctions:
    """Static class containing all weighting functions"""
    
    # Term Frequency Functions
    @staticmethod
    def tf_natural(term: str, termFrequencies: Dict[str, int]) -> float:
        return float(termFrequencies.get(term, 0))
    
    @staticmethod
    def tf_logarithm(term: str, termFrequencies: Dict[str, int]) -> float:
        tf = termFrequencies.get(term, 0)
        if tf == 0:
            return 0.0
        return 1.0 + math.log(tf)
    
    @staticmethod
    def tf_augmented(term: str, termFrequencies: Dict[str, int]) -> float:
        tf = termFrequencies.get(term, 0)
        max_tf = max(termFrequencies.values()) if termFrequencies else 0
        if max_tf == 0:
            return 0.0
        return 0.5 + (0.5 * tf) / max_tf
    
    @staticmethod
    def tf_boolean(term: str, termFrequencies: Dict[str, int]) -> float:
        return 1.0 if termFrequencies.get(term, 0) > 0 else 0.0

    @staticmethod
    def tf_log_ave(term: str, termFrequencies: Dict[str, int]) -> float:
        tf = termFrequencies.get(term, 0)
        if tf == 0:
            return 0.0
        avg_tf = sum(termFrequencies.values()) / len(termFrequencies) if termFrequencies else 0
        denominator = 1.0 + math.log(avg_tf) if avg_tf > 0 else 1.0
        return (1.0 + math.log(tf)) / denominator
    
    
    # Document Frequency Functions
    @staticmethod
    def df_none(term: str, corpus: 'Corpus') -> float:
        return 1.0
    
    @staticmethod
    def df_standard(term: str, corpus: 'Corpus') -> float:
        df = corpus.termOccurence(term)
        N = corpus.documents_count()
        if df == 0 or N == 0:
            return 0.0
        return math.log(N / df)
    @staticmethod
    def df_prob(term: str, corpus: 'Corpus') -> float:
        df = corpus.termOccurence(term)
        N = corpus.documents_count()
        if df == 0 or N == 0 or df == N:
            return 0.0
        return max(0.0, math.log((N - df) / df))
    
    # Normalization Functions
    @staticmethod
    def norm_none(term:str, termFrequencies: Dict[str, float]) -> float:
        return termFrequencies[term]
    
    @staticmethod
    def norm_cosine(term:str, termFrequencies: Dict[str, float]) -> float:
        if not termFrequencies:
            return termFrequencies[term]
        norm_factor = math.sqrt(sum(w ** 2 for w in termFrequencies.values()))
        if norm_factor == 0:
            return termFrequencies[term]
        return termFrequencies[term] / norm_factor
    @staticmethod
    def norm_max(term:str, termFrequencies: Dict[str, float]) -> float:
        if not termFrequencies:
            return termFrequencies[term]
        max_tf = max(termFrequencies.values())
        if max_tf == 0:
            return termFrequencies[term]
        return termFrequencies[term] / max_tf
    # @staticmethod
    # def norm_pivoted_unique(weights: Dict[str, float], doc_id: int, corpus: 'Corpus') -> Dict[str, float]:
    #     if not weights:
    #         return weights
    #     doc_length = corpus.documents_count()
    #     pivot = 100
    #     slope = 0.2
    #     norm_factor = slope * doc_length + (1 - slope) * pivot
    #     if norm_factor == 0:
    #         return weights
    #     factor = (1 - slope) * pivot / norm_factor
    #     return {term: w * factor for term, w in weights.items()}
    
    # @staticmethod
    # def norm_byte_size(weights: Dict[str, float], doc_id: int, corpus: 'Corpus') -> Dict[str, float]:
    #     if not weights:
    #         return weights
    #     doc_length = corpus.documents_count(doc_id)
    #     if doc_length == 0:
    #         return weights
    #     norm_factor = math.sqrt(doc_length)
    #     return {term: w / norm_factor for term, w in weights.items()}


# Stage 1: Term Frequency Builder
class TermFrequencyBuilder:
    """First stage builder - selects term frequency function"""
    @staticmethod
    def N() -> 'DocumentFrequencyBuilder':
        """Natural term frequency"""
        tf_func = WeightingFunctions.tf_natural
        return DocumentFrequencyBuilder(tf_func)
    @staticmethod
    def L() -> 'DocumentFrequencyBuilder':
        """Logarithmic term frequency"""
        tf_func = WeightingFunctions.tf_logarithm
        return DocumentFrequencyBuilder(tf_func)

    @staticmethod
    def A() -> 'DocumentFrequencyBuilder':
        """Augmented term frequency"""
        tf_func = WeightingFunctions.tf_augmented
        return DocumentFrequencyBuilder(tf_func)

    @staticmethod
    def B() -> 'DocumentFrequencyBuilder':
        """Boolean term frequency"""
        tf_func = WeightingFunctions.tf_boolean
        return DocumentFrequencyBuilder(tf_func)

    @staticmethod
    def LogAvg() -> 'DocumentFrequencyBuilder':
        """Log average term frequency"""
        tf_func = WeightingFunctions.tf_log_ave
        return DocumentFrequencyBuilder(tf_func)


# Stage 2: Document Frequency Builder
class DocumentFrequencyBuilder:
    """Second stage builder - selects document frequency function"""
    
    def __init__(self, tf_func: Callable):
        self.tf_func = tf_func
    
    def N(self) -> 'NormalizationBuilder':
        """No document frequency weighting"""
        df_func = WeightingFunctions.df_none
        return NormalizationBuilder(self.tf_func, df_func)
    
    def T(self) -> 'NormalizationBuilder':
        """Standard IDF"""
        df_func = WeightingFunctions.df_standard
        return NormalizationBuilder(self.tf_func, df_func)
    def P(self) -> 'NormalizationBuilder':
        """Probabilistic IDF"""
        df_func = WeightingFunctions.df_prob
        return NormalizationBuilder(self.tf_func, df_func)


# Stage 3: Normalization Builder
class NormalizationBuilder:
    """Third stage builder - selects normalization function"""
    
    def __init__(self, tf_func: Callable, df_func: Callable):
        self.tf_func = tf_func
        self.df_func = df_func
    
    def N(self) -> 'SmartConfig':
        """No normalization"""
        norm_func = WeightingFunctions.norm_none
        return SmartConfig(
            tf_func=self.tf_func,
            df_func=self.df_func,
            norm_func=norm_func,
        )
    
    def C(self) -> 'SmartConfig':
        """Cosine normalization"""
        norm_func = WeightingFunctions.norm_cosine
        return SmartConfig(
            tf_func=self.tf_func,
            df_func=self.df_func,
            norm_func=norm_func,
        )
    
    def U(self) -> 'SmartConfig':
        """Pivoted unique normalization"""
        norm_func = WeightingFunctions.norm_pivoted_unique
        return SmartConfig(
            tf_func=self.tf_func,
            df_func=self.df_func,
            norm_func=norm_func,
        )
    
    def B(self) -> 'SmartConfig':
        """Byte size normalization"""
        norm_func = WeightingFunctions.norm_byte_size
        return SmartConfig(
            tf_func=self.tf_func,
            df_func=self.df_func,
            norm_func=norm_func,
        )
    def Max(self) -> 'SmartConfig':
        """Max normalization"""
        norm_func = WeightingFunctions.norm_max
        return SmartConfig(
            tf_func=self.tf_func,
            df_func=self.df_func,
            norm_func=norm_func,
        )


# Final Stage: Creates the SmartConfig
SmartNotation = TermFrequencyBuilder




# Main entry point


class VecotrialModel:
    def __init__(self, corpusSmart: SmartConfig, querySmart: SmartConfig, corpus : 'Corpus') -> None:
        self.corpusSmart = corpusSmart
        self.querySmart = querySmart
        self.corpus = corpus
    
    def getTermsFrequencies(self, query:Query,docId:int) -> tuple[Dict[str,int],Dict[str,int]]:
        queryWeights ={}
        corpusWeights ={}
        for term in query.get_preprocessed_query():
            if term not in queryWeights:
                queryWeights[term] = 0
            queryWeights[term] += 1
            if term not in corpusWeights:
                corpusWeights[term] = 0
            
        for term in self.corpus.get_corpus()[docId]:
            if term not in corpusWeights:
                corpusWeights[term] = 0
            corpusWeights[term] += 1
            if term not in queryWeights:
                queryWeights[term] = 0
        return queryWeights, corpusWeights

    def getNaiveWeights(self, query: Query, docId:int) -> tuple[List[float],List[float]]:
        queryWeights, corpusWeights = self.getTermsFrequencies(query, docId)
        queryWeightList = []
        corpusWeightList = []
        for term, freq in queryWeights.items():
            queryWeightList.append(freq)
            corpusWeightList.append(corpusWeights.get(term))
        return queryWeightList, corpusWeightList
    def getSmartFrequencies(self, query: Query, docId:int) -> tuple[List[float],List[float]]:
        queryFrequencies, corpusFrequencies = self.getTermsFrequencies(query, docId)
        queryVector,corpusVector = {},{}
        queryVectorNormalized,corpusVectorNormalized = {},{}
        # Apply TF and DF
        for term in queryFrequencies:
            queryVector[term] = self.querySmart.tf_func(term, queryFrequencies) * self.querySmart.df_func(term, self.corpus)
            corpusVector[term]= self.corpusSmart.tf_func(term, corpusFrequencies) * self.corpusSmart.df_func(term, self.corpus)
        # Apply normalization
        for term in queryVector:
            queryVectorNormalized[term] = self.querySmart.norm_func(term, queryVector)
            corpusVectorNormalized[term] = self.corpusSmart.norm_func(term, corpusVector)
        return queryVectorNormalized, corpusVectorNormalized
    def getSmartWeights(self, query: Query, docId:int) -> tuple[List[float],List[float]]:
        queryVectorNormalized, corpusVectorNormalized = self.getSmartFrequencies(query, docId)
        queryWeightList = []
        corpusWeightList = []
        for term in queryVectorNormalized:
            queryWeightList.append(queryVectorNormalized[term])
            corpusWeightList.append(corpusVectorNormalized.get(term, 0.0))
        return queryWeightList, corpusWeightList

if __name__ == "__main__":
    vectoralModel = VecotrialModel(SmartNotation.L().T().C(), SmartNotation.B().N().N(), Corpus())
    queryWeights, corpusWeights = vectoralModel.getNaiveWeights(Query("cerveau intelligence"),0)
    smartqueryWeights, smartcorpusWeights = vectoralModel.getSmartWeights(Query("cerveau intelligence"),0)
    print("Naive Weights:")
    print("Query Weights:", queryWeights)
    print("Corpus Weights:", corpusWeights)
    print("\nSMART Weights:")
    print("Query Weights:", smartqueryWeights)
    print("Corpus Weights:", smartcorpusWeights)
    
    
    