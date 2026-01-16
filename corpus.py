import re
from typing import List
import unicodedata
from preprocessing import * 

CORPUS = [
    "Le cerveau humain contient environ 86 milliards de neurones.",
    "Les neurones communiquent entre eux grâce aux synapses et aux neurotransmetteurs.",
    "La plasticité cérébrale permet au cerveau de s'adapter aux expériences et à l'apprentissage.",
    "Les neurotransmetteurs comme la dopamine influencent l'humeur, la motivation et la mémoire.",
    "L'hippocampe joue un rôle clé dans la formation des souvenirs et la mémoire du cerveau.",
    "Les neurones moteurs transmettent les signaux du cerveau aux muscles et aux organes.",
    "La mémoire à court terme dépend fortement du cortex préfrontal et de l'activité neuronale.",
    "Les synapses excitatrices et inhibitrices régulent l'activité des neurones et la communication cérébrale.",
    "La dopamine est essentielle pour le système de récompense du cerveau et pour l'apprentissage.",
    "Les circuits neuronaux peuvent se réorganiser après une lésion cérébrale ou des traumatismes.",
    "Le cortex visuel et le cortex préfrontal sont responsables du traitement des informations visuelles et cognitives.",
    "Les axones permettent de transmettre les signaux électriques sur de longues distances entre neurones.",
    "La plasticité synaptique est la base de l'apprentissage, de la mémoire et de la communication neuronale.",
    "Les neurones sensoriels détectent les stimuli environnementaux et transmettent l'information au cerveau.",
    "Le cerveau droit et le cerveau gauche ont des fonctions complémentaires dans le traitement des informations.",
    "La sérotonine et la dopamine régulent l'humeur, le sommeil, l'appétit et la motivation.",
    "Les dendrites reçoivent les signaux des autres neurones et participent à la plasticité cérébrale.",
    "Les maladies neurodégénératives affectent la structure et la fonction des neurones et des synapses.",
    "L'amygdale est impliquée dans les émotions, la peur et le traitement des souvenirs.",
    "L'axone et la dendrite sont des structures essentielles pour la communication neuronale et la transmission des signaux.",
]



class Corpus: 
    def __init__(self, corpus: List[str] = CORPUS) -> None:
        self.__preprocessed_corpus = preprocess_corpus(corpus)
        self.__compressed_index = vbe_delta_compress(build_inverted_index(self.__preprocessed_corpus))



    def get_corpus(self) -> List[List[str]]:
        return self.__preprocessed_corpus 

    def terms_count(self) -> int:
        return len(self.__index)
    def documents_count(self) -> int:
        return len(self.get_corpus())
    def termOccurence(self, term: str) -> int:
        """ Returns the number of documents containing the term """
        if term in self.get_index():
            return len(self.get_index()[term])
        return 0
    def termOccurenceInDocument(self, term: str, doc_id: int) -> int:
        """ Returns the number of occurrences of the term in the specified document """
        if term in self.get_index():
            return self.__preprocessed_corpus[doc_id].count(term)
        return 0
    
    def get_index(self) -> Dict[str, List[int]]:
        return vbe_delta_decompress(self.__compressed_index)
    
    def get_average_doc_length(self) -> float:
        total_length = sum(len(doc) for doc in self.__preprocessed_corpus)
        return total_length / len(self.__preprocessed_corpus) if self.__preprocessed_corpus else 0.0
    
    def get_doc_length(self, doc_id: int) -> int:
        """ Returns the length of the specified document """
        if doc_id < 0 or doc_id >= len(self.__preprocessed_corpus):
            return 0
        return len(self.__preprocessed_corpus[doc_id])
    
    
    def get_max_term_freq_in_doc(self, doc_id: int) -> int:
        """ Returns the maximum term frequency in the specified document """
        if doc_id < 0 or doc_id >= len(self.__preprocessed_corpus):
            return 0
        term_freqs = {}
        for term in self.__preprocessed_corpus[doc_id]:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        return max(term_freqs.values()) if term_freqs else 0
    def get_avg_term_freq_in_doc(self, doc_id: int) -> float:
        """ Returns the average term frequency in the specified document """
        if doc_id < 0 or doc_id >= len(self.__preprocessed_corpus):
            return 0.0
        term_freqs = {}
        for term in self.__preprocessed_corpus[doc_id]:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        total_terms = len(self.__preprocessed_corpus[doc_id])
        return total_terms / len(term_freqs) if term_freqs else 0.0
class Query:
    def __init__(self, query: str) -> None:
        self.__preprocessed_query = preprocess_document(query)

    def get_preprocessed_query(self) -> List[str]:
        return self.__preprocessed_query
    def __str__(self) -> str:
        return ' '.join(self.__preprocessed_query)

if __name__ == "__main__":
    corpus_instance = Corpus(CORPUS)

    print("Number of terms in the corpus:", corpus_instance.terms_count())
    print("cerveau occurs in", corpus_instance.termOccurence("cerveau"), "documents.")
    print("cerveau occurs", corpus_instance.termOccurenceInDocument("cerveau", 0), "times in document 0.")

        


