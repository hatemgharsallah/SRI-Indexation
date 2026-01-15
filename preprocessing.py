import re
import unicodedata
from typing import List

from nltk.stem.snowball import SnowballStemmer


def tokenize(text: str) -> List[str]:
    # Replace apostrophes with space
    text = re.sub(r"[’'.]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split and remove empty tokens
    tokens = [token for token in text.split() if token]

    return tokens


stopwords = [
    "le",
    "la",
    "les",
    "l",
    "de",
    "d",
    "des",
    "du",
    "un",
    "une",
    "et",
    "ou",
    "en",
    "eux",
    "lui",
    "à",
    "au",
    "aux",
    "pour",
    "dans",
    "entre",
    "sur",
    "avec",
    "comme",
    "grace",
    "après",
    "avant",
    "ne",
    "pas",
    "est",
    "sont",
    "s",
    "se",
    "a",
    "ont",
    "ce",
    "cette",
    "ces",
    "il",
    "elle",
    "elles",
    "on",
    "nous",
    "qui",
    "que",
    "dont",
    "par",
    "milliards",
    "environ",
    "autre",
    "autres",
]


def filter_tokens(tokens: List[str], stopwords: List[str] = stopwords) -> List[str]:
    """Filters out French stopwords and numeric tokens from a list of tokens."""
    filtered = [
        token for token in tokens if token.lower() not in stopwords and not token.isdigit()
    ]
    return filtered


def normalize_text(text: str) -> str:
    """
    Normalizes text by removing punctuation, lowercasing, removing accents.
    """
    # 1. Remove punctuation
    text = re.sub(r"[.,;:!?()\[\]\"']", "", text)
    if text.strip() == "":
        # text is empty after punctuation removal
        return None

    # 2. Lowercase
    text = text.lower()

    # 3. Remove accents :
    ## Decomposes accented characters into base + accent.
    ## Removes all accents (diacritics).
    ## Joins the remaining characters back into a plain ASCII-like string.
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    return text


stemmer = SnowballStemmer("french")


def stem_tokens(tokens: List[str]) -> List[str]:
    """Stems a list of tokens using NLTK SnowballStemmer."""
    stems = [stemmer.stem(token) for token in tokens]
    return stems


def preprocess_document(doc: str) -> List[str]:
    """ Preprocesses a single document: tokenization, stopword removal, normalization, lemmatization """
    tokens = tokenize(doc)
    filtered = filter_tokens(tokens)
    normalized = [normalize_text(token) for token in filtered]
    stemmed = stem_tokens(normalized)
    return stemmed


def preprocess_corpus(corpus: List[str]) -> List[List[str]]:
    """ Preprocesses a corpus of documents: tokenization, stopword removal, normalization, lemmatization """
    preprocessed_corpus = []
    for doc in corpus:
        stemmed = preprocess_document(doc)
        preprocessed_corpus.append(stemmed)
    return preprocessed_corpus
