import re
import unicodedata
from typing import List, Dict, Set
from nltk.stem.snowball import SnowballStemmer
STOPWORDS = [
    "le", "la", "les", "l", 
    "de", "d", "des", "du", 
    "un", "une", 
    "et", "ou",
    "en", "eux",  "lui",
    "à", "au", "aux", 
    "pour", "dans", "entre","sur", "avec", "comme", "grace", "après", "avant",
    "ne", "pas","est", "sont", "s", "se","a","ont",
    "ce", "cette", "ces", 
    "il", "elle", "elles", "on", "nous", 
    "qui", "que", "dont", 
    "par","milliards", "environ", "autre","autres"
]


def tokenize(text: str) -> List[str]:
        # Replace apostrophes with space
        text = re.sub(r"[’'.]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        # Split and remove empty tokens
        tokens = [token for token in text.split() if token]
        return tokens
    
def filter_tokens(tokens: List[str], stopwords: List[str] ) -> List[str]:
        """Filters out French stopwords and numeric tokens from a list of tokens."""
        filtered = [ 
            token for token in tokens
            if token.lower() not in stopwords and not token.isdigit()
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
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

        return text
    
def stem_tokens(tokens: List[str]) -> List[str]:
        """Stems a list of tokens using NLTK SnowballStemmer."""
        stemmer = SnowballStemmer("french")

        stems = [stemmer.stem(token) for token in tokens]
        return stems

def preprocess_document(doc: str) -> List[str]:
        """ Preprocesses a single document: tokenization, stopword removal, normalization, lemmatization """
        tokens = tokenize(doc)
        filtered = filter_tokens(tokens,STOPWORDS)
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
    
    
def build_inverted_index(corpus_tokens: List[List[str]]) -> Dict[str, List[int]]:
        """  Builds a simple inverted index.It returns:  Dictionary mapping token -> sorted document IDs """
        inverted_index = {}

        for doc_id, tokens in enumerate(corpus_tokens):
            for token in tokens:
                if token not in inverted_index:
                    inverted_index[token] = set()
                inverted_index[token].add(doc_id)
        
        # Convert sets to sorted lists
        for token in inverted_index:
            inverted_index[token] = sorted(inverted_index[token])

        return inverted_index

def vbe_delta_compress(index: Dict[str, List[int]]) -> Dict[str, bytes]:
    """Compress an inverted index using Delta Encoding + VarByte."""
    compressed_index = {}
    for term, doc_ids in index.items():
        gap_compress_index = delta_encode(doc_ids)
        vbe_compress_index = vbe_compress(gap_compress_index)
        compressed_index[term] = vbe_compress_index

    return compressed_index

def delta_encode(sorted_ids):
    """Apply delta encoding to a sorted posting list."""
    if not sorted_ids:
        return []
    return [sorted_ids[0]] + [sorted_ids[i] - sorted_ids[i-1] for i in range(1, len(sorted_ids))]

def delta_decode(deltas):
    """Decode delta-encoded list back to absolute document IDs."""
    if not deltas:
        return []
    output = [deltas[0]]
    for i in range(1, len(deltas)):
        output.append(output[-1] + deltas[i])
    return output

def vbe_compress(numbers: List[int]) -> bytes:
    """Compress a list of integers using classic VBE (MSB=stop bit)."""
    result = bytearray()
    for num in numbers:
        bytes_list = []
        while True:
            b = num & 0b01111111  # take 7 least significant bits
            bytes_list.insert(0, b)  # prepend
            num >>= 7
            if num == 0:
                break
        # Set stop bit (MSB) on the last byte
        bytes_list[-1] |= 0b10000000
        result.extend(bytes_list)
    return bytes(result)

def vbe_decompress(data: bytes) -> List[int]:
    """Decompress VBE bytes into a list of integers."""
    numbers = []
    current = 0
    for byte in data:
        if byte & 0b10000000:  # stop bit = 1 → last byte
            current = (current << 7) | (byte & 0b01111111)
            numbers.append(current)
            current = 0
        else:  # stop bit = 0 → more bytes follow
            current = (current << 7) | byte
    return numbers

def vbe_delta_decompress(compressed_index: Dict[str, bytes]) -> Dict[str, List[int]]:
    """Compress an inverted index using Delta Encoding + VarByte."""
    decompressed_index = {}
    for term, compressed_doc_ids in compressed_index.items():
        vbe_decompressed = vbe_decompress(compressed_doc_ids)
        delta_decompressed = delta_decode(vbe_decompressed)
        decompressed_index[term] = delta_decompressed

    return decompressed_index

def vbe_delta_compress(index: Dict[str, List[int]]) -> Dict[str, bytes]:
    """Compress an inverted index using Delta Encoding + VarByte."""
    compressed_index = {}
    for term, doc_ids in index.items():
        gap_compress_index = delta_encode(doc_ids)
        vbe_compress_index = vbe_compress(gap_compress_index)
        compressed_index[term] = vbe_compress_index

    return compressed_index