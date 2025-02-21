# src/embeddings.py
import numpy as np
from collections import Counter

def create_vocab(texts, max_vocab_size=10000):
    """ Create a vocabulary from texts based on word frequency. """
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    most_common = word_counts.most_common(max_vocab_size)
    vocab = [word for word, _ in most_common]
    return vocab

def compute_document_embeddings(texts, vocab, w2v_model):
    """ Compute document embeddings by averaging the Word2Vec embeddings of words. """
    doc_embeddings = []
    for text in texts:
        words = [word for word in text.split() if word in vocab and word in w2v_model.wv]
        if words:
            embedding = np.mean([w2v_model.wv[word] for word in words], axis=0)
        else:
            embedding = np.zeros(w2v_model.vector_size)
        doc_embeddings.append(embedding)
    return np.array(doc_embeddings)

def get_query_embedding(query, vocab, w2v_model):
    """ Compute an embedding for a query by averaging its word vectors. """
    words = [word for word in query.split() if word in vocab and word in w2v_model.wv]
    if words:
        return np.mean([w2v_model.wv[word] for word in words], axis=0)
    else:
        return np.zeros(w2v_model.vector_size)
