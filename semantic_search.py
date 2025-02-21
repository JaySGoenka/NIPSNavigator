# src/semantic_search.py
import numpy as np
import pandas as pd
from .embeddings import get_query_embedding

def semantic_search(query, df, vocab, w2v_model, top_n=5):
    """
    Perform semantic search over document embeddings using cosine similarity.
    
    Args:
        query (str): User query.
        df (pd.DataFrame): DataFrame containing document embeddings.
        vocab (list): Vocabulary list.
        w2v_model (Word2Vec): Trained Word2Vec model.
        top_n (int): Number of top results to return.
    
    Returns:
        pd.DataFrame: Top search results with title, summary, and keywords.
    """
    query_embedding = get_query_embedding(query, vocab, w2v_model)
    similarities = np.array([
        np.dot(query_embedding, doc_embedding) for doc_embedding in df['document_embedding']
    ])
    top_indices = similarities.argsort()[::-1][:top_n]
    return df.iloc[top_indices][['title', 'summary', 'keywords']]
