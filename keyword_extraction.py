import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=embedding_model)

def extract_keywords_from_text(text, num_keywords=20, num_clusters=5, keyphrase_ngram_range=(1, 3), stop_words="english"):
    """
    Extract keywords using KeyBERT and cluster them for diversity.
    
    Args:
        text (str): Input text.
        num_keywords (int): Number of keywords to extract.
        num_clusters (int): Number of clusters to enforce diversity.
        keyphrase_ngram_range (tuple): N-gram range for keyword extraction.
        stop_words (str or list): Stop words to be used.
    
    """
    if pd.isna(text) or not text:
        return []
    
    try:
        # Extract keywords using KeyBERT
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            top_n=num_keywords
        )
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []
    
    keyword_list = [kw[0] for kw in keywords if kw and kw[0]]
    
    # If no keywords were extracted, return an empty list
    if not keyword_list:
        return []
    
    try:
        # Compute embeddings for the keywords
        keyword_embeddings = embedding_model.encode(keyword_list)
    except Exception as e:
        logging.error(f"Error encoding keywords: {e}")
        return keyword_list
    
    # Determine number of clusters to use
    n_clusters = min(num_clusters, len(keyword_list))
    if n_clusters <= 0:
        return keyword_list
    
    try:
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        clusters = clustering_model.fit_predict(keyword_embeddings)
    except Exception as e:
        logging.error(f"Error during clustering: {e}")
        return keyword_list
    
    # Select one representative keyword per cluster
    clustered_keywords = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_keywords:
            clustered_keywords[cluster_id] = keyword_list[idx]
    
    return list(clustered_keywords.values())

def update_dataframe_with_keywords(df, text_column='processed_text', keyword_column='keywords', num_keywords=20, num_clusters=5, num_rows=1000):
    """ Extract keywords for a subset of the DataFrame and add them as a new column."""
    if text_column not in df.columns:
        raise ValueError(f"DataFrame does not contain the '{text_column}' column.")
    
    df_subset = df.head(num_rows).copy()
    tqdm.pandas(desc="Extracting Keywords")
    df_subset[keyword_column] = df_subset[text_column].progress_apply(
        lambda x: extract_keywords_from_text(
            x, 
            num_keywords=num_keywords, 
            num_clusters=num_clusters
        )
    )
    return df_subset
