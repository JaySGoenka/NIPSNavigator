# streamlit_app.py
import streamlit as st
import pandas as pd
from data_processing import load_dataset
from model_training import load_word2vec_model, TextIterator, train_word2vec, save_word2vec_model
from embeddings import create_vocab, compute_document_embeddings
from semantic_search import semantic_search
from summarization import generate_summaries
from keyword_extraction import update_dataframe_with_keywords

# Cache expensive operations using Streamlit's caching
@st.cache(allow_output_mutation=True)
def load_data():
    # Using default batch_size and n_process values as defined in load_dataset
    df = load_dataset("data/papers.csv", limit_rows=1000)
    df.to_csv("data/processed_nips_dataset_final.csv", index=False)
    return df

@st.cache(allow_output_mutation=True)
def load_models():
    # Try loading a saved Word2Vec model; if unavailable, train a new one.
    try:
        w2v_model = load_word2vec_model("data/word2vec_v2_final.model")
    except Exception as e:
        df = load_data()
        text_iter = TextIterator(df['processed_text'])
        w2v_model = train_word2vec(text_iter)
        save_word2vec_model(w2v_model, "data/word2vec_v2_final.model")
    return w2v_model

@st.cache(allow_output_mutation=True)
def prepare_embeddings(df, w2v_model):
    vocab = create_vocab(df['processed_text'].tolist(), max_vocab_size=10000)
    doc_embeddings = compute_document_embeddings(df['processed_text'].tolist(), vocab, w2v_model)
    df['document_embedding'] = list(doc_embeddings)
    return df, vocab

def main():
    st.title("NIPS Papers Semantic Search and Summarization")
    
    with st.spinner("Loading dataset..."):
        df = load_data()
    st.write(f"Dataset loaded with {df.shape[0]} papers.")
    
    with st.spinner("Loading models..."):
        w2v_model = load_models()
    
    with st.spinner("Preparing embeddings..."):
        df, vocab = prepare_embeddings(df, w2v_model)
    
    # Generate summaries and keywords if they are not already present
    if 'summary' not in df.columns or df['summary'].isnull().all():
        st.write("Generating summaries...")
        with st.spinner("Generating summaries..."):
            df = generate_summaries(df, text_column='full_text', summary_column='summary', num_rows=100)
    if 'keywords' not in df.columns or df['keywords'].isnull().all():
        st.write("Extracting keywords...")
        with st.spinner("Extracting keywords..."):
            df = update_dataframe_with_keywords(df, text_column='processed_text', keyword_column='keywords', num_rows=100)
    
    st.write("Ready for semantic search!")
    query = st.text_input("Enter your search query:")
    
    if query:
        with st.spinner("Searching..."):
            results = semantic_search(query, df, vocab, w2v_model, top_n=5)
        st.write("Top Relevant Papers:")
        st.table(results)
    
if __name__ == "__main__":
    main()
