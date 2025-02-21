# src/summarization.py
import re
import spacy
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration
from tqdm import tqdm

# Load models globally for efficiency
longformer_tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
longformer_model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
nlp = spacy.load("en_core_web_sm")

def extract_sections(text, sections=['Introduction', 'Background', 'Method', 'Results', 'Conclusion']):
    """
    Extract key sections from the text.
    
    Args:
        text (str): Full text.
        sections (list): List of section names to extract.
    
    Returns:
        str: Combined text from the extracted sections.
    """
    extracted = {}
    for section in sections:
        pattern = rf'{section}\s*[\n\r]+(.*?)\s*(?={"|".join(sections)}|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted[section] = match.group(1).strip()
    return " ".join(extracted.values()) if extracted else text

def split_into_sentences(text):
    """
    Split text into sentences using spaCy.
    
    Args:
        text (str): Input text.
    
    Returns:
        list: List of sentence strings.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def clean_text(text):
    """
    Clean text by removing unwanted patterns.
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'\(cid:173\)', '', text)
    text = re.sub(r'\(cid:174\)', '', text)
    text = re.sub(r'[^A-Za-z0-9 .,]', '', text)
    return text

def generate_summary(text, max_chunk_len=1024, summary_max_len=900):
    """
    Generate an abstractive summary for the given text using the LED model.
    
    Args:
        text (str): Input full text.
        max_chunk_len (int): Maximum token length for input.
        summary_max_len (int): Maximum token length for the summary.
    
    Returns:
        str: Generated summary.
    """
    if pd.isna(text) or not text:
        return ""
    combined_text = extract_sections(text)
    combined_text = clean_text(combined_text)
    sentences = split_into_sentences(combined_text)
    top_sentences = " ".join(sentences[:50])
    inputs = longformer_tokenizer(top_sentences, return_tensors="pt", max_length=max_chunk_len, truncation=True)
    summary_ids = longformer_model.generate(
        inputs["input_ids"],
        max_length=summary_max_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = longformer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_text(summary)

def generate_summaries(df, text_column='full_text', summary_column='summary', num_rows=50):
    """
    Generate summaries for a subset of the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the full texts.
        text_column (str): Column name with full texts.
        summary_column (str): Column name to store summaries.
        num_rows (int): Number of rows to process.
    
    Returns:
        pd.DataFrame: DataFrame with a new summary column.
    """
    df_subset = df.head(num_rows).copy()
    tqdm.pandas(desc="Generating Summaries")
    df_subset[summary_column] = df_subset[text_column].progress_apply(
        lambda x: generate_summary(x)
    )
    return df_subset
