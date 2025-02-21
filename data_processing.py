import re
import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy model globally for efficiency
nlp = spacy.load("en_core_web_sm")

# Define additional stop words specific to the domain
ADDITIONAL_STOP_WORDS = {
    "figure", "table", "reference", "chapter", "section", "appendix", "bibliography",
    "cite", "citation", "abstract", "introduction", "conclusion", "method", "results",
    "discussion", "acknowledgement", "acknowledgments", "appendices", "references"
}
STOP_WORDS = STOP_WORDS.union(ADDITIONAL_STOP_WORDS)

def preprocess_texts(texts, batch_size=50, n_process=4):
    """
    Clean and preprocess the text for each paper by removing unwanted sections, HTML tags,
    extra spaces, and applying tokenization, lemmatization, and stop-word removal.
    Parameters:
        texts (list of str): The texts to preprocess.
        batch_size (int): Number of texts to process per batch.
        n_process (int): Number of processes to use for parallel processing via spaCy's nlp.pipe.
    
    """
    cleaned_texts = []
    # Regex patterns for sections to remove:
    # - Remove text from "abstract" until "introduction"
    # - Remove acknowledgments until "references"
    # - Remove text from "references" or "appendix" to the end of the text.
    sections_to_remove = [
        r'\babstract\b.*?(?=\bintroduction\b)',
        r'\backnowledgments?\b.*?(?=\breferences?\b)',
        r'\breferences?\b.*$',
        r'\bappendix\b.*$',
    ]
    sections_regex = re.compile('|'.join(sections_to_remove), re.DOTALL | re.IGNORECASE)
    
    for text in texts:
        if pd.notnull(text):
            text = re.sub(r'<.*?>', '', text)           # Remove HTML tags
            text = sections_regex.sub('', text)         # Remove specific sections
            text = re.sub(r'\s+', ' ', text)            # Normalize whitespace
            text = re.sub(r'[^a-zA-Z\s]', '', text)     # Keep alphabets only
            text = text.strip().lower()                 # Trim and lowercase
            cleaned_texts.append(text)
        else:
            cleaned_texts.append('')
    
    processed_texts = []
    # Use spaCy's pipe with batching and optional multiprocessing for efficiency.
    for doc in nlp.pipe(cleaned_texts, batch_size=batch_size, n_process=n_process, disable=["ner", "parser"]):
        tokens = [
            token.lemma_ for token in doc
            if token.text not in STOP_WORDS
            and not token.is_punct
            and not token.is_digit
            and token.is_alpha
            and len(token.text) > 2
            # Although the text is lowercased, these regex checks are maintained for explicit filtering.
            and not re.match(r'^[A-Z]{2,}$', token.text)
            and not re.match(r'^[a-zA-Z]*[0-9]+[a-zA-Z]*$', token.text)
        ]
        processed_text = " ".join(tokens)
        # Post-token cleanup: remove isolated letters and extra spaces.
        processed_text = re.sub(r'\b[a-zA-Z]\b', '', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        processed_texts.append(processed_text)
    
    return processed_texts

def load_dataset(filepath, limit_rows=None, batch_size=50, n_process=1):
    """ Load the NIPS CSV dataset, dropping unwanted columns/rows, and preprocessing the text. """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    if 'full_text' not in df.columns:
        raise ValueError("Dataset must contain a 'full_text' column.")
    
    # Optionally drop the 'abstract' column if present
    if 'abstract' in df.columns:
        df = df.drop(columns=['abstract'])
    
    # Drop rows where 'full_text' is missing
    df = df.dropna(subset=['full_text'])
    
    if limit_rows:
        df = df.head(limit_rows)
    
    df['processed_text'] = preprocess_texts(df['full_text'].tolist(), batch_size=batch_size, n_process=n_process)
    return df


def save_processed_dataset(df, filepath):
    """ Save the processed DataFrame to a CSV file. """
    df.to_csv(filepath, index=False)
