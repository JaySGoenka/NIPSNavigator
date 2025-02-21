# src/model_training.py
import multiprocessing
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

class TextIterator:
    """ Iterator for tokenizing texts using gensim's simple_preprocess. """
    def __init__(self, texts):
        self.texts = texts
        
    def __iter__(self):
        for text in self.texts:
            if text:
                yield simple_preprocess(text, deacc=True)
            else:
                yield []

def train_word2vec(text_iterator, vector_size=300, window=5, min_count=5, sg=1, negative=5, epochs=5):
    """
    Train a Word2Vec model on tokenized texts.
    
    Args:
        text_iterator (iterator): Iterator over tokenized texts.
        vector_size (int): Dimensionality of word vectors.
        window (int): Maximum distance between current and predicted word.
        min_count (int): Ignores words with total frequency below this.
        sg (int): 1 for skip-gram; 0 for CBOW.
        negative (int): Negative sampling parameter.
        epochs (int): Number of training epochs.
    
    """
    workers = multiprocessing.cpu_count() - 1
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative
    )
    print("Building vocabulary...")
    model.build_vocab(text_iterator, progress_per=1000)
    print(f"Vocabulary size: {len(model.wv.index_to_key)}")
    print("Training model...")
    model.train(
        text_iterator,
        total_examples=model.corpus_count,
        epochs=epochs,
        report_delay=1
    )
    print("Training complete.")
    return model

def save_word2vec_model(model, file_path='word2vec_v2_final.model'):
    """ Save a Word2Vec model to disk. """
    model.save(file_path)
    print(f"Model saved to {file_path}")

def load_word2vec_model(file_path='word2vec_v2_final.model'):
    """ Load a Word2Vec model from disk. """
    model = Word2Vec.load(file_path)
    print(f"Model loaded from {file_path}")
    return model
