import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def clean_text(text):
    """Basic text cleaning"""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def split_sentences(text):
    """Split text into sentences"""
    return sent_tokenize(text)