from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils.text_processing import chunk_text, clean_text

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(text, max_length=150, min_length=30, chunk_size=1024):
    """
    Summarize long text by processing it in chunks
    """
    # Clean and prepare text
    text = clean_text(text)
    
    # If text is short, process directly
    if len(text.split()) < 500:
        inputs = tokenizer([text], max_length=chunk_size, truncation=True, return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # For long texts, process in chunks and combine
    chunks = chunk_text(text, chunk_size=chunk_size)
    summaries = []
    
    for chunk in chunks:
        inputs = tokenizer([chunk], max_length=chunk_size, truncation=True, return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # Combine chunk summaries and summarize again if needed
    combined_summary = " ".join(summaries)
    if len(combined_summary.split()) > 500:
        return summarize_text(combined_summary, max_length, min_length)
    
    return combined_summary