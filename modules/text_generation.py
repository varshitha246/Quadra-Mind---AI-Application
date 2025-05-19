from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from utils.text_processing import clean_text

# Load model and tokenizer
model_name = "gpt2-medium"  # Can be changed to "gpt2-large" or other models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, length=100, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate coherent text based on a prompt using GPT-2
    Handles long generation by chunking
    """
    # Clean and prepare prompt
    prompt = clean_text(prompt)
    
    # Generate text in chunks if needed
    if length <= 100:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=length + len(inputs[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # For longer generation, use chunking
        generated = prompt
        remaining_length = length
        
        while remaining_length > 0:
            chunk_length = min(100, remaining_length)
            inputs = tokenizer.encode(generated, return_tensors="pt")
            
            # Prevent input from being too long
            if len(inputs[0]) > 1024:
                inputs = inputs[:, -1024:]
            
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + chunk_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = new_text
            remaining_length -= chunk_length
        
        return generated