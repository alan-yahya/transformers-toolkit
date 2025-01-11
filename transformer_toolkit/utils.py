import torch
from typing import List, Union
import re

def prepare_input(text: Union[str, List[str]], max_length: int = 512, clean_text: bool = True):
    """
    Prepare input text for transformer processing with enhanced cleaning options.
    
    Args:
        text: Input text or list of texts
        max_length (int): Maximum sequence length
        clean_text (bool): Whether to apply text cleaning
        
    Returns:
        Union[str, List[str]]: Processed text(s)
    """
    def clean(t: str) -> str:
        if clean_text:
            # Remove extra whitespace
            t = re.sub(r'\s+', ' ', t.strip())
            # Remove URLs
            t = re.sub(r'http\S+|www.\S+', '', t)
            # Remove special characters but keep punctuation
            t = re.sub(r'[^\w\s.,!?-]', '', t)
        return t[:max_length]
    
    if isinstance(text, str):
        return clean(text)
    return [clean(t) for t in text]

def get_attention_weights(model_outputs, layer: int = -1):
    """
    Extract attention weights from transformer outputs.
    
    Args:
        model_outputs: Transformer model outputs
        layer (int): Which layer's attention weights to extract
        
    Returns:
        torch.Tensor: Attention weights
    """
    if hasattr(model_outputs, 'attentions'):
        return model_outputs.attentions[layer]
    raise ValueError("Model outputs don't contain attention weights") 