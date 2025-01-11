from transformers import AutoTokenizer
from typing import List, Union

class CustomTokenizer:
    def __init__(self, model_name: str):
        """
        Initialize a tokenizer with enhanced functionality.
        
        Args:
            model_name (str): Name or path of the pretrained tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, text: Union[str, List[str]], **kwargs):
        """
        Tokenize input text with automatic batching and preprocessing.
        
        Args:
            text: Single string or list of strings to tokenize
            **kwargs: Additional arguments to pass to the tokenizer
            
        Returns:
            dict: Tokenized input
        """
        # Handle both single strings and lists
        if isinstance(text, str):
            text = [text]
            
        # Set default kwargs for better usability
        kwargs.setdefault('padding', True)
        kwargs.setdefault('truncation', True)
        kwargs.setdefault('return_tensors', 'pt')
        
        return self.tokenizer(text, **kwargs)
        
    def batch_encode(self, texts: List[str], batch_size: int = 32, **kwargs):
        """
        Encode texts in batches to manage memory efficiently.
        
        Args:
            texts (List[str]): List of texts to encode
            batch_size (int): Size of each batch
            **kwargs: Additional arguments for tokenizer
            
        Returns:
            list: List of encoded batches
        """
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self(batch, **kwargs)
            batches.append(encoded)
        return batches 