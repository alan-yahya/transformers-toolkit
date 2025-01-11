from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class TransformerModel:
    def __init__(self, model_name: str, task: str = "encode", device: str = None):
        """
        Initialize a transformer model with task-specific configuration.
        
        Args:
            model_name (str): Name or path of the pretrained model
            task (str): Task type ("encode", "classify", "generate")
            device (str, optional): Device to load the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = task
        
        if task == "classify":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def encode(self, input_text: str, tokenizer, pooling: str = "mean"):
        """
        Encode input text using the model with flexible pooling strategies.
        
        Args:
            input_text (str): Text to encode
            tokenizer: Tokenizer instance
            pooling (str): Pooling strategy ("mean", "max", "cls")
        
        Returns:
            torch.Tensor: Encoded representation
        """
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling strategy
        if pooling == "mean":
            return torch.mean(hidden_states, dim=1)
        elif pooling == "max":
            return torch.max(hidden_states, dim=1)[0]
        elif pooling == "cls":
            return hidden_states[:, 0, :]
        
    def classify(self, input_text: str, tokenizer):
        """
        Perform classification on input text.
        
        Args:
            input_text (str): Text to classify
            tokenizer: Tokenizer instance
            
        Returns:
            dict: Classification logits and probabilities
        """
        if self.task != "classify":
            raise ValueError("Model not initialized for classification task")
            
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "probabilities": probs
        } 