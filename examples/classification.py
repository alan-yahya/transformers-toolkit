from transformer_toolkit import TransformerModel, CustomTokenizer

def classification_example():
    """Demonstrate text classification."""
    # Initialize model for classification
    classifier = TransformerModel(
        "distilbert-base-uncased-finetuned-sst-2-english", 
        task="classify"
    )
    tokenizer = CustomTokenizer("distilbert-base-uncased-finetuned-sst-2-english")

    # Example texts
    texts = [
        "This movie was absolutely fantastic!",
        "The service was terrible and I wouldn't recommend it.",
        "It was okay, nothing special."
    ]

    # Classify each text
    for text in texts:
        result = classifier.classify(text, tokenizer)
        probs = result["probabilities"]
        print(f"\nText: {text}")
        print(f"Probabilities: {probs}")

if __name__ == "__main__":
    classification_example() 