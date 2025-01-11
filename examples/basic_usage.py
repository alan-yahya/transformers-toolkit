from transformer_toolkit import TransformerModel, CustomTokenizer, prepare_input

def basic_encoding_example():
    """Demonstrate basic text encoding."""
    # Initialize model and tokenizer
    model = TransformerModel("bert-base-uncased", task="encode")
    tokenizer = CustomTokenizer("bert-base-uncased")

    # Clean and encode text
    text = "Check out https://example.com! This is a sample text with some noise...   "
    cleaned_text = prepare_input(text, max_length=512, clean_text=True)
    encoded = model.encode(cleaned_text, tokenizer, pooling="mean")
    print(f"Encoded shape: {encoded.shape}")

if __name__ == "__main__":
    basic_encoding_example() 