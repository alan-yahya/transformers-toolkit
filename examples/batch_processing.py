from transformer_toolkit import TransformerModel, CustomTokenizer, prepare_input

def batch_processing_example():
    """Demonstrate batch processing capabilities."""
    model = TransformerModel("bert-base-uncased", task="encode")
    tokenizer = CustomTokenizer("bert-base-uncased")

    # Multiple texts
    texts = [
        "First example text",
        "Second example text",
        "Third example text",
        "Fourth example text"
    ]

    # Clean all texts
    cleaned_texts = prepare_input(texts, max_length=128, clean_text=True)

    # Batch encode texts
    print("Processing batches...")
    batched_outputs = tokenizer.batch_encode(cleaned_texts, batch_size=2)
    for i, batch in enumerate(batched_outputs):
        encodings = model.encode(batch, tokenizer, pooling="cls")
        print(f"Batch {i+1} encoding shape: {encodings.shape}")

if __name__ == "__main__":
    batch_processing_example() 