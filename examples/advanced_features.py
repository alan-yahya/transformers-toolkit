from transformer_toolkit import (
    TransformerModel, 
    CustomTokenizer, 
    prepare_input, 
    get_attention_weights
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def pooling_strategies_example():
    """Demonstrate different pooling strategies."""
    model = TransformerModel("bert-base-uncased", task="encode")
    tokenizer = CustomTokenizer("bert-base-uncased")
    
    text = "Let's try different pooling strategies!"
    
    # Try different pooling methods
    for pooling in ["mean", "max", "cls"]:
        encoded = model.encode(text, tokenizer, pooling=pooling)
        print(f"\n{pooling.upper()} pooling shape: {encoded.shape}")

def plot_attention_head(weights, tokens, head_idx, title):
    """
    Plot attention weights for a specific attention head.
    
    Args:
        weights: Attention weights tensor
        tokens: List of tokens
        head_idx: Index of the attention head to plot
        title: Plot title
    """
    # Get weights for specific head
    attention = weights[0, head_idx].detach().cpu().numpy()  # [seq_len, seq_len]
    
    # Create heatmap
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        vmin=0,
        vmax=np.max(attention)
    )
    
    plt.title(title)
    plt.xlabel('Target Token')
    plt.ylabel('Source Token')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

def attention_visualization_example():
    """Demonstrate attention patterns across different heads."""
    model = TransformerModel("bert-base-uncased", task="encode")
    tokenizer = CustomTokenizer("bert-base-uncased")
    device = model.device
    
    # Example sentence
    text = "The cat sat on the mat."
    
    # Get tokens and model outputs
    tokens_dict = tokenizer.tokenizer(text, return_tensors=None)
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(tokens_dict['input_ids'])
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.model(**inputs, output_attentions=True)
    
    # Get attention weights for middle layer (layer 6)
    weights = get_attention_weights(outputs, layer=6)
    
    # Plot first 6 attention heads
    plt.figure(figsize=(20, 10))
    for head_idx in range(6):
        plt.subplot(2, 3, head_idx + 1)
        plot_attention_head(
            weights, 
            tokens, 
            head_idx, 
            f'Layer 6, Head {head_idx}\n"{text}"'
        )
    
    plt.show()

if __name__ == "__main__":
    print("Running pooling strategies example...")
    pooling_strategies_example()
    
    print("\nRunning attention visualization example...")
    attention_visualization_example() 