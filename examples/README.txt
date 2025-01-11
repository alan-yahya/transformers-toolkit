TRANSFORMER TOOLKIT EXAMPLES
==========================

This directory contains example scripts demonstrating the features and capabilities
of the Transformer Toolkit library.

EXAMPLE FILES
------------
1. basic_usage.py
   - Demonstrates basic text encoding
   - Shows how to initialize models and tokenizers
   - Includes text cleaning and encoding example

2. classification.py
   - Shows text classification functionality
   - Uses pre-trained sentiment analysis model
   - Includes multiple example texts with probability outputs

3. batch_processing.py
   - Demonstrates efficient batch processing
   - Shows how to handle multiple texts
   - Includes memory-efficient batching strategies

4. attention_visualization.py
   - Shows advanced toolkit features
   - Demonstrates different pooling strategies (mean, max, cls)
   - Visualizes attention patterns across different heads
   - Includes attention heatmap visualization for layer analysis
   - Shows how different attention heads specialize in different patterns

5. text_preprocessing.py
   - Demonstrates text cleaning capabilities
   - Shows URL removal and special character handling
   - Includes length control examples

VISUALIZATION FEATURES
--------------------
The attention_visualization.py script includes attention visualization that shows:
- Individual attention head patterns
- Layer-specific attention weights
- Token-to-token relationships
- Attention heatmaps with interpretable coloring
- Multiple attention heads side by side

USAGE
-----
To run any example:
python examples/basic_usage.py
python examples/classification.py
etc.

For attention visualization:
python examples/attention_visualization.py

REQUIREMENTS
-----------
- transformer_toolkit (installed with pip install -e .)
- PyTorch >= 1.8.0
- Transformers >= 4.0.0
- NumPy >= 1.19.0
- Matplotlib (for visualization)
- Seaborn (for heatmaps)

NOTES
-----
- Models will be downloaded automatically on first run
- GPU will be used if available (can be forced to CPU with device="cpu")
- Each example can be run independently
- All examples include detailed comments explaining the functionality
- Visualizations require a display or Jupyter notebook environment

EXAMPLE USAGE
------------
# Basic encoding example
from transformer_toolkit import TransformerModel, CustomTokenizer
model = TransformerModel("bert-base-uncased")
tokenizer = CustomTokenizer("bert-base-uncased")
encoded = model.encode("Hello, world!", tokenizer)

# Attention visualization example
from transformer_toolkit import get_attention_weights
outputs = model.model(**inputs, output_attentions=True)
weights = get_attention_weights(outputs, layer=6)
# See attention_visualization.py for full visualization code