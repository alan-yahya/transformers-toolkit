TRANSFORMER TOOLKIT
==================

A Python library for working with transformer models and utility functions.

![Comparison of attn heads](./position_sensitivity.png)

## Installation 


`git clone https://github.com/alan-yahya/transformers-toolkit`

`cd transformer_toolkit`

`pip install -e .`

## Quick Start

from transformer_toolkit import TransformerModel, CustomTokenizer

Check out the `examples` directory for detailed examples:

1. Basic Usage (`examples/basic_usage.py`)
2. Classification (`examples/classification.py`)
3. Batch Processing (`examples/batch_processing.py`)
4. Attention Visualization (`examples/attention_visualization.py`)
5. Text Preprocessing (`examples/text_preprocessing.py`)

## Requirements

- Python >= 3.7
- PyTorch >= 1.8.0
- Transformers >= 4.0.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0 (for visualization)
- Seaborn >= 0.11.0 (for heatmaps)

## License

MIT License
