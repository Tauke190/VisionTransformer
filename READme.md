# Vision Transformer (ViT) Implementation

A from-scratch PyTorch implementation of the Vision Transformer architecture from ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929).

## Architecture

The implementation includes:
- **PatchEmbedding**: Converts images into patch embeddings using Conv2d
- **MultiHeadSelfAttentionBlock**: Multi-head self-attention with layer normalization
- **MLPBlock**: Feed-forward network with GELU activation
- **TransformerEncoderBlock**: Combines MSA and MLP blocks with residual connections
- **VIT**: Full model with class token, positional embeddings, and classification head

## Project Structure

```
models/
├── vit_module.py   # Core ViT components
└── model.py        # Main VIT class
utilities/          # Training and data utilities
```

## Usage

```python
from models.model import VIT

model = VIT(
    img_size=224,
    patch_size=16,
    num_transformer_layers=12,
    num_heads=12,
    embedding_dim=768
)
```