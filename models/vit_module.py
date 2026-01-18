import torch.nn as nn
import torch


# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,embed_dimension = 768,num_heads = 12 , attn_dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dimension)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dimension,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
        
    
    def forward(self,x):
        x = self.layer_norm(x)
        attn_output,_= self.multihead_attn(query=x, key=x ,value=x ,need_weights=False)
        return attn_output
    
class MLPBlock(nn.Module):
    def __init__(self, embed_dimension = 768,mlp_size = 3072,dropout = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dimension)
        self.mlp = nn.Sequential(nn.Linear(in_features=embed_dimension,out_features=mlp_size),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(in_features=mlp_size,out_features=embed_dimension),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                )

    def forward(self,x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dimension = 768, 
                 num_heads = 12,
                 mlp_size = 3072,
                 mlp_dropout = 0.1,
                 attn_dropout = 0.1):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embed_dimension=embed_dimension,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embed_dimension=embed_dimension,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
        
    def forward(self,x):

        # Residual Connection from MSA Block input to output
        x = self.msa_block(x) + x 
        # Residual Connection from MLP Block input to output
        x = self.mlp_block(x) + x

        return x