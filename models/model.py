import torch.nn as nn
import torch
from .vit_module import PatchEmbedding, MultiHeadSelfAttentionBlock,TransformerEncoderBlock

class VIT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 patch_size:int=16,
                 in_channels:int=3,
                 embedding_dim:int=768,
                 num_transformer_layers:int=12,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0.1,
                 embedding_dropout:float=0.1):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        #Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_token = nn.Parameter(torch.ones(1,1,embedding_dim),requires_grad=True)

        #Create learnable position embedding
        self.position_embeddings = nn.Parameter(torch.rand(1,self.num_patches + 1,embedding_dim),requires_grad=True)
        
        #Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(
                embed_dimension=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout
            ) for _ in range(num_transformer_layers)]
        )

        self.mlp_head = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                  nn.Linear(in_features=embedding_dim,out_features=3))
        
    def forward(self,x):
        
        batch_size = x.shape[0]

        class_token = self.class_token.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        x = self.patch_embedding(x)

        # Prepend class token embedding to patch embedding
        x = torch.cat((class_token,x),dim=1)

        # Add position embeddings
        x = x + self.position_embeddings

        x = self.embedding_dropout(x)

        # Pass through the transformer encoder blocks
        x = self.encoder(x)

        # Take the output of the class token and pass it through the MLP head for classification
        class_token_output = x[:,0,:] # take the first token which is the class token
        output = self.mlp_head(class_token_output)

        return output