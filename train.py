import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from modulars.data_setup import create_dataloaders
from models.model import PatchEmbedding , MultiHeadSelfAttentionBlock , TransformerEncoderBlock
import torch
from torchinfo import summary

image_path = Path("/Users/avinash/Desktop/VisionTransformer/data/pizza_steak_sushi")

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"

# Create image size (from Table 3 in the ViT paper)
IMG_SIZE = 224
PATCH_SIZE = 16

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
print(f"Manually created transforms: {manual_transforms}")

# Set the batch size
BATCH_SIZE = 32 # this is lower than the ViT paper but it's because we're starting small

# Create data loaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

#print(train_dataloader, test_dataloader, class_names)

if __name__ == "__main__":
    image , label = next(iter(test_dataloader))
    patchify = PatchEmbedding(in_channels=3,patch_size=16,embedding_dim=768)
    print(f"Input image shape: {image.unsqueeze(0).shape}")
    patch_embedded_image = patchify(image) # add an extra batch dimension on the 0th index, otherwise will error
    print(f"Output patch embedding shape: {patch_embedded_image.shape}")



    # 6. Create class token embedding

    batch_size = patch_embedded_image.shape[0]
    embed_dimension = patch_embedded_image.shape[-1] # Hidden dimension
    class_token = nn.Parameter(torch.ones(batch_size,1,embed_dimension),requires_grad=True)

    # 7. Prepend class token embedding to patch embedding
    patch_embedded_image_with_classtoken = torch.cat((class_token,patch_embedded_image),dim=1)
    print(f"Sequence of patch embeddings with class token prepended shape: {patch_embedded_image_with_classtoken.shape} -> [batch_size, number_of_patches, embedding_dimension]")

    # 8. Create position encoding

    num_patches = int((IMG_SIZE*IMG_SIZE)/ PATCH_SIZE **2)
    position_embeddings = nn.Parameter(torch.rand(1,num_patches + 1,embed_dimension),requires_grad=True)

    # 9. Add position embeddings
    patch_and_position_embedding = patch_embedded_image_with_classtoken + position_embeddings

    # print(patch_and_position_embedding)
    # multi_head_self_attn = MultiHeadSelfAttentionBlock(embed_dimension=768,num_heads=12)

    # print(multi_head_self_attn)
    # patch_through_msa_block = multi_head_self_attn(patch_and_position_embedding)

    # print(f"Input shape of MSA block: {patch_and_position_embedding.shape}")

    # print(f"Output shape MSA block: {patch_through_msa_block.shape}")


    transformer_encoder_block = TransformerEncoderBlock(embed_dimension=768,
                                                        num_heads=12,
                                                        mlp_size=3072,
                                                        mlp_dropout=0.1,
                                                        attn_dropout=0.1)

    # # Print an input and output summary of our Transformer Encoder (uncomment for full output)
    summary(model=transformer_encoder_block,
        input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])





