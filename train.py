import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from utilities.data_setup import create_dataloaders
from utilities import engine
from models.model import VIT
import torch
from torchinfo import summary
from helper_functions import plot_loss_curves

image_path = Path("data/pizza_steak_sushi")

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
# print(f"Manually created transforms: {manual_transforms}")

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

    model = VIT(img_size=IMG_SIZE,
                in_channels=3,
                patch_size=PATCH_SIZE,
                embedding_dim=768,
                num_transformer_layers=12,
                num_heads=12,
                mlp_size=3072,
                mlp_dropout=0.1,
                attn_dropout=0.1,
                embedding_dropout=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    print(f"Model summary: {summary(model, input_size=image.shape)}")

    random_image_tensor = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    output = model(random_image_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=1e-3,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,)
    
    loss_fn = nn.CrossEntropyLoss()

    results = engine.train(model=model,
                           train_dataloader=train_dataloader,    
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=10,
                           device=device)

    plot_loss_curves(results)