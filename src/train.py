import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import yaml
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from dataset import PCamZarrDataset

def train():
    # Load configuration
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)["train"]
    
    device = torch.device("cpu")

    # Initialize Dataset and DataLoader
    train_ds = PCamZarrDataset(
        zarr_path="data/processed/train_x_norm.zarr",
        labels_path="data/raw/Labels/Labels/camelyonpatch_level_2_split_train_y.h5"
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Setup ResNet18 architecture for binary classification
    model = models.resnet18(weights=None)
    
    model.avgpool = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=config["dropout_rate"])
    )
    
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))

    # MLFlow Tracking
    mlflow.set_experiment("Histo-Flow-ResNet18")
    
    with mlflow.start_run():
        mlflow.pytorch.autolog()
        mlflow.log_params(config)

        # Training loop
        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
            
            for images, labels in loop:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        # Export model
        torch.save(model.state_dict(), "models/model.pth")
        print("Training completed. Model saved.")

if __name__ == "__main__":
    train()