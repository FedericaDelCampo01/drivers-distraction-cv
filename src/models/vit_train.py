import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import yaml
from src.data.dataset import DriverDataset
from torchvision.models.vision_transformer import VisionTransformer

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_preds, all_labels

def main():
    with open("src/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Vision Transformer on {device}")

    batch_size = config["batch_size"]
    num_epochs = config["train"]["epochs"]
    num_classes = 3 if config["task"] == "levels3" else 10

    # === Datasets y DataLoaders ===
    train_ds = DriverDataset("data/interim/train.csv", split="train")
    val_ds = DriverDataset("data/interim/val.csv", split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # === Modelo: Vision Transformer liviano (128x128) ===
    model = VisionTransformer(
        image_size=128,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\n🌀 Epoch [{epoch+1}/{num_epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, device)
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_vit_b16.pth")

    print(f"\n✅ Training complete. Best Val Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
