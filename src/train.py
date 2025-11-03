import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import DriverDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # === Configuración ===
    with open("src/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    batch_size = config["batch_size"]
    num_epochs = config["train"]["epochs"]
    num_classes = 3 if config["task"] == "levels3" else 10

    # === Datasets y DataLoaders ===
    train_ds = DriverDataset("data/interim/train.csv", split="train")
    val_ds = DriverDataset("data/interim/val.csv", split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # === Modelo: ResNet18 baseline ===
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])

    best_acc = 0.0
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\n🌀 Epoch [{epoch+1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_resnet18.pth"))

    print(f"\n✅ Entrenamiento finalizado. Mejor accuracy en validación: {best_acc:.4f}")

    # === Evaluación final ===
    print("\n🔍 Reporte de clasificación:")
    print(classification_report(labels, preds, digits=3))

    plot_confusion_matrix(labels, preds, labels=range(num_classes), save_path="outputs/confusion_resnet18.png")
    print("📊 Matriz de confusión guardada en outputs/confusion_resnet18.png")


if __name__ == "__main__":
    main()
