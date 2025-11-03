import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from src.data.dataset import DriverDataset


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
        title="Confusion Matrix (Test Set)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # === Cargar config ===
    with open("src/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluando modelo en dispositivo: {device}")

    num_classes = 3 if config["task"] == "levels3" else 10
    batch_size = config["batch_size"]

    # === Dataset de test ===
    test_ds = DriverDataset("data/interim/test.csv", split="val")  # usamos 'val' para no aplicar augmentations
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # === Cargar modelo entrenado ===
    model = models.resnet18(weights=None)  # no descargamos pesos
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/best_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()

    print("📦 Modelo cargado correctamente. Iniciando evaluación...")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Métricas ===
    print("\n🔍 Reporte de clasificación (Test Set):")
    print(classification_report(all_labels, all_preds, digits=3))

    plot_confusion_matrix(all_labels, all_preds, labels=range(num_classes),
                          save_path="outputs/confusion_test_resnet18.png")
    print("📊 Matriz de confusión guardada en outputs/confusion_test_resnet18.png")

    # (Opcional) Guardar predicciones
    results = pd.DataFrame({"true": all_labels, "pred": all_preds})
    results.to_csv("outputs/test_predictions_resnet18.csv", index=False)
    print("📁 Predicciones guardadas en outputs/test_predictions_resnet18.csv")


if __name__ == "__main__":
    main()
