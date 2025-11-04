import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import yaml
from src.data.dataset import DriverDataset


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def show_gradcam(image_tensor, cam, save_path):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * heatmap / 255.0 + 0.5 * image
    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(overlay, 0, 1))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # === Config ===
    with open("src/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    num_classes = 3 if config["task"] == "levels3" else 10

    # === Modelo ===
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("models/best_resnet18.pth", map_location="cpu"))
    model.eval()

    target_layer = model.layer4[-1]  # última capa conv
    gradcam = GradCAM(model, target_layer)

    # === Dataset ===
    ds = DriverDataset("data/interim/test.csv", split="val")

    print("🚗 Generando mapas Grad-CAM para algunas imágenes del test set...")
    for idx in [10, 100, 500, 1000, 2000]:  # elegimos algunas muestras
        image_tensor, label = ds[idx]
        input_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, 1).item()

        cam = gradcam.generate(input_tensor, pred_class)
        save_path = f"outputs/gradcam_{idx}_pred{pred_class}_true{label}.png"
        show_gradcam(image_tensor, cam, save_path)
        print(f"🖼️  Guardado: {save_path}")


if __name__ == "__main__":
    main()
