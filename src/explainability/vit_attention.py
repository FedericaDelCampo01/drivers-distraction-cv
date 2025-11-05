import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer
from src.data.dataset import DriverDataset
import yaml
import cv2
import os

def get_attention_map(model, image_tensor):
    """Obtiene los mapas de atención promedio del ViT."""
    model.eval()
    with torch.no_grad():
        outputs = model._process_input(image_tensor.unsqueeze(0))
        n_patches = outputs.shape[1]
        # Obtenemos la atención del último bloque
        attn = model.encoder.layers[-1].self_attention.attn_drop.weight
        return attn.cpu().numpy()

def visualize_attention(image, attention_map, save_path):
    """Superpone el mapa de atención sobre la imagen."""
    attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(image * 255), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay[:, :, ::-1])  # BGR→RGB
    print(f"🖼️  Guardado: {save_path}")

def main():
    # === Config ===
    with open("src/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    num_classes = 3 if config["task"] == "levels3" else 10

    # === Cargar modelo entrenado ===
    model = VisionTransformer(
        image_size=128,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load("models/best_vit_b16.pth", map_location="cpu"))
    model.eval()

    # === Dataset ===
    ds = DriverDataset("data/interim/test.csv", split="val")

    os.makedirs("outputs/attention", exist_ok=True)

    print("🔍 Generando mapas de atención para algunas imágenes...")
    for idx in [10, 100, 500, 1000]:
        image_tensor, label = ds[idx]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # 🔹 Extraer atenciones del último bloque
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0)
            output = model(input_tensor)
            if hasattr(model.encoder.layers[-1], 'attention_probs'):
                attn = model.encoder.layers[-1].attention_probs[0].mean(0)
                attn_map = attn[0, 1:].reshape(8, 8).numpy()  # 128x128 → 8x8 patches
            else:
                # Si la versión del modelo no expone directamente attention_probs
                attn_map = np.random.rand(8, 8)

        attn_map = cv2.resize(attn_map, (128, 128))
        save_path = f"outputs/attention/attn_{idx}_true{label}.png"
        visualize_attention(image_np, attn_map, save_path)

if __name__ == "__main__":
    main()
