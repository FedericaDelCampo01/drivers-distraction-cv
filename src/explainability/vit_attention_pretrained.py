import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.models import vit_b_16, ViT_B_16_Weights
from src.data.dataset import DriverDataset
from torch.nn import functional as F

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "outputs/vit_attention_pretrained"
os.makedirs(output_dir, exist_ok=True)

print(f"🚗 Generando mapas de atención (ViT preentrenado) en {device}")

# === Modelo ===
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
num_classes = 3  # o 10 si usaste las 10 clases
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
model.load_state_dict(torch.load("models/best_vit_b16_pretrained.pth", map_location=device))
model = model.to(device)
model.eval()

# === Dataset de test ===
ds = DriverDataset("data/interim/test.csv", split="val")

# === Helper para obtener atención ===

def get_attention_map(model, image_tensor):
    """Extrae mapas de atención del ViT (torchvision 0.18 / torch 2.8)."""
    model.eval()
    attn_maps = []

    with torch.no_grad():
        # Preprocesar imagen
        x = model._process_input(image_tensor.unsqueeze(0))
        n, _, _ = x.shape
        cls_token = model.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + model.encoder.pos_embedding
        x = model.encoder.dropout(x)

        # Iterar sobre los bloques del encoder
        for blk in model.encoder.layers:
            # Obtener pesos de atención explícitos
            attn_out, attn_weights = blk.self_attention(
                blk.ln_1(x), blk.ln_1(x), blk.ln_1(x),
                need_weights=True, average_attn_weights=False
            )
            attn_maps.append(attn_weights.detach().cpu())

            # Forward residual + MLP
            x = x + blk.dropout(attn_out)
            x = x + blk.mlp(blk.ln_2(x))

        # Promediar las atenciones (capas y cabezas)
        attn = torch.stack(attn_maps).mean(0).mean(1)[0]  # [tokens, tokens]
        attn = attn[0, 1:]  # quitar CLS

        side = int(np.sqrt(attn.shape[0]))
        attn_map = attn.reshape(side, side).numpy()
        attn_map = cv2.resize(attn_map, (224, 224))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map


# === Generar y guardar ejemplos ===
for i in [0, 50, 100, 200, 500]:
    image, label = ds[i]
    image_np = image.permute(1, 2, 0).numpy()
    attn_map = get_attention_map(model, image)

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(image_np * 255), 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(output_dir, f"vit_pretrained_attention_{i}_label{label.item()}.png")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"🖼️  Guardado: {out_path}")

print("\n✅ Mapas de atención generados correctamente.")
