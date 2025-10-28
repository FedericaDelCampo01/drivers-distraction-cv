import os
import sys
import glob
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import yaml

# === Cargar config ===
with open("src/configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATA_DIR   = config["data_dir"]          # e.g., data/raw
SPLITS_DIR = config["splits_dir"]        # e.g., data/interim
os.makedirs(SPLITS_DIR, exist_ok=True)

# === Buscar driver_imgs_list.csv ===
# Intenta en ubicaciones típicas:
CANDIDATES = [
    os.path.join(DATA_DIR, "driver_imgs_list.csv"),
    os.path.join(os.path.dirname(DATA_DIR), "driver_imgs_list.csv"),
    "driver_imgs_list.csv",
]

driver_csv_path = None
for cand in CANDIDATES:
    if os.path.isfile(cand):
        driver_csv_path = cand
        break

if driver_csv_path is None:
    # Busca de forma recursiva por si quedó en otra subcarpeta
    matches = glob.glob(os.path.join(DATA_DIR, "**", "driver_imgs_list.csv"), recursive=True)
    if matches:
        driver_csv_path = matches[0]

if driver_csv_path is None:
    print("❌ No se encontró 'driver_imgs_list.csv'.")
    print("   Copiá ese archivo al proyecto (recomendado en data/raw/) y volvé a correr el script.")
    print("   Ejemplo PowerShell:")
    print('   Copy-Item "C:\\ruta\\driver_imgs_list.csv" -Destination "data/raw"')
    sys.exit(1)

print(f"🔎 Usando driver list: {driver_csv_path}")
drivers_df = pd.read_csv(driver_csv_path)  # columnas esperadas: subject, classname, img

# Normalizar nombres de columnas por si vienen con capitalización distinta
cols = {c.lower(): c for c in drivers_df.columns}
for expected in ["subject", "classname", "img"]:
    if expected not in [c.lower() for c in drivers_df.columns]:
        raise ValueError("El archivo driver_imgs_list.csv debe tener columnas: subject, classname, img")
drivers_df.columns = [c.lower() for c in drivers_df.columns]

# === Escanear imágenes en data/raw/c0..c9 ===
records = []
for class_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    if class_name not in config["class_map_10"]:
        # saltear carpetas que no sean c0..c9
        continue
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        records.append({
            "filepath": os.path.join(class_dir, fname),
            "class": class_name,
            "img": fname
        })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError(f"No se encontraron imágenes en {DATA_DIR}. ¿Copiaste c0..c9 ahí dentro?")

# === Merge con driver list (para tener driver_id real) ===
df = df.merge(
    drivers_df[["subject", "classname", "img"]].rename(columns={"subject": "driver_id", "classname": "class"}),
    on=["class", "img"],
    how="left"
)

if df["driver_id"].isna().any():
    faltan = df[df["driver_id"].isna()]
    # Puede ser que el CSV tenga rutas relativas (train/img) en vez de solo 'img'
    # Intento extraer solo el basename si fuese necesario
    if "img" in drivers_df.columns:
        # Reintento mapeando por basename
        drivers_df["__img_base"] = drivers_df["img"].apply(lambda x: os.path.basename(str(x)))
        df2 = df[df["driver_id"].isna()].drop(columns=["driver_id"]).merge(
            drivers_df[["subject", "classname", "__img_base"]].rename(columns={"subject":"driver_id", "classname":"class"}),
            left_on=["class","img"], right_on=["class","__img_base"], how="left"
        ).drop(columns=["__img_base"])
        df.loc[df["driver_id"].isna(), "driver_id"] = df2["driver_id"]
    if df["driver_id"].isna().any():
        raise RuntimeError("No se pudo asociar driver_id para algunas imágenes. Verificá que driver_imgs_list.csv coincida con tus archivos.")

# === Mapear etiquetas ===
df["label10"] = df["class"].map(config["class_map_10"])

def map_level3(cname: str) -> int:
    for level, classes in config["class_map_3"].items():
        if cname in classes:
            return int(level)
    return None

df["label3"] = df["class"].apply(map_level3)

# === Split por driver_id (sin fuga) ===
val_size  = float(config["split"]["val_size"])
test_size = float(config["split"]["test_size"])
seed      = int(config["seed"])

# Primero: train vs (val+test)
gss = GroupShuffleSplit(
    n_splits=1,
    train_size=1 - (val_size + test_size),
    test_size=(val_size + test_size),
    random_state=seed
)
train_idx, temp_idx = next(gss.split(df, groups=df["driver_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df  = df.iloc[temp_idx].reset_index(drop=True)

# Luego: val vs test dentro del bloque temporal
gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=test_size / (val_size + test_size),
    random_state=seed
)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["driver_id"]))
val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

# === Guardar ===
train_csv = os.path.join(SPLITS_DIR, "train.csv")
val_csv   = os.path.join(SPLITS_DIR, "val.csv")
test_csv  = os.path.join(SPLITS_DIR, "test.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("✅ Splits creados correctamente:")
print(f"  Train: {len(train_df)} imágenes, drivers únicos: {train_df['driver_id'].nunique()}")
print(f"  Val:   {len(val_df)} imágenes, drivers únicos: {val_df['driver_id'].nunique()}")
print(f"  Test:  {len(test_df)} imágenes, drivers únicos: {test_df['driver_id'].nunique()}")

# Resumen por clase (opcional)
summary = df.groupby("class").size().rename("count").reset_index()
print("\n📊 Distribución total por clase:")
print(summary.to_string(index=False))
