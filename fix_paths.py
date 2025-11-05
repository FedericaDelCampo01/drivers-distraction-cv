import pandas as pd
import os

for split in ["train", "val", "test"]:
    path = f"data/interim/{split}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "filepath" in df.columns:
            df["filepath"] = df["filepath"].str.replace("\\", "/", regex=False)
            df.to_csv(path, index=False)
            print(f"✅ Fixed paths in {path}")
        else:
            print(f"⚠️ No 'filepath' column found in {path}")
    else:
        print(f"❌ File not found: {path}")

