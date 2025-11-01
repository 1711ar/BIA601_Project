import os
import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    load_digits,
)
from datetime import datetime

# إنشاء مجلد لحفظ البيانات إن لم يكن موجودًا
os.makedirs("data", exist_ok=True)

def save_sklearn_dataset(loader, filename, target_name="target"):
    """تحميل بيانات من sklearn وحفظها بصيغة CSV داخل مجلد data"""
    try:
        d = loader()
        feature_names = getattr(d, "feature_names", None)
        if feature_names is None or len(feature_names) != d.data.shape[1]:
            feature_names = [f"feature_{i}" for i in range(d.data.shape[1])]

        # إنشاء DataFrame
        X = pd.DataFrame(d.data, columns=feature_names)
        y = pd.Series(d.target, name=target_name)
        df = pd.concat([X, y], axis=1)

        # حفظ الملف  
        out_path = os.path.join("data", filename)
        df.to_csv(out_path, index=False, encoding="utf-8")

        print(f" Saved: {out_path}  | rows={len(df)}  | cols={df.shape[1]}")
        return out_path

    except Exception as e:
        print(f" Error saving dataset '{filename}': {e}")
        return None


#  إنشاء مجموعات بيانات جاهزة 
datasets = [
    (load_iris, "iris.csv"),
    (load_wine, "wine.csv"),
    (load_breast_cancer, "breast_cancer.csv"),
    (load_diabetes, "diabetes.csv"),
    (load_digits, "digits.csv"),
]

print(" Generating example datasets...\n")

for loader, filename in datasets:
    save_sklearn_dataset(loader, filename)

print("\n All datasets generated successfully at:")
print(os.path.abspath("data"))

with open("data/_log.txt", "a", encoding="utf-8") as log:
    log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] datasets regenerated\n")

print("\n🗒️ Log updated successfully.")
