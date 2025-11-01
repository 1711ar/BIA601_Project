import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from ga_feature_select import GeneticFeatureSelector
import warnings

warnings.filterwarnings("ignore")

# تحميل البيانات 
iris = load_iris()
X = iris.data
y = iris.target

# تقسيم البيانات إلى تدريب واختبار
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#  النموذج الأساسي 
base_est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))


#  الخوارزمية الجينية
print(" Running Genetic Algorithm Feature Selection...")
ga_selector = GeneticFeatureSelector(
    estimator=base_est,
    generations=30,
    pop_size=20,
    lambda_penalty=0.15,
    cv_splits=5,
    random_state=42,
    early_stopping_rounds=10,
)
ga_selector.fit(Xtr, ytr)

# تطبيق القناع الناتج من الخوارزمية الجينية
Xtr_ga = ga_selector.transform(Xtr)
Xte_ga = ga_selector.transform(Xte)

# تدريب النموذج على الميزات المختارة
base_est.fit(Xtr_ga, ytr)
yhat_ga = base_est.predict(Xte_ga)
acc_ga = accuracy_score(yte, yhat_ga)
selected_features = ga_selector.best_mask_.sum()

# تحليل المكونات الرئيسية 
print(" Running PCA Feature Extraction...")
n_pca = min(2, X.shape[1])
pca = PCA(n_components=n_pca, random_state=42)
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

base_est.fit(Xtr_pca, ytr)
yhat_pca = base_est.predict(Xte_pca)
acc_pca = accuracy_score(yte, yhat_pca)


# الطريقة 3: اختبار F-Statistic (SelectKBest)
print(" Running SelectKBest (F-Test)...")
k_val = min(2, X.shape[1])
skb = SelectKBest(score_func=f_classif, k=k_val)
Xtr_skb = skb.fit_transform(Xtr, ytr)
Xte_skb = skb.transform(Xte)

base_est.fit(Xtr_skb, ytr)
yhat_skb = base_est.predict(Xte_skb)
acc_skb = accuracy_score(yte, yhat_skb)


# مقارنة النتائج النهائية
print("\n===============================================")
print(" Comparison of Feature Selection Methods")
print("===============================================")
print(f"Genetic Algorithm: {acc_ga:.4f} (Selected {selected_features}/{X.shape[1]} features)")
print(f"PCA:               {acc_pca:.4f} (Top {n_pca} components)")
print(f"SelectKBest:       {acc_skb:.4f} (Top {k_val} features)")
print("===============================================")


#  رسم النتائج بيانياً (Bar Chart)
methods = ["Genetic Algorithm", "PCA", "SelectKBest"]
scores = [acc_ga, acc_pca, acc_skb]

# ترتيب النتائج من الأعلى إلى الأدنى
sorted_indices = np.argsort(scores)[::-1]
methods = [methods[i] for i in sorted_indices]
scores = [scores[i] for i in sorted_indices]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, scores, color=["#3498db", "#2ecc71", "#e67e22"], alpha=0.8)
plt.title("Comparison of Feature Selection Methods", fontsize=14, pad=15)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# عرض القيم الرقمية فوق الأعمدة
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

# إبراز العمود الأعلى
max_index = np.argmax(scores)
bars[max_index].set_color("#e74c3c")

plt.tight_layout()
plt.show()

#  تحديد الطريقة الأفضل نصياً
best_method = methods[int(np.argmax(scores))]
print(f"\n Best Performing Method: {best_method} with accuracy {max(scores):.4f}")
