from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

from ga_feature_select import GeneticFeatureSelector

print(" جاري تحميل البيانات وتشغيل التجربة...\n")

#  تحميل بيانات iris الجاهزة
iris = load_iris()
X = iris.data
y = iris.target

print(f" تم تحميل مجموعة بيانات Iris ({X.shape[0]} صف × {X.shape[1]} عمود)\n")


#  تقسيم البيانات إلى تدريب واختبار
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f" تدريب: {Xtr.shape}, اختبار: {Xte.shape}\n")


#  إعداد النموذج الأساسي
base_est = make_pipeline(
    StandardScaler(), 
    LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=None)
)
print(" تم تجهيز نموذج الانحدار اللوجستي\n")


#  إعداد الخوارزمية الجينية
selector = GeneticFeatureSelector(
    estimator=base_est,
    pop_size=20,      
    generations=30,      
    crossover_prob=0.8, 
    lambda_penalty=0.1, 
    cv_splits=5,
    early_stopping_rounds=10,
    random_state=42
)
print(" تم تهيئة الخوارزمية الجينية بنجاح\n")


#  تدريب الخوارزمية الجينية
try:
    selector.fit(Xtr, ytr)
    best_fitness = getattr(selector, "best_fitness_", None)
    best_mask = getattr(selector, "best_mask_", None)
    print(" تم تدريب الخوارزمية بنجاح\n")

    if best_fitness is not None:
        print(f" أفضل لياقة (Fitness): {best_fitness:.4f}")
    if best_mask is not None:
        print(f" عدد الميزات المختارة: {np.sum(best_mask)} / {X.shape[1]}")

except Exception as e:
    print(f" خطأ أثناء تدريب الخوارزمية الجينية: {e}")
    exit()


#  تجربة النموذج بعد اختيار الميزات

try:
    if hasattr(selector, "transform"):
        Xtr_sel = selector.transform(Xtr)
        Xte_sel = selector.transform(Xte)
    else:
        print(" لا توجد دالة transform، سيتم استخدام كل الأعمدة.")
        Xtr_sel, Xte_sel = Xtr, Xte

    # حماية إضافية في حال القناع غير متطابق
    if Xtr_sel.shape[1] == 0 or Xte_sel.shape[1] == 0:
        print(" لم يتم اختيار أي ميزات! سيتم استخدام أول 3 أعمدة فقط.")
        Xtr_sel = Xtr[:, :3]
        Xte_sel = Xte[:, :3]

    # تدريب النموذج على الميزات المختارة
    base_est.fit(Xtr_sel, ytr)
    yhat = base_est.predict(Xte_sel)
    acc = accuracy_score(yte, yhat)
    print(f"\n دقة النموذج بعد اختيار الميزات: {acc:.4f}")

except Exception as e:
    print(f" خطأ أثناء اختبار النموذج بعد اختيار الميزات: {e}")

print("\n انتهت التجربة بنجاح.")
print("------------------------------------------------------")
