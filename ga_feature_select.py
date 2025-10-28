# path: ga_feature_select.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

TaskT = Literal["auto", "clf", "reg"]


@dataclass
class GeneticFeatureSelector:
    estimator: object
    generations: int = 30
    pop_size: int = 20
    crossover_prob: float = 0.8
    mutation_prob: float = 0.05
    lambda_penalty: float = 0.1
    cv_splits: int = 5
    early_stopping_rounds: int = 8
    random_state: Optional[int] = 42
    task: TaskT = "auto"   # "auto" | "clf" | "reg"
    n_jobs: int = 1        # استخدم 1 على ويندوز داخل Streamlit لمنع التعليق

    # ------------- helpers -------------
    def _infer_task(self, y) -> TaskT:
        """يختار نوع المهمة تلقائياً إذا task='auto'."""
        if self.task in ("clf", "reg"):
            return self.task
        y_arr = np.asarray(y)
        # غير رقمي → تصنيف
        if not np.issubdtype(y_arr.dtype, np.number):
            return "clf"
        # رقمي: عدد قيم مميّزة قليل نسبيًا → تصنيف، غير ذلك انحدار
        uniq = np.unique(y_arr)
        return "clf" if len(uniq) <= max(20, int(0.05 * len(y_arr))) else "reg"

    def _build_cv(self, y, task: TaskT):
        """ينشئ KFold مناسب ويضمن عدم كسر StratifiedKFold مع الفئات الصغيرة."""
        if task == "clf":
            _, counts = np.unique(y, return_counts=True)
            n_splits = int(min(self.cv_splits, counts.min() if counts.size else self.cv_splits))
            n_splits = max(n_splits, 2)
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

    def _scoring(self, task: TaskT) -> str:
        return "accuracy" if task == "clf" else "r2"

    def _fitness(self, mask: np.ndarray, X: np.ndarray, y: np.ndarray, task: TaskT) -> float:
        # تجنّب مجموعة فارغة
        if mask.sum() == 0:
            return -1e9
        Xs = X[:, mask]
        cv = self._build_cv(y, task)
        scoring = self._scoring(task)
        # استخدم self.n_jobs (1 داخل الواجهة على ويندوز، أو -1 ببيئات CLI)
        scores = cross_val_score(self.estimator, Xs, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        penalty = self.lambda_penalty * (mask.sum() / X.shape[1])  # عقوبة على كثرة الميزات
        return float(scores.mean() - penalty)

    # ------------- API -------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        task = self._infer_task(y)
        n = X.shape[1]

        # تهيئة السكان عشوائياً
        pop = rng.integers(0, 2, size=(self.pop_size, n), dtype=bool)

        best_mask = None
        best_fit = -1e9
        no_imp = 0

        for _ in range(self.generations):
            fits = np.array([self._fitness(ind, X, y, task) for ind in pop], dtype=float)

            # elitism
            elite_idx = int(np.argmax(fits))
            if fits[elite_idx] > best_fit + 1e-12:
                best_fit = float(fits[elite_idx])
                best_mask = pop[elite_idx].copy()
                no_imp = 0
            else:
                no_imp += 1
                if self.early_stopping_rounds and no_imp >= self.early_stopping_rounds:
                    break

            # roulette-wheel selection
            shifted = fits - fits.min() + 1e-12
            probs = shifted / shifted.sum()
            parents_idx = rng.choice(self.pop_size, size=self.pop_size, p=probs)
            parents = pop[parents_idx]

            # crossover (one-point)
            children = parents.copy()
            for i in range(0, self.pop_size - 1, 2):
                if rng.random() < self.crossover_prob:
                    cut = rng.integers(1, n)
                    a = children[i].copy()
                    b = children[i + 1].copy()
                    children[i, :cut], children[i + 1, :cut] = b[:cut], a[:cut]

            # mutation (bit-flip)
            mut = rng.random(children.shape) < self.mutation_prob
            children ^= mut

            # ثبّت الأفضل (elitism)
            if best_mask is not None:
                children[0] = best_mask

            pop = children

        self.best_mask_ = np.asarray(best_mask if best_mask is not None else np.ones(n, dtype=bool))
        self.best_fitness_ = float(best_fit)
        self.task_ = task
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.best_mask_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
