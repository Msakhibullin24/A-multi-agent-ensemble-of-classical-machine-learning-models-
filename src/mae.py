"""Мультиагентный ансамбль классических моделей (Multi-Agent Ensemble, MAE).

Архитектура:
  * агенты-эксперты — каждый видит свою смысловую группу признаков
    (академическая, семейно-социальная, поведенческая, общая) и сам выбирает
    лучшую классическую модель по внутренней кросс-валидации;
  * агент-координатор — объединяет вероятности экспертов с весами, равными
    произведению глобальной компетентности (OOF ROC AUC) и локальной
    компетентности (точность Брайера на k ближайших соседях);
  * агент-калибровщик порога — подбирает порог отнесения к группе риска,
    максимизирующий индекс Юдена (TPR - FPR) на out-of-fold прогнозах;
  * агент-интерпретатор — возвращает вклад каждого эксперта в итоговый риск.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_preprocessor(columns, categorical):
    cat = [c for c in columns if c in categorical]
    num = [c for c in columns if c not in categorical]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def base_models(seed):
    """Классические модели, из которых выбирают агенты и которые служат базовыми."""
    return {
        "LR": LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "NB": GaussianNB(),
        "DT": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                     class_weight="balanced", random_state=seed),
        # вероятности SVM — калибровка Платта
        "SVM": CalibratedClassifierCV(SVC(C=1.0, class_weight="balanced"),
                                      method="sigmoid", ensemble=False),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced_subsample",
                                     random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                       random_state=seed),
        "GB": GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=seed),
    }


AGENT_CANDIDATES = ("LR", "SVM", "RF", "GB", "kNN", "NB")


@dataclass
class ExpertAgent:
    name: str
    features: list
    categorical: set
    seed: int
    candidates: tuple = AGENT_CANDIDATES
    chosen: str = ""
    oof_proba: np.ndarray = field(default=None, repr=False)
    auc: float = 0.0

    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        models = base_models(self.seed)
        best = None
        for name in self.candidates:
            pipe = make_pipeline(make_preprocessor(self.features, self.categorical),
                                 clone(models[name]))
            proba = cross_val_predict(pipe, X[self.features], y, cv=cv,
                                      method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
            if best is None or auc > best[1]:
                best = (name, auc, proba, pipe)
        self.chosen, self.auc, self.oof_proba, pipe = best
        self.model_ = pipe.fit(X[self.features], y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[self.features])[:, 1]


def youden_threshold(risk, y):
    """Агент-калибровщик: порог с максимумом индекса Юдена (TPR - FPR)."""
    grid = np.linspace(0.1, 0.9, 81)
    # максимум сбалансированной точности эквивалентен максимуму индекса Юдена
    scores = [balanced_accuracy_score(y, risk >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


class YoudenThreshold(BaseEstimator, ClassifierMixin):
    """Любая модель + тот же калибровщик порога по внеблоковым прогнозам.

    Контрольный вариант для честного сравнения с MAE по пороговым метрикам.
    """

    def __init__(self, estimator, seed=0):
        self.estimator = estimator
        self.seed = seed

    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(clone(self.estimator), X, y, cv=cv,
                                method="predict_proba")[:, 1]
        self.threshold_ = youden_threshold(oof, np.asarray(y))
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


class MultiAgentEnsemble(BaseEstimator, ClassifierMixin):
    """Координатор мультиагентного ансамбля.

    local_k=0 отключает локальную компетентность, tune_threshold=False
    фиксирует порог 0.5 (используется в абляционном анализе).
    """

    def __init__(self, groups, categorical, local_k=25, tune_threshold=True, seed=0):
        self.groups = groups
        self.categorical = categorical
        self.local_k = local_k
        self.tune_threshold = tune_threshold
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.agents_ = [ExpertAgent(name, cols, self.categorical, self.seed).fit(X, y)
                        for name, cols in self.groups.items()]
        self.global_w_ = np.array([max(a.auc - 0.5, 1e-3) for a in self.agents_])
        self.oof_ = np.column_stack([a.oof_proba for a in self.agents_])
        self.y_ = y
        if self.local_k:
            self.space_ = make_preprocessor(list(X.columns), self.categorical).fit(X)
            Z = self.space_.transform(X)
            self.nn_ = NearestNeighbors(n_neighbors=self.local_k + 1).fit(Z)
            # для обучающих объектов исключаем сам объект из окрестности явно:
            # при дубликатах признаков он не обязательно стоит первым
            idx = self.nn_.kneighbors(Z, return_distance=False)
            idx = np.array([[j for j in row if j != i][:self.local_k]
                            for i, row in enumerate(idx)])
            weights = self._weights_from_neighbours(idx)
        else:
            weights = np.tile(self.global_w_, (len(y), 1))
        oof_risk = (weights * self.oof_).sum(1) / weights.sum(1)
        self.threshold_ = youden_threshold(oof_risk, y) if self.tune_threshold else 0.5
        return self

    def _weights_from_neighbours(self, idx):
        # локальная компетентность = 1 - средняя ошибка Брайера эксперта в окрестности
        err = (self.oof_[idx] - self.y_[idx][..., None]) ** 2
        local = 1.0 - err.mean(axis=1)
        return local * self.global_w_

    def global_oof_risk(self):
        """Внеблоковый риск при одних глобальных весах — для абляции."""
        return (self.global_w_ * self.oof_).sum(1) / self.global_w_.sum()

    def agent_weights(self, X):
        if not self.local_k:
            return np.tile(self.global_w_, (len(X), 1))
        idx = self.nn_.kneighbors(self.space_.transform(X), n_neighbors=self.local_k,
                                  return_distance=False)
        return self._weights_from_neighbours(idx)

    def agent_probas(self, X):
        return np.column_stack([a.predict_proba(X) for a in self.agents_])

    def predict_proba(self, X):
        w = self.agent_weights(X)
        p = (w * self.agent_probas(X)).sum(1) / w.sum(1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)

    def explain(self, X):
        """Агент-интерпретатор: доля каждого эксперта в итоговой оценке риска."""
        w = self.agent_weights(X)
        contrib = w * self.agent_probas(X) / w.sum(1, keepdims=True)
        return {a.name: contrib[:, i] for i, a in enumerate(self.agents_)}
