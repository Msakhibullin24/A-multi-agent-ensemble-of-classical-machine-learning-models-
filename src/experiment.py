"""Сравнение базовых классических моделей, классических ансамблей и MAE.

Повторённая стратифицированная кросс-валидация 5x5 на наборах UCI Student
Performance (математика, португальский язык) в двух сценариях:
  early — до начала обучения (без оценок G1, G2);
  mid   — после первого аттестационного периода (с оценкой G1).
Целевая переменная: группа академического риска, G3 < 10 (по 20-балльной шкале).
Признак absences исключён: в UCI это пропуски за весь учебный год, т. е. информация
из будущего для обоих сценариев.
"""

import copy
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, fbeta_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).parent))
from mae import (MultiAgentEnsemble, YoudenThreshold, base_models,  # noqa: E402
                 make_preprocessor, youden_threshold)

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"
SEED, N_SPLITS, N_REPEATS = 42, 5, 5
N_JOBS = -1

GROUPS = {
    "Академический": ["school", "reason", "studytime", "failures",
                      "schoolsup", "paid", "higher"],
    "Семейно-социальный": ["address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob",
                           "Fjob", "guardian", "traveltime", "famsup", "nursery",
                           "internet", "famrel"],
    "Поведенческий": ["sex", "age", "activities", "romantic", "freetime", "goout",
                      "Dalc", "Walc", "health"],
}


def load(subject):
    df = pd.read_csv(DATA / f"student-{subject}.csv", sep=";")
    y = (df["G3"] < 10).astype(int).to_numpy()
    X = df.drop(columns=["G2", "G3", "absences"])
    categorical = set(X.select_dtypes(include="object").columns)
    return X, y, categorical


def scenario_groups(scenario, all_cols):
    groups = {k: list(v) for k, v in GROUPS.items()}
    if scenario == "mid":
        groups["Академический"].append("G1")
    groups["Общий"] = list(all_cols)
    return groups


def build_models(X, categorical, groups):
    pre = make_preprocessor(list(X.columns), categorical)
    base = base_models(SEED)
    models = {n: make_pipeline(clone(pre), m) for n, m in base.items()}
    estimators = list(base.items())
    models["Voting (soft)"] = make_pipeline(clone(pre), VotingClassifier(estimators, voting="soft"))
    models["Stacking"] = make_pipeline(clone(pre), StackingClassifier(
        estimators, final_estimator=LogisticRegression(max_iter=2000), cv=5,
        stack_method="predict_proba"))
    # контроль: те же модели + тот же калибровщик порога, что у MAE
    for name in ("LR", "RF", "GB", "Voting (soft)"):
        models[f"{name} + порог"] = YoudenThreshold(clone(models[name]), seed=SEED)
    # контроль: один агент с выбором из 6 моделей на всех признаках + порог
    models["Выбор из 6 + порог"] = MultiAgentEnsemble(
        groups={"Общий": list(X.columns)}, categorical=categorical, local_k=0,
        tune_threshold=True, seed=SEED)
    models["MAE (полная)"] = MultiAgentEnsemble(groups=groups, categorical=categorical,
                                                local_k=25, tune_threshold=True, seed=SEED)
    return models


def ablations(mae):
    """Абляционные варианты MAE на тех же обученных агентах."""
    glob, glob_t, local = copy.copy(mae), copy.copy(mae), copy.copy(mae)
    glob.local_k, glob.threshold_ = 0, 0.5
    glob_t.local_k = 0
    glob_t.threshold_ = youden_threshold(mae.global_oof_risk(), mae.y_)
    local.threshold_ = 0.5
    return {"MAE: глоб. веса": glob, "MAE: глоб. веса + порог": glob_t,
            "MAE: глоб.+лок. веса": local}


def scores(y, proba, pred):
    return {
        "ROC AUC": roc_auc_score(y, proba),
        "PR AUC": average_precision_score(y, proba),
        "Balanced Acc": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "F2": fbeta_score(y, pred, beta=2),
    }


TESTED = ("ROC AUC", "Balanced Acc", "Precision", "F1", "F2")


def holm(p):
    """Поправка Холма — Бонферрони."""
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def corrected_ttest(a, b, n_train, n_test):
    """Скорректированный t-тест для повторённой CV (Nadeau, Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def run_fold(X, y, categorical, groups, tr, te):
    warnings.filterwarnings("ignore")
    fold_scores, agents = {}, {}
    for name, model in build_models(X, categorical, groups).items():
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        fold_scores[name] = scores(y[te], proba, model.predict(X.iloc[te]))
        if name == "MAE (полная)":
            for abl_name, abl in ablations(model).items():
                fold_scores[abl_name] = scores(y[te], abl.predict_proba(X.iloc[te])[:, 1],
                                               abl.predict(X.iloc[te]))
            w = model.agent_weights(X.iloc[te])
            w = w / w.sum(1, keepdims=True)
            agents["experts"] = [(a.name, a.chosen, a.auc, float(w[:, i].mean()))
                                 for i, a in enumerate(model.agents_)]
            agents["threshold"] = model.threshold_
    return fold_scores, agents


def run(subject, scenario):
    X, y, categorical = load(subject)
    if scenario == "early":
        X = X.drop(columns=["G1"])
    groups = scenario_groups(scenario, X.columns)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = Parallel(n_jobs=N_JOBS)(delayed(run_fold)(X, y, categorical, groups, tr, te)
                                for tr, te in cv.split(X, y))
    per_fold = defaultdict(list)
    agent_log = {"chosen": defaultdict(Counter), "auc": defaultdict(list),
                 "weight": defaultdict(list), "threshold": []}
    for fold_scores, agents in folds:
        for name, sc in fold_scores.items():
            per_fold[name].append(sc)
        for name, chosen, auc, weight in agents["experts"]:
            agent_log["chosen"][name][chosen] += 1
            agent_log["auc"][name].append(auc)
            agent_log["weight"][name].append(weight)
        agent_log["threshold"].append(agents["threshold"])

    n_test = len(y) // N_SPLITS
    n_train = len(y) - n_test
    rows, tests = [], []
    for name, folds in per_fold.items():
        df = pd.DataFrame(folds)
        row = {"model": name}
        for m in df.columns:
            row[m] = df[m].mean()
            row[m + " sd"] = df[m].std(ddof=1)
        rows.append(row)
    ref = pd.DataFrame(per_fold["MAE (полная)"])
    for name, folds in per_fold.items():
        if name == "MAE (полная)":
            continue
        df = pd.DataFrame(folds)
        for m in TESTED:
            t, p = corrected_ttest(ref[m], df[m], n_train, n_test)
            tests.append({"model": name, "metric": m, "diff": ref[m].mean() - df[m].mean(),
                          "t": t, "p": p})
    tests = pd.DataFrame(tests)
    # поправка Холма на множественные сравнения внутри задачи и метрики
    # (семейство — все модели сравнения, кроме абляционных вариантов MAE)
    tests["p_holm"] = np.nan
    family = ~tests.model.str.startswith("MAE")
    for m in TESTED:
        sel = family & (tests.metric == m)
        tests.loc[sel, "p_holm"] = holm(tests.loc[sel, "p"].to_numpy())
    agents = {
        "chosen": {k: dict(v) for k, v in agent_log["chosen"].items()},
        "auc_mean": {k: float(np.mean(v)) for k, v in agent_log["auc"].items()},
        "weight_mean": {k: float(np.mean(v)) for k, v in agent_log["weight"].items()},
        "threshold_mean": float(np.mean(agent_log["threshold"])),
        "threshold_sd": float(np.std(agent_log["threshold"], ddof=1)),
    }
    info = {"n": int(len(y)), "at_risk": int(y.sum()), "share": float(y.mean()),
            "n_features": int(X.shape[1])}
    return pd.DataFrame(rows), tests, agents, info


def main():
    OUT.mkdir(exist_ok=True)
    summary = {}
    for subject in ("mat", "por"):
        for scenario in ("early", "mid"):
            key = f"{subject}_{scenario}"
            part = OUT / f"{key}_summary.json"
            if part.exists():  # задача уже посчитана — эксперимент возобновляемый
                summary[key] = json.loads(part.read_text())
                continue
            t0 = time.time()
            res, tests, agents, info = run(subject, scenario)
            res.to_csv(OUT / f"{key}_metrics.csv", index=False)
            tests.to_csv(OUT / f"{key}_tests.csv", index=False)
            summary[key] = {"data": info, "agents": agents, "seconds": time.time() - t0}
            part.write_text(json.dumps(summary[key], ensure_ascii=False, indent=2))
            print(f"{key}: {time.time() - t0:.0f} s", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
