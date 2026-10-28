"""Рисунки и таблицы для статьи (300 dpi, оттенки серого — для печатных журналов)."""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from experiment import OUT, SEED, load, scenario_groups  # noqa: E402
from mae import MultiAgentEnsemble  # noqa: E402

FIG = OUT.parent / "figures"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
LABELS = {"Voting (soft)": "Голосование", "Stacking": "Стекинг", "MAE (полная)": "MAE",
          "Voting (soft) + порог": "Голосование + порог"}
TASKS = {"mat_early": "Математика, до начала обучения", "mat_mid": "Математика, после 1-го периода",
         "por_early": "Португальский, до начала обучения", "por_mid": "Португальский, после 1-го периода"}


def box(ax, x, y, w, h, text, fill="white", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fill, ec="black", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, lw=0.9, color="black"))


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    box(ax, 3.0, 6.3, 4.0, 0.7, "Данные об обучающемся\n(анкета, посещаемость, оценки)", "#e6e6e6")
    experts = ["Агент\n«Академический»", "Агент\n«Семейно-\nсоциальный»",
               "Агент\n«Поведенческий»", "Агент\n«Общий»"]
    xs = [0.2, 2.65, 5.1, 7.55]
    for x, name in zip(xs, experts):
        arrow(ax, 5.0, 6.3, x + 1.1, 5.55)
        box(ax, x, 4.35, 2.2, 1.2, name, bold=True)
        ax.text(x + 1.1, 4.12, "выбор модели:\nLR, SVM, kNN,\nNB, RF, GB", ha="center",
                va="top", fontsize=7, style="italic")
        arrow(ax, x + 1.1, 3.35, 5.0, 2.75)
    box(ax, 1.6, 1.95, 6.8, 0.8,
        "Агент-координатор\nвес агента = глобальная × локальная компетентность", "#d0d0d0",
        bold=True)
    arrow(ax, 5.0, 1.95, 2.6, 1.3)
    arrow(ax, 5.0, 1.95, 7.4, 1.3)
    box(ax, 0.9, 0.45, 3.4, 0.85, "Агент-калибровщик порога\n(индекс Юдена)", "#eeeeee")
    box(ax, 5.7, 0.45, 3.4, 0.85, "Агент-интерпретатор\n(вклад агентов в риск)", "#eeeeee")
    ax.text(5.0, 0.12, "Решение: «группа риска» + объяснение для преподавателя",
            ha="center", fontsize=8.5, fontweight="bold")
    fig.savefig(FIG / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def comparison():
    panels = {
        "ROC AUC": ["LR", "kNN", "NB", "DT", "SVM", "RF", "AdaBoost", "GB",
                    "Voting (soft)", "Stacking", "Выбор из 6 + порог", "MAE (полная)"],
        # пороговая метрика — только модели с одинаковым калибровщиком порога
        "Balanced Acc": ["LR + порог", "RF + порог", "GB + порог", "Voting (soft) + порог",
                         "Выбор из 6 + порог", "MAE (полная)"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})
    markers = ["o", "s", "^", "D"]
    for ax, (metric, models) in zip(axes, panels.items()):
        for i, (key, title) in enumerate(TASKS.items()):
            df = pd.read_csv(OUT / f"{key}_metrics.csv").set_index("model").loc[models]
            ax.scatter(df[metric], range(len(models)), marker=markers[i], s=26,
                       facecolors="none" if i % 2 else "black", edgecolors="black",
                       label=title, zorder=3)
        ax.set_title("ROC AUC" if metric == "ROC AUC"
                     else "Сбалансированная точность\n(одинаковая калибровка порога)",
                     fontsize=9.5)
        ax.grid(axis="x", ls=":", color="grey")
        ax.xaxis.set_major_formatter(COMMA)
        ax.axhline(len(models) - 1.5, color="grey", lw=0.6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([LABELS.get(m, m) for m in models], fontsize=8.5)
        ax.invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def agent_weights(summary):
    agents = ["Академический", "Семейно-социальный", "Поведенческий", "Общий"]
    hatches = ["", "///", "...", "xxx"]
    shades = ["#404040", "#bfbfbf", "white", "#808080"]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    left = np.zeros(len(TASKS))
    for a, h, c in zip(agents, hatches, shades):
        w = np.array([summary[k]["agents"]["weight_mean"][a] for k in TASKS])
        ax.barh(range(len(TASKS)), w, left=left, color=c, hatch=h, edgecolor="black",
                label=a, height=0.6)
        for i, (lft, v) in enumerate(zip(left, w)):
            if v > 0.07:
                ax.text(lft + v / 2, i, f"{v:.2f}".replace(".", ","), ha="center",
                        va="center", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        left += w
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(TASKS.values(), fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_xlabel("Средний нормированный вес агента в решении координатора")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.28),
              frameon=False)
    fig.savefig(FIG / "fig3_agent_weights.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def case_study():
    """Пример работы агента-интерпретатора на отложенной выборке (математика, сценарий mid)."""
    X, y, categorical = load("mat")
    groups = scenario_groups("mid", X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=SEED)
    mae = MultiAgentEnsemble(groups, categorical, seed=SEED).fit(X_tr, y_tr)
    risk = mae.predict_proba(X_te)[:, 1]
    contrib = pd.DataFrame(mae.explain(X_te), index=X_te.index)
    flagged = np.where(risk >= mae.threshold_)[0]
    # три показательных случая: верно выявленный с наибольшим риском, верно выявленный
    # с наибольшим поведенческим вкладом, ложная тревога с наибольшим риском
    hits = [i for i in flagged if y_te[i] == 1]
    false_alarm = [i for i in flagged if y_te[i] == 0]
    picks = [max(hits, key=lambda i: risk[i])]
    behav = contrib["Поведенческий"].to_numpy()
    rest = [i for i in hits if i not in picks]
    if rest:
        picks.append(max(rest, key=lambda i: behav[i]))
    if false_alarm:
        picks.append(max(false_alarm, key=lambda i: risk[i]))
    rows = []
    for i in picks:
        r = X_te.iloc[i]
        c = contrib.iloc[i]
        rows.append({
            "Обучающийся": f"№ {X_te.index[i] + 1}",
            "Оценка риска": round(float(risk[i]), 2),
            "Факт (G3<10)": "да" if y_te[i] else "нет",
            **{f"Вклад: {k}": round(float(v), 2) for k, v in c.items()},
            "G1": int(r["G1"]), "Неуд. ранее": int(r["failures"]),
            "Время на учёбу": int(r["studytime"]),
            "Планирует высшее": r["higher"],
        })
    out = {"threshold": mae.threshold_, "experts": [(a.name, a.chosen, round(a.auc, 3))
                                                    for a in mae.agents_],
           "cases": rows}
    (OUT / "case_study.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / "summary.json").read_text())
    architecture()
    comparison()
    agent_weights(summary)
    case_study()


if __name__ == "__main__":
    main()
