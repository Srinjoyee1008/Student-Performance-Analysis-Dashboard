"""Descriptive statistics, charts and the at-risk prediction model."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import SUBJECTS

NUMERIC = ["attendance", "study_hours", "sleep_hours", "prev_exam", "internet_access", "tutoring"]
CATEGORICAL = ["parent_education"]


def describe(df: pd.DataFrame, pass_mark: float) -> dict:
    return {
        "students": len(df),
        "mean": df.final_avg.mean(),
        "median": df.final_avg.median(),
        "std": df.final_avg.std(),
        "pass_rate": (df.final_avg >= pass_mark).mean(),
        "subject_means": df[SUBJECTS].mean().round(1).to_dict(),
        "correlations": df[NUMERIC + ["final_avg"]].corr()["final_avg"].drop("final_avg").sort_values(ascending=False).round(2).to_dict(),
        "tutoring_gap": df.groupby("tutoring").final_avg.mean().round(1).to_dict(),
        "parent_edu": df.groupby("parent_education").final_avg.mean().round(1).to_dict(),
    }


def make_charts(df: pd.DataFrame, pass_mark: float, out: Path) -> list:
    out.mkdir(parents=True, exist_ok=True)
    files = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df.final_avg, bins=15, color="#4c8fb5", edgecolor="white")
    ax.axvline(pass_mark, color="#c2410c", ls="--", label=f"Pass mark {pass_mark:g}")
    ax.set(title="Distribution of final average", xlabel="Final average", ylabel="Students")
    ax.legend(); f = out / "distribution.png"; fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig); files.append(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df.attendance, df.final_avg, s=14, alpha=.6, color="#0f7b6c")
    m, b = np.polyfit(df.attendance, df.final_avg, 1)
    xs = np.linspace(df.attendance.min(), df.attendance.max(), 50)
    axes[0].plot(xs, m * xs + b, color="#c2410c")
    axes[0].set(title="Attendance vs. score", xlabel="Attendance %", ylabel="Final average")
    axes[1].scatter(df.study_hours, df.final_avg, s=14, alpha=.6, color="#4c8fb5")
    m, b = np.polyfit(df.study_hours, df.final_avg, 1)
    xs = np.linspace(0, df.study_hours.max(), 50)
    axes[1].plot(xs, m * xs + b, color="#c2410c")
    axes[1].set(title="Study hours vs. score", xlabel="Hours per day")
    f = out / "drivers.png"; fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig); files.append(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    df[SUBJECTS].boxplot(ax=ax)
    ax.set(title="Score spread by subject", ylabel="Marks")
    f = out / "subjects.png"; fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig); files.append(f)
    return files


def _pipeline(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("model", model)])


def train_risk_model(df: pd.DataFrame, pass_mark: float, out: Path):
    """Predict 'at risk' (final average below the pass mark) from early indicators.

    Subject marks are deliberately NOT used as inputs: the goal is to flag students
    before exams, using attendance, habits, background and the previous exam.
    """
    X = df[NUMERIC + CATEGORICAL]
    y = (df.final_avg < pass_mark).astype(int)
    if y.sum() < 10:
        raise ValueError("Too few at-risk students for modelling; raise the pass mark.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    results, fitted = {}, {}
    for name, mdl in {"logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
                      "random_forest": RandomForestClassifier(300, class_weight="balanced", random_state=0)}.items():
        pipe = _pipeline(mdl).fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        results[name] = {"auc": roc_auc_score(yte, prob),
                         "cv_auc": cross_val_score(_pipeline(mdl), X, y, cv=5, scoring="roc_auc").mean()}
        fitted[name] = pipe

    best = max(results, key=lambda k: results[k]["cv_auc"])
    pipe = fitted[best]
    pred = pipe.predict(Xte)
    report = classification_report(yte, pred, target_names=["on track", "at risk"])
    cm = confusion_matrix(yte, pred)

    # feature importance
    names = pipe.named_steps["pre"].get_feature_names_out()
    mdl = pipe.named_steps["model"]
    imp = np.abs(mdl.coef_[0]) if best == "logistic" else mdl.feature_importances_
    imp = pd.Series(imp, index=[n.split("__")[1] for n in names]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color="#0f7b6c"); ax.set_title(f"What drives risk ({best})")
    f = out / "feature_importance.png"; fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig)

    scored = df[["student_id"]].copy()
    scored["risk_probability"] = pipe.predict_proba(X)[:, 1].round(3)
    scored["final_avg"] = df.final_avg
    scored = scored.sort_values("risk_probability", ascending=False)
    return {"results": results, "best": best, "report": report, "cm": cm,
            "importance": imp.sort_values(ascending=False).round(3).to_dict(),
            "chart": f, "scored": scored, "y": y}
