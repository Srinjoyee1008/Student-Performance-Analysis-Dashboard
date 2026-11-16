"""Run the full pipeline:  python main.py [--csv your_data.csv] [--pass-mark 45]"""
import argparse
from pathlib import Path
from data import make_dataset, load_csv
from analysis import describe, make_charts, train_risk_model
from economics import estimate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to your own data (see README for columns)")
    ap.add_argument("--pass-mark", type=float, default=45.0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    df = load_csv(args.csv) if args.csv else make_dataset()
    df.to_csv(out / "dataset.csv", index=False)

    s = describe(df, args.pass_mark)
    charts = make_charts(df, args.pass_mark, out)
    m = train_risk_model(df, args.pass_mark, out)
    m["scored"].to_csv(out / "risk_scores.csv", index=False)
    eco = estimate(len(df), int(m["y"].sum()))

    lines = [
        "# Student performance report", "",
        f"Students: {s['students']}  |  Mean: {s['mean']:.1f}  |  Median: {s['median']:.1f}  |  Std: {s['std']:.1f}",
        f"Pass rate (average >= {args.pass_mark:g}): {s['pass_rate']:.0%}", "",
        "## Subject averages", *[f"- {k}: {v}" for k, v in s["subject_means"].items()], "",
        "## Correlation with final average", *[f"- {k}: {v}" for k, v in s["correlations"].items()], "",
        f"Tutoring gap (0 = no, 1 = yes): {s['tutoring_gap']}",
        f"By parent education: {s['parent_edu']}", "",
        f"## At-risk model (best: {m['best']})",
        *[f"- {k}: test AUC {v['auc']:.2f}, 5-fold CV AUC {v['cv_auc']:.2f}" for k, v in m["results"].items()], "",
        "```", m["report"], "```", "",
        "Top drivers: " + ", ".join(f"{k} ({v})" for k, v in list(m["importance"].items())[:5]), "",
        "## Economic estimate (assumption-based, INR)",
        *[f"- {k.replace('_', ' ')}: {v:,}" for k, v in eco.items()], "",
        "Charts: " + ", ".join(c.name for c in charts) + ", feature_importance.png",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
