"""Data loading / synthetic data generation for the student performance project."""
import numpy as np
import pandas as pd

SUBJECTS = ["math", "science", "english", "social", "bengali"]


def make_dataset(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic class dataset.

    Replace this with real (anonymised) school data via load_csv().
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "gender": rng.choice(["F", "M"], n),
        "attendance": np.clip(rng.normal(85, 10, n), 45, 100).round(1),
        "study_hours": np.clip(rng.normal(2.5, 1.1, n), 0, 7).round(1),
        "sleep_hours": np.clip(rng.normal(7, 1, n), 4, 10).round(1),
        "prev_exam": np.clip(rng.normal(60, 15, n), 15, 100).round(1),
        "parent_education": rng.choice(["primary", "secondary", "graduate"], n, p=[.3, .45, .25]),
        "internet_access": rng.choice([0, 1], n, p=[.3, .7]),
        "tutoring": rng.choice([0, 1], n, p=[.65, .35]),
    })
    pe = df["parent_education"].map({"primary": 0, "secondary": 3, "graduate": 6})
    ability = (0.45 * df["prev_exam"] + 0.25 * df["attendance"] + 4.0 * df["study_hours"]
               + 1.2 * df["sleep_hours"] + pe + 3 * df["tutoring"] + 2 * df["internet_access"])
    base = (ability - ability.mean()) / ability.std() * 12 + 58
    for i, s in enumerate(SUBJECTS):
        shift = [-4, 0, 3, 1, 2][i]
        df[s] = np.clip(base + shift + rng.normal(0, 6, n), 5, 100).round(0)
    df["final_avg"] = df[SUBJECTS].mean(axis=1).round(1)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load your own data. Required columns: see REQUIRED."""
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if "final_avg" not in df.columns:
        df["final_avg"] = df[SUBJECTS].mean(axis=1).round(1)
    return df


REQUIRED = ["student_id", "attendance", "study_hours", "sleep_hours", "prev_exam",
            "parent_education", "internet_access", "tutoring"] + SUBJECTS
