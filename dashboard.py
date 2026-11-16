"""Interactive dashboard for the bundled student performance data."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from data import SUBJECTS


ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = [
    "student_id",
    "attendance",
    "study_hours",
    "sleep_hours",
    "prev_exam",
    "parent_education",
    "tutoring",
    *SUBJECTS,
]
SCORE_COLUMNS = [*SUBJECTS, "final_avg"]
SUBJECT_LABELS = {
    "math": "Mathematics",
    "science": "Science",
    "english": "English",
    "social": "Social Studies",
    "bengali": "Bengali",
}


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    dataset_path = ROOT / "dataset.csv"
    risk_path = ROOT / "risk_scores.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Student data file not found: {dataset_path}")
    if not risk_path.is_file():
        raise FileNotFoundError(f"Risk scores file not found: {risk_path}")

    students = pd.read_csv(dataset_path)
    risk_scores = pd.read_csv(risk_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(students.columns))
    if missing:
        raise ValueError(f"dataset.csv is missing required columns: {', '.join(missing)}")
    if not {"student_id", "risk_probability"}.issubset(risk_scores.columns):
        raise ValueError("risk_scores.csv must contain student_id and risk_probability columns.")

    if students["student_id"].isna().any() or risk_scores["student_id"].isna().any():
        raise ValueError("Student IDs cannot be empty.")
    students["student_id"] = students["student_id"].astype(str)
    risk_scores["student_id"] = risk_scores["student_id"].astype(str)
    if students["student_id"].duplicated().any():
        raise ValueError("dataset.csv contains duplicate student IDs.")
    if risk_scores["student_id"].duplicated().any():
        raise ValueError("risk_scores.csv contains duplicate student IDs.")

    if "final_avg" not in students.columns:
        students["final_avg"] = students[SUBJECTS].mean(axis=1)
    for column in SCORE_COLUMNS + [
        "attendance",
        "study_hours",
        "sleep_hours",
        "prev_exam",
        "tutoring",
    ]:
        students[column] = pd.to_numeric(students[column], errors="coerce")
    risk_scores["risk_probability"] = pd.to_numeric(
        risk_scores["risk_probability"], errors="coerce"
    )
    if students[SCORE_COLUMNS].isna().any().any():
        raise ValueError("Student scores contain missing or non-numeric values.")
    if risk_scores["risk_probability"].isna().any() or not risk_scores[
        "risk_probability"
    ].between(0, 1).all():
        raise ValueError("Risk probabilities must be numeric values between 0 and 1.")

    student_ids = set(students["student_id"])
    risk_ids = set(risk_scores["student_id"])
    if student_ids != risk_ids:
        raise ValueError("dataset.csv and risk_scores.csv must contain the same student IDs.")

    return students.merge(
        risk_scores[["student_id", "risk_probability"]],
        on="student_id",
        how="left",
        validate="one_to_one",
    )


def main() -> None:
    st.set_page_config(
        page_title="Student Performance Analysis",
        page_icon="📚",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #10151f; color: #eef1f5; }
        .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
        [data-testid="stMetric"] {
            background: #191f2c; border: 1px solid #303849; border-top: 3px solid #d65c52;
            border-radius: 12px; padding: 14px 16px;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3n + 2)
        [data-testid="stMetric"] { border-top-color: #e8a246; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3n)
        [data-testid="stMetric"] { border-top-color: #49a878; }
        [data-testid="stMetricLabel"] { color: #a7b0bf; }
        [data-testid="stMetricValue"] { color: #f0f3f7 !important; }
        section[data-testid="stSidebar"] { background: #171d29; }
        div[data-testid="stAlert"] { border-left: 4px solid #e8a246; }
        h1, h2, h3 { color: #f0f3f7 !important; }
        [data-testid="stDataFrame"] { border: 1px solid #303849; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_dashboard_data()
    st.title("Student performance analysis")
    st.caption(
        f"Bundled dataset · {len(df):,} students across five subjects · "
        "Use the filters to explore performance and support needs."
    )

    subject_filter, pass_filter, _ = st.columns([1.2, 0.8, 2.4])
    with subject_filter:
        selected_subject = st.selectbox("Subject", ["All subjects", *SUBJECT_LABELS.values()])
    with pass_filter:
        pass_mark = st.selectbox("Pass mark", [33, 40, 50, 65], index=1)

    with st.expander("Additional filters", expanded=False):
        education_filter, tutoring_filter, search_filter = st.columns(3)
        education_options = sorted(df["parent_education"].dropna().astype(str).unique())
        with education_filter:
            education = st.selectbox("Parent education", ["All"] + education_options)
        with tutoring_filter:
            tutoring = st.selectbox("Tutoring", ["All", "Tutored", "Not tutored"])
        with search_filter:
            student_search = st.text_input("Find student ID")

    filtered = df
    if education != "All":
        filtered = filtered[filtered["parent_education"].astype(str) == education]
    if tutoring == "Tutored":
        filtered = filtered[filtered["tutoring"] == 1]
    elif tutoring == "Not tutored":
        filtered = filtered[filtered["tutoring"] == 0]
    if student_search.strip():
        filtered = filtered[
            filtered["student_id"].str.contains(student_search.strip(), case=False, regex=False)
        ]

    if filtered.empty:
        st.info("No students match the selected filters. Adjust the filters above to see results.")
        return

    score_column = (
        "final_avg"
        if selected_subject == "All subjects"
        else next(key for key, label in SUBJECT_LABELS.items() if label == selected_subject)
    )
    score_values = filtered[score_column]
    score_rows = filtered.assign(average_score=score_values)
    score_rows["status"] = np.select(
        [
            score_rows["average_score"] < pass_mark,
            score_rows["attendance"] < 75,
            score_rows["risk_probability"] >= 0.5,
        ],
        ["Below pass mark", "Low attendance", "Model flagged"],
        default="On track",
    )
    flagged_count = int((score_rows["status"] != "On track").sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Mean score", f"{score_values.mean():.1f}")
    metric_columns[1].metric("Median", f"{score_values.median():.1f}")
    standard_deviation = score_values.std()
    metric_columns[2].metric(
        "Std. deviation",
        f"{standard_deviation:.1f}" if pd.notna(standard_deviation) else "—",
    )
    metric_columns[3].metric(
        f"Pass rate (≥{pass_mark})",
        f"{score_values.ge(pass_mark).mean():.0%}",
    )
    flagged_metric = st.columns(4)[0]
    flagged_metric.metric("Students flagged", f"{flagged_count:,}")

    with st.container(border=True):
        st.subheader("Score distribution")
        bins = np.arange(0, 110, 10)
        counts, edges = np.histogram(score_values.clip(0, 100), bins=bins)
        distribution = pd.DataFrame(
            {
                "Score range": [
                    f"{int(edges[i])}–{int(edges[i + 1])}" for i in range(len(counts))
                ],
                "Students": counts,
            }
        )
        st.bar_chart(distribution, x="Score range", y="Students", color="#e8a246")

    correlation = filtered["attendance"].corr(score_values)
    score_label = "Final average" if selected_subject == "All subjects" else selected_subject
    with st.container(border=True):
        st.subheader("Attendance vs. score")
        st.scatter_chart(
            score_rows,
            x="attendance",
            y="average_score",
            color="#49a878",
            x_label="Attendance (%)",
            y_label=f"{score_label} (%)",
        )
        if pd.isna(correlation):
            st.caption("Attendance correlation is unavailable for this filtered group.")
        else:
            strength = (
                "weak"
                if abs(correlation) < 0.3
                else "moderate"
                if abs(correlation) < 0.7
                else "strong"
            )
            st.caption(
                f"Correlation r = {correlation:.2f}: {strength} "
                "relationship. Correlation does not establish causation."
            )

    with st.container(border=True):
        st.subheader("Average score by subject")
        subject_averages = (
            filtered[SUBJECTS]
            .mean()
            .rename(index=SUBJECT_LABELS)
            .rename_axis("Subject")
            .reset_index(name="Average score")
        )
        st.bar_chart(
            subject_averages,
            x="Subject",
            y="Average score",
            color="#49a878",
            horizontal=True,
            sort=False,
            x_label="Subject",
            y_label="Average score",
        )

    with st.container(border=True):
        heading, download = st.columns([3, 1])
        with heading:
            st.subheader("Students")
        ranked_students = score_rows.sort_values(
            "average_score", ascending=False
        ).copy()
        ranked_students["Rank"] = np.arange(1, len(ranked_students) + 1)
        display_table = ranked_students[
            [
                "student_id",
                "average_score",
                "attendance",
                "Rank",
                "status",
                "risk_probability",
            ]
        ].rename(
            columns={
                "student_id": "Student",
                "average_score": "Average",
                "attendance": "Attendance",
                "status": "Status",
                "risk_probability": "Model risk",
            }
        )
        display_table["Attendance"] = display_table["Attendance"].map(
            lambda value: f"{value:.1f}%"
        )
        display_table["Model risk"] = display_table["Model risk"].map(lambda value: f"{value:.0%}")
        with download:
            st.download_button(
                "Download CSV",
                data=display_table.to_csv(index=False).encode("utf-8"),
                file_name="student_performance.csv",
                mime="text/csv",
            )
        st.dataframe(display_table, hide_index=True)
        st.caption(
            "Flagged when the selected score is below the pass mark, attendance is under 75%, "
            "or the supplied model risk is at least 50%. Model scores were trained against "
            "the bundled 45-point pass mark. Use flags to offer support, not to penalize students."
        )


if __name__ == "__main__":
    main()
