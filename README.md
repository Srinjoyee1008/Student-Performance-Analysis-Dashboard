# Student Performance Analysis & Early-Warning System

Analyses student marks and habits, finds what drives results, predicts which students are
at risk of failing *before* exams, and estimates the economic return of acting early.

## Run
    pip install -r requirements.txt
    python main.py                       # synthetic demo data
    python main.py --csv my_data.csv --pass-mark 40
    python -m streamlit run dashboard.py # interactive dashboard using the bundled CSV files

## Files
- data.py      – synthetic data generator + CSV loader (required columns in `REQUIRED`)
- analysis.py  – statistics, charts, at-risk model (logistic regression vs random forest)
- economics.py – cost-benefit estimator (edit the `Assumptions` numbers)
- main.py      – runs everything, writes `outputs/` (report.md, charts, risk_scores.csv)
- dashboard.py – reference-inspired dark dashboard for `dataset.csv` and `risk_scores.csv`

## Your CSV needs
student_id, attendance, study_hours, sleep_hours, prev_exam, parent_education
(primary/secondary/graduate), internet_access (0/1), tutoring (0/1), math, science, english, social, bengali

## Important notes
- Demo data is synthetic, and the demo model scores are inflated because the data was built
  from the same features. Expect lower, more honest accuracy on real data.
- Use anonymised data and get school/parent consent. Use risk scores to offer support,
  never to label or exclude students.
- Economic figures are assumptions, not measurements. Replace them with local numbers.
