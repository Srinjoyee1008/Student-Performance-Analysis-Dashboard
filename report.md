# Student performance report

Students: 400  |  Mean: 58.4  |  Median: 59.2  |  Std: 12.1
Pass rate (average >= 45): 86%

## Subject averages
- math: 54.2
- science: 58.4
- english: 60.6
- social: 59.0
- bengali: 59.6

## Correlation with final average
- prev_exam: 0.76
- study_hours: 0.46
- attendance: 0.22
- internet_access: 0.12
- sleep_hours: 0.09
- tutoring: 0.08

Tutoring gap (0 = no, 1 = yes): {0: 57.7, 1: 59.6}
By parent education: {'graduate': 63.9, 'primary': 54.2, 'secondary': 58.1}

## At-risk model (best: logistic)
- logistic: test AUC 1.00, 5-fold CV AUC 0.98
- random_forest: test AUC 0.97, 5-fold CV AUC 0.96

```
              precision    recall  f1-score   support

    on track       1.00      0.94      0.97        86
     at risk       0.74      1.00      0.85        14

    accuracy                           0.95       100
   macro avg       0.87      0.97      0.91       100
weighted avg       0.96      0.95      0.95       100

```

Top drivers: prev_exam (3.374), study_hours (2.25), parent_education_primary (1.665), parent_education_graduate (1.305), attendance (1.005)

## Economic estimate (assumption-based, INR)
- students treated: 97
- programme cost: 291,150
- students kept on track: 16.0
- economic benefit: 3,032,400
- net benefit: 2,741,250
- benefit cost ratio: 10.42

Charts: distribution.png, drivers.png, subjects.png, feature_importance.png