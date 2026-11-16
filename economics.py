"""Simple cost-benefit estimator for early-intervention programmes.

Every number below is an ASSUMPTION you should replace with local figures
(school fees, cost of a repeat year, tutoring cost, expected earnings gain).
"""
from dataclasses import dataclass


@dataclass
class Assumptions:
    cost_per_student_intervention: float = 3000.0   # remedial classes / mentoring per year (INR)
    success_rate_of_intervention: float = 0.35      # share of at-risk students who get back on track
    cost_of_failure_repeat_year: float = 40000.0    # extra schooling cost per repeat/dropout (INR)
    lifetime_earnings_gain_per_pass: float = 150000.0  # present-value earnings gain per student kept on track (INR)
    recall_of_model: float = 0.8                    # share of truly at-risk students the model catches
    false_positive_rate: float = 0.15               # on-track students wrongly flagged


def estimate(n_students: int, n_at_risk: int, a: Assumptions = Assumptions()) -> dict:
    caught = n_at_risk * a.recall_of_model
    false_flags = (n_students - n_at_risk) * a.false_positive_rate
    treated = caught + false_flags
    cost = treated * a.cost_per_student_intervention
    saved_students = caught * a.success_rate_of_intervention
    benefit = saved_students * (a.cost_of_failure_repeat_year + a.lifetime_earnings_gain_per_pass)
    return {"students_treated": round(treated), "programme_cost": round(cost),
            "students_kept_on_track": round(saved_students, 1), "economic_benefit": round(benefit),
            "net_benefit": round(benefit - cost), "benefit_cost_ratio": round(benefit / cost, 2) if cost else None}
