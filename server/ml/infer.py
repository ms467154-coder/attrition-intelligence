from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# The joblib artifact contains a DomainInteractions transformer from this module.
# Import it explicitly before unpickling so production and local script execution
# resolve the same serialized class path.
import round2_optimize  # noqa: F401,E402
import joblib
import pandas as pd


class Ensemble:
    """Compatibility class for the serialized Logistic/XGBoost ensemble."""

    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probability = sum(
            self.weights[name] * model.predict_proba(X)[:, 1]
            for name, model in self.models.items()
        )
        return [[float(1 - value), float(value)] for value in probability]


FEATURES = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EnvironmentSatisfaction", "Gender",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

THRESHOLDS = {"high_recall": 0.27, "balanced": 0.32, "high_precision": 0.45}
LABELS = {"high_recall": "High Recall", "balanced": "Balanced", "high_precision": "High Precision"}


_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = Path(__file__).resolve().parents[2] / "ml_artifacts" / "best_f1_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Validated model artifact not found: {path}")
    try:
        _MODEL = joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Unable to load best_f1_model.joblib from {path}: {error}") from error
    return _MODEL


def infer(payload: dict):
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    row = {feature: payload[feature] for feature in FEATURES}
    probability = float(load_model().predict_proba(pd.DataFrame([row]))[0][1])
    modes = {}
    for key, threshold in THRESHOLDS.items():
        modes[key] = {
            "label": LABELS[key],
            "threshold": threshold,
            "predictedAttrition": probability >= threshold,
        }
    return {"probability": probability, "modes": modes, "modelArtifact": "best_f1_model.joblib"}


if __name__ == "__main__":
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(infer(request)))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1)
