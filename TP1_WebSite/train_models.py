"""Train the TP1 models and save them in the models/ directory."""
from pathlib import Path
import joblib
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Diabetes: preserve the two regression models from the notebook.
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
slr_diabetes = LinearRegression().fit(X[:, [2]], y)
mlr_diabetes = LinearRegression().fit(X, y)
joblib.dump(slr_diabetes, MODEL_DIR / "diabetes_slr.joblib")
joblib.dump(mlr_diabetes, MODEL_DIR / "diabetes_mlr.joblib")
joblib.dump({"feature_names": diabetes.feature_names, "min": X.min(axis=0), "max": X.max(axis=0), "mean": X.mean(axis=0)}, MODEL_DIR / "diabetes_info.joblib")

# Iris: preserve the notebook's sepal-length regression models.
iris = load_iris()
Xi, yi = iris.data, iris.data[:, 0]
iris_slr = LinearRegression().fit(Xi[:, [2]], yi)
iris_mlr = LinearRegression().fit(Xi[:, 1:4], yi)
joblib.dump(iris_slr, MODEL_DIR / "iris_slr.joblib")
joblib.dump(iris_mlr, MODEL_DIR / "iris_mlr.joblib")

# Additional real classification model: predict the Iris species from all 4 measurements.
iris_classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
iris_classifier.fit(Xi, iris.target)
joblib.dump(iris_classifier, MODEL_DIR / "iris_classifier.joblib")
joblib.dump({
    "feature_names": iris.feature_names,
    "target_names": list(iris.target_names),
    "min": Xi.min(axis=0),
    "max": Xi.max(axis=0),
    "mean": Xi.mean(axis=0),
}, MODEL_DIR / "iris_info.joblib")

print("All TP1 models trained and saved successfully.")
