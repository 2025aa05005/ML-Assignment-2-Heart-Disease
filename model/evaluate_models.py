import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load dataset
df = pd.read_csv("synthetic_heart_dataset.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Load trained models
with open("model/trained_models.pkl", "rb") as f:
    models, scaler = pickle.load(f)

X_scaled = scaler.transform(X)

results = {}

for name, model in models.items():
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    }

# Display results
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
