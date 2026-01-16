import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;color:#d62828;'>Heart Disease Classification Dashboard</h1>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load trained models and scaler
# --------------------------------------------------
@st.cache_resource
def load_models():
    with open("model/train_models.py", "rb") as f:
        models, scaler = pickle.load(f)
    return models, scaler

models, scaler = load_models()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("User Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

selected_model = models[model_name]

st.sidebar.success(f"Selected Model: {model_name}")

# --------------------------------------------------
# Main Application
# --------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "HeartDisease" not in df.columns:
        st.error("Uploaded CSV must contain 'HeartDisease' column.")
        st.stop()

    X = df.drop("HeartDisease", axis=1)
    y_true = df["HeartDisease"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = selected_model.predict(X_scaled)
    y_prob = selected_model.predict_proba(X_scaled)[:, 1]

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    # --------------------------------------------------
    # Display Dataset
    # --------------------------------------------------
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Display Metrics
    # --------------------------------------------------
    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    metric_items = list(metrics.items())
    for i, (metric, value) in enumerate(metric_items):
        if i % 3 == 0:
            col1.metric(metric, f"{value:.4f}")
        elif i % 3 == 1:
            col2.metric(metric, f"{value:.4f}")
        else:
            col3.metric(metric, f"{value:.4f}")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

else:
    st.info("Please upload a test CSV file to start prediction.")
