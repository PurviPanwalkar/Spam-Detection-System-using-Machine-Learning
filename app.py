import csv
import html
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import analyze_message, clean_text


BASE_DIR = Path(__file__).parent
MODEL_PATHS = [BASE_DIR / "models" / "spam_model.pkl", BASE_DIR / "spam_model.pkl"]
VECTORIZER_PATHS = [BASE_DIR / "models" / "vectorizer.pkl", BASE_DIR / "vectorizer.pkl"]
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
HISTORY_PATH = BASE_DIR / "outputs" / "prediction_history.csv"
FEEDBACK_PATH = BASE_DIR / "outputs" / "prediction_feedback.csv"


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


@st.cache_resource
def load_model_files():
    model_path = first_existing(MODEL_PATHS)
    vectorizer_path = first_existing(VECTORIZER_PATHS)

    if not model_path.exists() or not vectorizer_path.exists():
        st.error("Model files are missing. Please run: python train.py")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_prediction(message, prediction, confidence):
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    file_exists = HISTORY_PATH.exists()

    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["date_time", "message", "prediction", "confidence"])
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                message,
                prediction,
                round(confidence, 2),
            ]
        )


def save_feedback(message, prediction, confidence, feedback):
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    file_exists = FEEDBACK_PATH.exists()

    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["date_time", "message", "prediction", "confidence", "feedback"])
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                message,
                prediction,
                round(confidence, 2),
                feedback,
            ]
        )


def predict_message(message):
    cleaned_message = clean_text(message)
    vector = vectorizer.transform([cleaned_message])
    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(vector)[0]
        confidence = float(probability[prediction]) * 100
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(vector)[0])
        spam_probability = 1 / (1 + math.exp(-score))
        confidence = spam_probability * 100 if prediction == 1 else (1 - spam_probability) * 100
    else:
        confidence = 100.0

    label = "Spam" if prediction == 1 else "Ham"
    return label, confidence


def predict_batch(messages):
    rows = []

    for message in messages:
        prediction, confidence = predict_message(str(message))
        risk_label, _ = get_risk_level(prediction, confidence)
        analysis = analyze_message(str(message))
        rows.append(
            {
                "message": message,
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "risk_level": risk_label,
                "suspicious_keywords": ", ".join(analysis["suspicious_keywords"]),
                "contains_link": "Detected" if analysis["has_link"] else "Not Detected",
                "phone_number": "Detected" if analysis["has_phone"] else "Not Detected",
            }
        )

    return pd.DataFrame(rows)


def get_risk_level(prediction, confidence):
    if prediction == "Spam" and confidence >= 75:
        return "High Risk", "risk-high"
    if prediction == "Spam":
        return "Medium Risk", "risk-medium"
    if confidence >= 75:
        return "Low Risk", "risk-low"
    return "Needs Review", "risk-medium"


def render_confidence_bar(confidence, risk_class):
    st.markdown(
        f"""
        <div class="confidence-track">
            <div class="confidence-fill {risk_class}" style="width: {confidence:.0f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_checks(history):
    recent = history.tail(5).iloc[::-1]

    for _, row in recent.iterrows():
        prediction = row["prediction"]
        badge_class = "risk-high" if prediction == "Spam" else "risk-low"
        message = html.escape(str(row["message"]))
        short_message = message[:80] + ("..." if len(message) > 80 else "")

        st.markdown(
            f"""
            <div class="history-row">
                <div>
                    <span class="history-message">{short_message}</span>
                    <span class="history-date">{row["date_time"]}</span>
                </div>
                <span class="badge {badge_class}">{prediction}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Spam Detection System", page_icon="Mail", layout="wide")

model, vectorizer = load_model_files()
metrics = load_metrics()

st.title("Spam Detection System")
st.caption("Machine learning based SMS classifier with confidence score and text analysis.")

st.markdown(
    """
    <style>
        .badge {
            border-radius: 999px;
            color: white;
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
        }

        .risk-high {
            background: #dc2626;
        }

        .risk-medium {
            background: #d97706;
        }

        .risk-low {
            background: #16a34a;
        }

        .confidence-track {
            background: #e5e7eb;
            border-radius: 999px;
            height: 0.75rem;
            margin: 0.75rem 0 1rem;
            overflow: hidden;
            width: 100%;
        }

        .confidence-fill {
            border-radius: 999px;
            height: 100%;
        }

        .history-row {
            align-items: center;
            border: 1px solid #e5e7eb;
            border-radius: 0.5rem;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            padding: 0.7rem 0.8rem;
        }

        .history-message {
            display: block;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .history-date {
            color: #6b7280;
            display: block;
            font-size: 0.75rem;
            margin-top: 0.15rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.subheader("Check a Message")
    sample = st.selectbox(
        "Try an example",
        [
            "",
            "Congratulations! You have won a free iPhone. Click here to claim now.",
            "URGENT! Your account has been selected for a cash prize. Call now.",
            "Hey, are you coming to class today?",
            "Can we meet tomorrow for the project discussion?",
        ],
    )

    message = st.text_area("Enter message", value=sample, height=160)

    if st.button("Analyze Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message.")
        else:
            prediction, confidence = predict_message(message)
            analysis = analyze_message(message)
            risk_label, risk_class = get_risk_level(prediction, confidence)
            save_prediction(message, prediction, confidence)
            st.session_state["last_prediction"] = {
                "message": message,
                "prediction": prediction,
                "confidence": confidence,
            }

            if prediction == "Spam":
                st.error(f"Result: Spam detected with {confidence:.2f}% confidence")
            else:
                st.success(f"Result: Ham detected with {confidence:.2f}% confidence")

            st.markdown(
                f'Risk Level: <span class="badge {risk_class}">{risk_label}</span>',
                unsafe_allow_html=True,
            )
            render_confidence_bar(confidence, risk_class)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Characters", analysis["characters"])
            metric_cols[1].metric("Words", analysis["words"])
            metric_cols[2].metric("Link", "Detected" if analysis["has_link"] else "Not Detected")
            metric_cols[3].metric("Phone Number", "Detected" if analysis["has_phone"] else "Not Detected")

            if analysis["suspicious_keywords"]:
                st.warning(
                    "Suspicious keywords found: "
                    + ", ".join(analysis["suspicious_keywords"])
                )
            else:
                st.info("No common spam keywords found.")

    if "last_prediction" in st.session_state:
        last_prediction = st.session_state["last_prediction"]
        st.caption("Was this prediction correct?")
        feedback_cols = st.columns(2)

        if feedback_cols[0].button("Correct", use_container_width=True):
            save_feedback(
                last_prediction["message"],
                last_prediction["prediction"],
                last_prediction["confidence"],
                "Correct",
            )
            st.success("Feedback saved.")

        if feedback_cols[1].button("Incorrect", use_container_width=True):
            save_feedback(
                last_prediction["message"],
                last_prediction["prediction"],
                last_prediction["confidence"],
                "Incorrect",
            )
            st.warning("Feedback saved for review.")

    st.subheader("Batch CSV Upload")
    uploaded_file = st.file_uploader("Upload a CSV file with messages", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if batch_df.empty:
            st.warning("The uploaded CSV file is empty.")
        else:
            default_index = 0
            for index, column in enumerate(batch_df.columns):
                if column.lower() in ["message", "messages", "text", "sms"]:
                    default_index = index
                    break

            message_column = st.selectbox(
                "Select the message column",
                batch_df.columns,
                index=default_index,
            )

            if st.button("Analyze CSV", use_container_width=True):
                results = predict_batch(batch_df[message_column].fillna(""))
                st.session_state["batch_results"] = results

            if "batch_results" in st.session_state:
                st.dataframe(
                    st.session_state["batch_results"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download Predictions",
                    data=st.session_state["batch_results"].to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

with right_col:
    st.subheader("Model Summary")

    if metrics:
        st.metric("Best Model", metrics.get("best_model", "Not available"))
        st.metric("Model Test Accuracy", f"{metrics.get('best_accuracy', 0) * 100:.2f}%")
    else:
        st.info("Run python train.py again to generate the latest model summary.")

    st.subheader("Recent Checks")
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        render_recent_checks(history)

        st.download_button(
            "Download History",
            data=history.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    else:
        st.caption("No predictions checked yet.")

    st.subheader("Feedback")
    if FEEDBACK_PATH.exists():
        feedback = pd.read_csv(FEEDBACK_PATH)
        total_feedback = len(feedback)
        incorrect_feedback = int((feedback["feedback"] == "Incorrect").sum())
        st.metric("Feedback Received", total_feedback)
        st.metric("Needs Review", incorrect_feedback)
    else:
        st.caption("No feedback submitted yet.")

st.divider()

with st.expander("View Model Performance"):
    if metrics:
        report = metrics.get("classification_report", {})
        spam_report = report.get("1", {})

        performance_cols = st.columns(3)
        performance_cols[0].metric("Spam Precision", f"{spam_report.get('precision', 0) * 100:.2f}%")
        performance_cols[1].metric("Spam Recall", f"{spam_report.get('recall', 0) * 100:.2f}%")
        performance_cols[2].metric("Spam F1-score", f"{spam_report.get('f1-score', 0) * 100:.2f}%")

        comparison = pd.DataFrame(metrics.get("model_scores", []))
        if not comparison.empty:
            st.caption("Development comparison")
            st.dataframe(comparison, use_container_width=True, hide_index=True)

    chart_cols = st.columns(3)
    charts = [
        ("Confusion Matrix", BASE_DIR / "outputs" / "confusion_matrix.png"),
        ("Model Accuracy", BASE_DIR / "outputs" / "model_accuracy.png"),
        ("Class Distribution", BASE_DIR / "outputs" / "class_distribution.png"),
    ]

    for column, (title, path) in zip(chart_cols, charts):
        with column:
            st.caption(title)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Run python train.py to create this chart.")
