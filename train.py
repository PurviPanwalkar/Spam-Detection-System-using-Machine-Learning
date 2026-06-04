import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from utils import clean_text


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_dataset():
    df = pd.read_csv(BASE_DIR / "spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df.drop_duplicates(inplace=True)
    df["message"] = df["message"].apply(clean_text)
    return df


def save_confusion_matrix(matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png")
    plt.close()


def save_accuracy_chart(scores):
    names = [score["model"] for score in scores]
    accuracies = [score["accuracy"] for score in scores]

    plt.figure(figsize=(8, 4))
    sns.barplot(x=names, y=accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "model_accuracy.png")
    plt.close()


def save_class_distribution(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="label", data=df)
    plt.xticks([0, 1], ["Ham", "Spam"])
    plt.title("Spam vs Ham Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "class_distribution.png")
    plt.close()


def main():
    df = load_dataset()

    print("Dataset Shape:", df.shape)
    print(df.head())

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42),
    }

    scores = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        predictions = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, predictions)

        trained_models[name] = model
        scores.append({"model": name, "accuracy": round(float(accuracy), 4)})
        print(f"{name}: {accuracy:.4f}")

    best_score = max(scores, key=lambda item: item["accuracy"])
    best_model_name = best_score["model"]
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(X_test_vec)
    matrix = confusion_matrix(y_test, best_predictions)
    report = classification_report(y_test, best_predictions, output_dict=True)

    print("\nBest Model:", best_model_name)
    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions))
    print("\nConfusion Matrix:")
    print(matrix)

    with open(MODELS_DIR / "spam_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    with open(MODELS_DIR / "vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    with open(BASE_DIR / "spam_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    with open(BASE_DIR / "vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    metrics = {
        "best_model": best_model_name,
        "best_accuracy": best_score["accuracy"],
        "model_scores": scores,
        "classification_report": report,
    }

    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(scores).to_csv(OUTPUTS_DIR / "model_scores.csv", index=False)

    save_confusion_matrix(matrix)
    save_accuracy_chart(scores)
    save_class_distribution(df)

    print("\nModel, vectorizer, metrics, and charts saved successfully.")


if __name__ == "__main__":
    main()
