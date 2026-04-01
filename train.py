# ==============================
# 📧 Spam Detection - Training Script
# ==============================

# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove duplicates
df.drop_duplicates(inplace=True)

print("Dataset Shape:", df.shape)
print(df.head())


# ==============================
# 3. Text Preprocessing
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'\W', ' ', text)       # remove special chars
    text = re.sub(r'\d', '', text)        # remove numbers
    text = re.sub(r'\s+', ' ', text)      # remove extra spaces
    return text

df['message'] = df['message'].apply(clean_text)


# ==============================
# 4. Train-Test Split
# ==============================
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 5. TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ==============================
# 6. Train Models
# ==============================
nb_model = MultinomialNB()
lr_model = LogisticRegression()

nb_model.fit(X_train_vec, y_train)
lr_model.fit(X_train_vec, y_train)


# ==============================
# 7. Predictions
# ==============================
nb_pred = nb_model.predict(X_test_vec)
lr_pred = lr_model.predict(X_test_vec)


# ==============================
# 8. Accuracy Comparison
# ==============================
nb_acc = accuracy_score(y_test, nb_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print("\n📊 Model Accuracy:")
print("Naive Bayes:", nb_acc)
print("Logistic Regression:", lr_acc)


# ==============================
# 9. Detailed Report
# ==============================
print("\n📄 Classification Report (Naive Bayes):")
print(classification_report(y_test, nb_pred))

print("\n📄 Confusion Matrix:")
cm = confusion_matrix(y_test, nb_pred)
print(cm)


# ==============================
# 10. Visualization
# ==============================

# 🔹 Confusion Matrix Heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 🔹 Model Accuracy Comparison
models = ['Naive Bayes', 'Logistic Regression']
scores = [nb_acc, lr_acc]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()


# 🔹 Class Distribution
df['label'].value_counts().plot(kind='bar')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title("Spam vs Ham Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# ==============================
# 11. Save Best Model
# ==============================
best_model = lr_model   # Logistic Regression performs better

pickle.dump(best_model, open('spam_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("\n✅ Model and Vectorizer saved successfully!")


# ==============================
# 12. Test Prediction Function
# ==============================
def predict_message(msg):
    msg = clean_text(msg)
    vec = vectorizer.transform([msg])
    result = best_model.predict(vec)
    
    return "Spam 🚫" if result[0] == 1 else "Ham ✅"

# Test examples
print("\n🔍 Sample Predictions:")
print(predict_message("Congratulations! You won a free iPhone"))
print(predict_message("Hey, are you coming to class?"))
