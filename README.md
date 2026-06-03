# Spam Detection System using Machine Learning

A Streamlit-based spam detection app that classifies SMS or text messages as **Spam** or **Ham** using machine learning. The project includes text preprocessing, model training, model comparison, single-message prediction, batch CSV prediction, confidence scoring, feedback collection, and visual performance reports.

---

# 📌 Overview

Spam messages often contain misleading offers, suspicious links, urgent wording, or fake rewards. This project uses **Natural Language Processing (NLP)** and **Supervised Machine Learning** to detect such messages and provide a simple interface for testing predictions.

The app is designed for both learning and practical use:

* Train and compare multiple machine learning models
* Detect whether a message is spam or ham
* View prediction confidence and risk level
* Analyze message details such as links, phone numbers, and suspicious keywords
* Upload a CSV file for batch predictions
* Save prediction history and user feedback
* View model accuracy, confusion matrix, and class distribution charts

---

# ✨ Features

## ✅ Single Message Detection

Enter any SMS or text message and get an instant spam or ham prediction.

## ✅ Confidence Score

Displays how confident the model is about the prediction.

## ✅ Risk Level Indicator

Shows **High**, **Medium**, **Low**, or **Review Needed** risk status.

## ✅ Message Analysis

Checks:

* Character count
* Word count
* Links
* Phone numbers
* Money patterns
* Suspicious keywords

## ✅ Batch CSV Upload

Upload a CSV file and classify multiple messages at once.

## ✅ Downloadable Results

Export batch predictions and prediction history as CSV files.

## ✅ Feedback Tracking

Mark predictions as correct or incorrect for future review.

## ✅ Model Performance Dashboard

View:

* Accuracy comparison
* Confusion matrix
* Class distribution charts

---

# 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# 🤖 Machine Learning Models

The training script compares the following models:

* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine
* Random Forest

The best-performing model is saved automatically and used by the Streamlit app.

---

# 📊 Current Best Model

| Metric        | Value                  |
| ------------- | ---------------------- |
| Best Model    | Support Vector Machine |
| Test Accuracy | 98.07%                 |

---

# 📁 Project Structure

```bash
Spam-Detection-System-using-Machine-Learning/
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── spam.csv
├── spam_model.pkl
├── vectorizer.pkl
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
└── outputs/
    ├── metrics.json
    ├── model_scores.csv
    ├── confusion_matrix.png
    ├── model_accuracy.png
    ├── class_distribution.png
    ├── prediction_history.csv
    └── prediction_feedback.csv
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Spam-Detection-System-using-Machine-Learning.git
cd Spam-Detection-System-using-Machine-Learning
```

## 2️⃣ Create and Activate Virtual Environment

### For Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### For macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

# Train the Model

Run the training script to preprocess the dataset, compare models, save the best model, and generate performance charts.

```bash
python train.py
```

This creates or updates:

* `models/spam_model.pkl`
* `models/vectorizer.pkl`
* `outputs/metrics.json`
* `outputs/model_scores.csv`
* `outputs/confusion_matrix.png`
* `outputs/model_accuracy.png`
* `outputs/class_distribution.png`

---

# ▶️ Run the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

# 📂 CSV Upload Format

For batch prediction, upload a CSV file that contains a message column.

The app can automatically detect common column names such as:

* `message`
* `messages`
* `text`
* `sms`

## Example

```csv
message
"Congratulations! You have won a free prize. Click now."
"Can we meet tomorrow for the project discussion?"
```

---

# 📄 Output Files

| File                              | Description                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| `outputs/metrics.json`            | Stores best model, accuracy, scores, and classification report |
| `outputs/model_scores.csv`        | Accuracy comparison of all trained models                      |
| `outputs/confusion_matrix.png`    | Confusion matrix visualization                                 |
| `outputs/model_accuracy.png`      | Model accuracy comparison chart                                |
| `outputs/class_distribution.png`  | Spam vs Ham dataset distribution chart                         |
| `outputs/prediction_history.csv`  | Saved message prediction history                               |
| `outputs/prediction_feedback.csv` | User feedback for predictions                                  |

---

# 🔍 How It Works

1. The dataset is loaded from `spam.csv`
2. Labels are converted into numeric values:

   * `ham = 0`
   * `spam = 1`
3. Messages are cleaned by:

   * Lowercasing text
   * Removing links
   * Removing punctuation
   * Removing digits
   * Removing extra spaces
4. Text is converted into numerical features using **TF-IDF Vectorization**
5. Multiple models are trained and evaluated
6. The best model is saved and used for live predictions
7. The app displays:

   * Prediction confidence
   * Risk level
   * Message-level analysis

---

# 📸 Screenshots

Add screenshots of your Streamlit app here after running the project.

```md
![App Screenshot](screenshots/app.png)
```

---

# 🚀 Future Improvements

* Add user authentication for saved prediction history
* Improve spam explanation using SHAP or LIME
* Add multilingual spam detection
* Deploy the app on Streamlit Community Cloud
* Add automated tests for preprocessing and prediction logic

---

# 👩‍💻 Author

**Purvi Panwalkar**

📧 Email: [panwalkarpurvi@gmail.com](mailto:panwalkarpurvi@gmail.com)

🔗 LinkedIn: Purvi Panwalkar

🎨 Behance: Purvi Panwalkar

---

# 📜 License

This project is open source and available for learning and educational use.
