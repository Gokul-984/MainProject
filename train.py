<<<<<<< HEAD
import pandas as pd
import re
import glob
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# ----------------------------
# Text Cleaning (FAST + LIGHT)
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)      # letters only
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ----------------------------
# Load & Merge Multiple Datasets
# ----------------------------
def load_datasets(path_pattern):
    """
    path_pattern example:
    'data/*.csv'
    """
    files = glob.glob(path_pattern)
    if not files:
        raise ValueError("No dataset files found!")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if 'body' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Missing columns in {file}")
        dfs.append(df[['body', 'label']])

    return pd.concat(dfs, ignore_index=True)


# ----------------------------
# Main Training Pipeline
# ----------------------------
def train_model(data_path_pattern):

    # Load data
    df = load_datasets(data_path_pattern)

    # Clean text
    df['body'] = df['body'].apply(clean_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['body'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # TF-IDF (EDGE FRIENDLY)
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(model, "phishing_nb_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

    print("\nModel and vectorizer saved successfully.")


# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    # Example: all CSV files in data folder
    train_model("data/*.csv")
=======
import pandas as pd
import re
import glob
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# ----------------------------
# Text Cleaning (FAST + LIGHT)
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)      # letters only
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ----------------------------
# Load & Merge Multiple Datasets
# ----------------------------
def load_datasets(path_pattern):
    """
    path_pattern example:
    'data/*.csv'
    """
    files = glob.glob(path_pattern)
    if not files:
        raise ValueError("No dataset files found!")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if 'body' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Missing columns in {file}")
        dfs.append(df[['body', 'label']])

    return pd.concat(dfs, ignore_index=True)


# ----------------------------
# Main Training Pipeline
# ----------------------------
def train_model(data_path_pattern):

    # Load data
    df = load_datasets(data_path_pattern)

    # Clean text
    df['body'] = df['body'].apply(clean_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['body'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # TF-IDF (EDGE FRIENDLY)
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(model, "phishing_nb_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

    print("\nModel and vectorizer saved successfully.")


# ----------------------------
# Run Training
# ----------------------------
if __name__ == "__main__":
    # Example: all CSV files in data folder
    train_model("data/*.csv")
>>>>>>> ef49141 (Add phishing dataset, ML model, and training scripts)
