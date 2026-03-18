import os
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from app.ml_model.text_utils import tokenize_vn, clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data.csv")

texts = []
labels = []

with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row["text"])
        labels.append(row["category"])

clean_texts = [clean_text(t) for t in texts]

X_train, X_test, y_train, y_test = train_test_split(
    clean_texts, labels, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=tokenize_vn,
        token_pattern=None,
        ngram_range=(1, 2),
        max_df=0.9
    )),
    ("clf", LinearSVC(
        dual=False,
        random_state=42
    ))
])

print("Đang huấn luyện AI...")
pipeline.fit(X_train, y_train)
print("Accuracy:", pipeline.score(X_test, y_test))

model_path = os.path.join(BASE_DIR, "expense_model.pkl")
joblib.dump(pipeline, model_path)
print("Lưu mô hình tại: ", model_path)