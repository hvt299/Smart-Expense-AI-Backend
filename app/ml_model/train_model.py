import os
import csv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from app.ml_model.text_utils import tokenize_vn, clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data.csv")

texts, categories, types = [], [], []

with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(row["text"])
        categories.append(row["category"])
        types.append(row["type"])

clean_texts = [clean_text(t) for t in texts]

cat_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=tokenize_vn, token_pattern=None, ngram_range=(1, 2))),
    ("clf", LinearSVC(dual=False, random_state=42))
])

type_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=tokenize_vn, token_pattern=None, ngram_range=(1, 2))),
    ("clf", LinearSVC(dual=False, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(clean_texts, categories, test_size=0.2, random_state=42)
cat_pipeline.fit(X_train, y_train)
y_pred_cat = cat_pipeline.predict(X_test)

accuracy_cat = accuracy_score(y_test, y_pred_cat)
precision_cat = precision_score(y_test, y_pred_cat, average='weighted', zero_division=1)
recall_cat = recall_score(y_test, y_pred_cat, average='weighted', zero_division=1)
f1_cat = f1_score(y_test, y_pred_cat, average='weighted', zero_division=1)
conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)

print(f"Category Classification Accuracy: {accuracy_cat:.4f}")
print(f"Category Precision: {precision_cat:.4f}")
print(f"Category Recall: {recall_cat:.4f}")
print(f"Category F1-Score: {f1_cat:.4f}")
print("Category Confusion Matrix:\n", conf_matrix_cat)
print("\nCategory Classification Report:\n", classification_report(y_test, y_pred_cat))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cat, annot=True, fmt="d", cmap="Blues", xticklabels=cat_pipeline.classes_, yticklabels=cat_pipeline.classes_)
plt.title("Confusion Matrix (Category)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(BASE_DIR, "category_confusion_matrix.png"))
plt.close()

X_train, X_test, y_train, y_test = train_test_split(clean_texts, types, test_size=0.2, random_state=42)
type_pipeline.fit(X_train, y_train)
y_pred_type = type_pipeline.predict(X_test)

accuracy_type = accuracy_score(y_test, y_pred_type)
precision_type = precision_score(y_test, y_pred_type, average='weighted', zero_division=1)
recall_type = recall_score(y_test, y_pred_type, average='weighted', zero_division=1)
f1_type = f1_score(y_test, y_pred_type, average='weighted', zero_division=1)
conf_matrix_type = confusion_matrix(y_test, y_pred_type)

print(f"Type Classification Accuracy: {accuracy_type:.4f}")
print(f"Type Precision: {precision_type:.4f}")
print(f"Type Recall: {recall_type:.4f}")
print(f"Type F1-Score: {f1_type:.4f}")
print("Type Confusion Matrix:\n", conf_matrix_type)
print("\nType Classification Report:\n", classification_report(y_test, y_pred_type))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_type, annot=True, fmt="d", cmap="Blues", xticklabels=type_pipeline.classes_, yticklabels=type_pipeline.classes_)
plt.title("Confusion Matrix (Type)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(BASE_DIR, "type_confusion_matrix.png"))
plt.close()

model_path = os.path.join(BASE_DIR, "expense_model.pkl")
joblib.dump({"category_model": cat_pipeline, "type_model": type_pipeline}, model_path)
print("Training complete! Model saved at:", model_path)