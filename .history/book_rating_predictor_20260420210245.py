# ============================================================
#   BOOK RATING PREDICTOR - FINAL ML PROJECT (COMPLETE)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

print("=" * 60)
print("   BOOK RATING PREDICTOR - FINAL ML PROJECT")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
print("\n[1/6] Loading dataset...")

df = pd.read_csv("Books.csv", encoding="latin-1", on_bad_lines="skip")

# ------------------------------------------------------------
# STEP 2: CLEAN DATA
# ------------------------------------------------------------
print("\n[2/6] Cleaning data...")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
df = df.dropna(subset=["average_rating"])

df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
df["pages"] = pd.to_numeric(df["pages"], errors="coerce")

df.fillna({
    "ratings_count": 0,
    "pages": df["pages"].median(),
    "description": ""
}, inplace=True)

df["label"] = (df["average_rating"] >= 4).astype(int)

print("Good books:", df["label"].sum())
print("Not good books:", len(df) - df["label"].sum())


# ------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING
# ------------------------------------------------------------
print("\n[3/6] Feature Engineering...")

X_num = df[["average_rating", "ratings_count", "pages"]]

tfidf = TfidfVectorizer(max_features=300, stop_words="english")
X_text = tfidf.fit_transform(df["description"])

X = hstack([X_text, X_num])
y = df["label"]


# ------------------------------------------------------------
# STEP 4: TRAIN MODEL
# ------------------------------------------------------------
print("\n[4/6] Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# ------------------------------------------------------------
# STEP 5: VISUALIZATION (IMPORTANT)
# ------------------------------------------------------------
print("\n[5/6] Creating charts...")

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("confusion_matrix.png")
plt.close()

# Rating Distribution
plt.figure()
plt.hist(df["average_rating"], bins=30)
plt.axvline(x=4, linestyle="--")
plt.title("Rating Distribution")
plt.savefig("rating_distribution.png")
plt.close()

# Good vs Not Good
counts = df["label"].value_counts()
plt.figure()
plt.bar(["Not Good", "Good"], [counts[0], counts[1]])
plt.title("Good vs Not Good Books")
plt.savefig("good_vs_bad.png")
plt.close()

print("Charts saved: confusion_matrix.png, rating_distribution.png, good_vs_bad.png")


# ------------------------------------------------------------
# STEP 6: PREDICT NEW BOOK
# ------------------------------------------------------------
print("\n[6/6] Predict new book...")

rating = float(input("Enter rating (0-5): "))
reviews = int(input("Enter reviews: "))
pages = int(input("Enter pages: "))
description = input("Enter book description: ")

text_vec = tfidf.transform([description])
num_vec = np.array([[rating, reviews, pages]])

final_input = hstack([text_vec, num_vec])

prediction = model.predict(final_input)[0]
probability = model.predict_proba(final_input)[0]

# Smart override
if rating < 3:
    prediction = 0

print("\nPrediction:",
      "GOOD BOOK ✅" if prediction == 1 else "NOT GOOD ❌")

print("Confidence:", round(max(probability)*100, 2), "%")

print("\nPROJECT COMPLETE 🚀")