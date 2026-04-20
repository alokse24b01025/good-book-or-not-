# ============================================================
#   BOOK RATING PREDICTOR - Full ML Project
#   Predicts whether a book is "good" (rating >= 4.0) or not
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

print("=" * 55)
print("   BOOK RATING PREDICTOR - ML Project")
print("=" * 55)


# ─────────────────────────────────────────────
# STEP 1: LOAD THE DATASET
# ─────────────────────────────────────────────
print("\n[1/6] Loading dataset...")

try:
    df = pd.read_csv("Books.csv", on_bad_lines="skip", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("Books.csv", on_bad_lines="skip", encoding="latin-1")

print(f"    Loaded {len(df)} books with {len(df.columns)} columns")
print(f"    Columns found: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 2: EXPLORE THE DATA
# ─────────────────────────────────────────────
print("\n[2/6] Exploring data...")
print(df.head(3).to_string())
print(f"\n    Missing values:\n{df.isnull().sum()}")


# ─────────────────────────────────────────────
# STEP 3: CLEAN AND PREPARE DATA
# ─────────────────────────────────────────────
print("\n[3/6] Cleaning data...")

# Normalize column names: lowercase + strip spaces
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(f"    Normalized columns: {list(df.columns)}")

# Find the rating column automatically
rating_col = None
for col in df.columns:
    if "average_rating" in col or col == "rating":
        rating_col = col
        break

if rating_col is None:
    print("\n  ERROR: Could not find a rating column.")
    print("  Your columns are:", list(df.columns))
    print("  Look for the column with ratings like 3.5, 4.2 etc.")
    exit()

print(f"    Using rating column: '{rating_col}'")

df = df.dropna(subset=[rating_col])
df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
df = df.dropna(subset=[rating_col])

# Create TARGET: 1 = Good Book (rating >= 4.0), 0 = Not Good
df["is_good_book"] = (df[rating_col] >= 4.0).astype(int)

print(f"    Good books  (>= 4.0) : {df['is_good_book'].sum()}")
print(f"    Other books (<  4.0) : {(df['is_good_book'] == 0).sum()}")

# Clean numeric columns
for col in ["ratings_count", "text_reviews_count", "num_pages",
            "  num_pages", "ratings_count", "work_ratings_count"]:
    col = col.strip()
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df_num = df.select_dtypes(include=[np.number])
df[df_num.columns] = df_num.fillna(df_num.median())


# ─────────────────────────────────────────────
# STEP 4: PICK FEATURES AND SPLIT DATA
# ─────────────────────────────────────────────
print("\n[4/6] Preparing features...")

# Try to find useful numeric feature columns automatically
candidate_features = [
    "ratings_count", "work_ratings_count", "work_text_reviews_count",
    "text_reviews_count", "num_pages", "  num_pages", "books_count"
]
features = []
for f in candidate_features:
    f = f.strip()
    if f in df.columns and f != rating_col and f != "is_good_book":
        features.append(f)

if len(features) == 0:
    # fallback: use all numeric columns except target and rating
    features = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in [rating_col, "is_good_book", "book_id",
                              "goodreads_book_id", "best_book_id", "work_id",
                              "isbn", "isbn13"]]

print(f"    Features selected: {features}")

X = df[features]
y = df["is_good_book"]

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# STEP 5: TRAIN TWO MODELS AND COMPARE
# ─────────────────────────────────────────────
print("\n[5/6] Training models...")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred  = lr_model.predict(X_test_scaled)
lr_acc   = accuracy_score(y_test, lr_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_pred)

print(f"\n    Logistic Regression Accuracy : {lr_acc * 100:.2f}%")
print(f"    Random Forest Accuracy       : {rf_acc * 100:.2f}%")

if rf_acc >= lr_acc:
    best_model = rf_model
    best_pred  = rf_pred
    best_name  = "Random Forest"
    best_acc   = rf_acc
else:
    best_model = lr_model
    best_pred  = lr_pred
    best_name  = "Logistic Regression"
    best_acc   = lr_acc

print(f"\n    Best model : {best_name} ({best_acc * 100:.2f}% accuracy)")
print(f"\n    Detailed Report:\n")
print(classification_report(y_test, best_pred,
      target_names=["Not Good", "Good Book"]))


# ─────────────────────────────────────────────
# STEP 6: VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[6/6] Creating charts...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Book Rating Predictor - ML Project Results",
             fontsize=15, fontweight="bold")

# Chart 1: Rating distribution
axes[0, 0].hist(df[rating_col].dropna(), bins=30,
                color="#4C72B0", edgecolor="white")
axes[0, 0].axvline(x=4.0, color="red", linestyle="--",
                   linewidth=2, label="4.0 threshold")
axes[0, 0].set_title("Distribution of Book Ratings")
axes[0, 0].set_xlabel("Average Rating")
axes[0, 0].set_ylabel("Number of Books")
axes[0, 0].legend()

# Chart 2: Good vs Not Good
counts = df["is_good_book"].value_counts()
bars = axes[0, 1].bar(
    ["Not Good (< 4.0)", "Good Book (≥ 4.0)"],
    [counts.get(0, 0), counts.get(1, 0)],
    color=["#DD8452", "#55A868"], edgecolor="white"
)
for bar in bars:
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(int(bar.get_height())),
        ha="center", fontweight="bold"
    )
axes[0, 1].set_title("Good vs Not-Good Books")
axes[0, 1].set_ylabel("Count")

# Chart 3: Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Not Good", "Good Book"])
disp.plot(ax=axes[1, 0], colorbar=False, cmap="Blues")
axes[1, 0].set_title(f"Confusion Matrix — {best_name}")

# Chart 4: Accuracy comparison
accs = [lr_acc * 100, rf_acc * 100]
bar2 = axes[1, 1].bar(["Logistic\nRegression", "Random\nForest"],
                       accs, color=["#4C72B0", "#55A868"],
                       edgecolor="white", width=0.5)
for b, a in zip(bar2, accs):
    axes[1, 1].text(b.get_x() + b.get_width() / 2,
                    b.get_height() - 4,
                    f"{a:.1f}%", ha="center",
                    color="white", fontweight="bold", fontsize=12)
axes[1, 1].set_title("Model Accuracy Comparison")
axes[1, 1].set_ylabel("Accuracy (%)")
axes[1, 1].set_ylim(0, 115)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Saved results.png in your project folder")


# ─────────────────────────────────────────────
# STEP 7: PREDICT A NEW BOOK
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("   PREDICTING A NEW BOOK")
print("=" * 55)

# Fill in values based on which features were available
sample_values = {
    "ratings_count":            50000,
    "work_ratings_count":       55000,
    "work_text_reviews_count":  2000,
    "text_reviews_count":       2000,
    "num_pages":                320,
    "books_count":              5,
}

new_input = pd.DataFrame(
    [[sample_values.get(f, 0) for f in features]],
    columns=features
)

if best_name == "Logistic Regression":
    new_scaled   = scaler.transform(new_input)
    prediction   = best_model.predict(new_scaled)[0]
    probability  = best_model.predict_proba(new_scaled)[0]
else:
    prediction   = best_model.predict(new_input)[0]
    probability  = best_model.predict_proba(new_input)[0]

print(f"\n    Input values : {dict(zip(features, new_input.values[0]))}")
print(f"    Prediction   : {'✅  GOOD BOOK' if prediction == 1 else '❌  NOT A GOOD BOOK'}")
print(f"    Confidence   : {max(probability) * 100:.1f}%")

print("\n" + "=" * 55)
print("   PROJECT COMPLETE!  Check results.png for charts.")
print("=" * 55)