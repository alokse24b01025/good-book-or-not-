import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack

st.set_page_config(page_title="Book Predictor", layout="centered")

st.title("📚 Book Rating Predictor")
st.write("Predict whether a book is GOOD or NOT")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Books.csv", encoding="latin-1", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")
    df["pages"] = pd.to_numeric(df["pages"], errors="coerce")

    df = df.dropna(subset=["average_rating"])

    df.fillna({
        "ratings_count": 0,
        "pages": df["pages"].median(),
        "description": ""
    }, inplace=True)

    df["label"] = (df["average_rating"] >= 4).astype(int)

    return df

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    X_num = df[["average_rating", "ratings_count", "pages"]]

    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    X_text = tfidf.fit_transform(df["description"])

    X = hstack([X_text, X_num])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    return model, tfidf, acc, cm

model, tfidf, accuracy, cm = train_model(df)

# ---------------- SHOW METRICS ----------------
st.subheader("📊 Model Performance")
st.write(f"Accuracy: {round(accuracy*100,2)}%")

# Confusion Matrix Plot
fig_cm, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig_cm)

# ---------------- DATA CHARTS ----------------
st.subheader("📈 Data Insights")

# Rating Distribution
fig1, ax1 = plt.subplots()
ax1.hist(df["average_rating"], bins=30)
ax1.axvline(x=4, linestyle="--")
ax1.set_title("Rating Distribution")
st.pyplot(fig1)

# Good vs Bad
counts = df["label"].value_counts()
fig2, ax2 = plt.subplots()
ax2.bar(["Not Good", "Good"], [counts[0], counts[1]])
ax2.set_title("Good vs Not Good Books")
st.pyplot(fig2)

# ---------------- USER INPUT ----------------
st.subheader("🔍 Enter Book Details")

rating = st.slider("Rating", 0.0, 5.0, 3.5)
reviews = st.number_input("Number of Reviews", 0, 1000000, 1000)
pages = st.number_input("Pages", 1, 2000, 300)
description = st.text_area("Book Description")

# ---------------- PREDICT ----------------
if st.button("Predict"):
    text_vec = tfidf.transform([description])
    num_vec = np.array([[rating, reviews, pages]])
    final_input = hstack([text_vec, num_vec])

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0]

    # Rule override
    if rating < 3:
        prediction = 0

    if prediction == 1:
        st.success("📗 GOOD BOOK")
    else:
        st.error("📕 NOT A GOOD BOOK")

    st.write(f"Confidence: {round(max(probability)*100,2)}%")