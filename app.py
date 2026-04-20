import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
    # ✅ FIX: include rating
    X_num = df[["average_rating", "ratings_count", "pages"]]

    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    X_text = tfidf.fit_transform(df["description"])

    X = hstack([X_text, X_num])
    y = df["label"]

    # ✅ FIX: balanced model
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    model.fit(X, y)

    return model, tfidf

model, tfidf = train_model(df)

# ---------------- USER INPUT ----------------
st.subheader("Enter Book Details")

rating = st.slider("Rating", 0.0, 5.0, 3.5)
reviews = st.number_input("Number of Reviews", 0, 1000000, 1000)
pages = st.number_input("Pages", 1, 2000, 300)
description = st.text_area("Book Description")

# ---------------- PREDICT ----------------
if st.button("Predict"):
    text_vec = tfidf.transform([description])

    # ✅ FIX: include rating in input
    num_vec = np.array([[rating, reviews, pages]])

    final_input = hstack([text_vec, num_vec])

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0]

    # ✅ STRONG RULE FIX
    if rating < 3:
        prediction = 0

    # ---------------- OUTPUT ----------------
    if prediction == 1:
        st.success("📗 GOOD BOOK")
    else:
        st.error("📕 NOT A GOOD BOOK")

    st.write(f"Confidence: {round(max(probability)*100,2)}%")