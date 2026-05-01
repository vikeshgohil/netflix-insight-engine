import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from utils.nlp_helpers import combine_features, clean_for_nlp


@st.cache_data
def build_recommendation_model(df):
    df = df.copy()
    df['combined'] = df.apply(combine_features, axis=1)

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix, df


def get_recommendations(title, df, similarity_matrix, n=5):
    matches = df[df['title'].str.lower() == title.lower()]

    if matches.empty:
        matches = df[df['title'].str.lower().str.contains(
            title.lower(), na=False
        )]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]
    sim_scores = sim_scores[:n]

    result_indices = [s[0] for s in sim_scores]
    result_scores = [round(s[1], 3) for s in sim_scores]

    results = df.iloc[result_indices][
        ['title', 'type', 'listed_in', 'rating',
         'release_year', 'description']
    ].copy()
    results['similarity_score'] = result_scores

    return results.reset_index(drop=True)


@st.cache_data
def train_genre_model(df):
    df = df.copy()
    df['primary_genre'] = df['listed_in'].str.split(',').str[0].str.strip()
    genre_counts = df['primary_genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 30].index
    df = df[df['primary_genre'].isin(valid_genres)]
    df['clean_description'] = df['description'].apply(clean_for_nlp)
    df = df[df['clean_description'].str.strip() != '']

    X = df['clean_description']
    y = df['primary_genre']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42

    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    return model, tfidf, le, accuracy


def predict_genre(text, model, tfidf, le, top_n=3):
    cleaned = clean_for_nlp(text)
    if not cleaned.strip():
        return []

    text_vec = tfidf.transform([cleaned])
    probabilities = model.predict_proba(text_vec)[0]

    top_indices = probabilities.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        genre = le.inverse_transform([idx])[0]
        confidence = round(probabilities[idx] * 100, 1)
        results.append({
            'genre': genre,
            'confidence': confidence
        })

    return results