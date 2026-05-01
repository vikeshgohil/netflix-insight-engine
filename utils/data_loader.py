import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")

    # --- Cleaning Step 1: Fill missing values (modern pandas style) ---
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unknown')
    df['duration'] = df['duration'].fillna('Unknown')
    df['description'] = df['description'].fillna('')

    # --- Cleaning Step 2: Convert date_added to proper datetime ---
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')

    # --- Cleaning Step 3: Extract year and month from date_added ---
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    # --- Cleaning Step 4: Split duration into number and unit ---
    df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['duration_unit'] = df['duration'].str.extract(r'([a-zA-Z]+)')

    # --- Cleaning Step 5: Remove duplicate rows ---
    df = df.drop_duplicates(subset='show_id')

    # --- Cleaning Step 6: Strip whitespace from text columns ---
    df['title'] = df['title'].str.strip()
    df['listed_in'] = df['listed_in'].str.strip()
    df['country'] = df['country'].str.strip()

    # --- Cleaning Step 7: Reset index after all cleaning ---
    df = df.reset_index(drop=True)

    return df


def get_movies(df):
    return df[df['type'] == 'Movie'].copy()


def get_tvshows(df):
    return df[df['type'] == 'TV Show'].copy()


def filter_data(df, keyword='', content_type='All', genre='All',
                country='All', year_range=(1925, 2021)):

    filtered = df.copy()

    if keyword.strip():
        keyword_lower = keyword.lower()
        mask = (
            filtered['title'].str.lower().str.contains(keyword_lower, na=False) |
            filtered['cast'].str.lower().str.contains(keyword_lower, na=False) |
            filtered['director'].str.lower().str.contains(keyword_lower, na=False) |
            filtered['description'].str.lower().str.contains(keyword_lower, na=False)
        )
        filtered = filtered[mask]

    if content_type != 'All':
        filtered = filtered[filtered['type'] == content_type]

    if genre != 'All':
        filtered = filtered[filtered['listed_in'].str.contains(genre, na=False)]

    if country != 'All':
        filtered = filtered[filtered['country'].str.contains(country, na=False)]

    filtered = filtered[
        filtered['release_year'].between(year_range[0], year_range[1])
    ]

    return filtered.reset_index(drop=True)


def get_all_genres(df):
    genres = df['listed_in'].str.split(', ').explode().str.strip().unique()
    return sorted([g for g in genres if g and g != 'Unknown'])


def get_all_countries(df):
    countries = df['country'].str.split(', ').explode().str.strip().unique()
    return sorted([c for c in countries if c and c != 'Unknown'])