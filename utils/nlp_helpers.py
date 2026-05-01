import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

NETFLIX_EXTRA_STOPWORDS = {
    'netflix', 'series', 'season', 'episode', 'film', 'movie',
    'show', 'watch', 'story', 'one', 'two', 'three', 'new',
    'life', 'world', 'man', 'woman', 'find', 'must', 'take'
}

ALL_STOPWORDS = STOP_WORDS.union(NETFLIX_EXTRA_STOPWORDS)


def clean_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    words = text.split()
    filtered = [w for w in words if w not in ALL_STOPWORDS and len(w) > 2]
    return ' '.join(filtered)


def clean_for_nlp(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def prepare_corpus(df, column='description'):
    corpus = df[column].apply(clean_for_nlp)
    return corpus.tolist()


def combine_features(row):
    description = str(row.get('description', ''))
    genre = str(row.get('listed_in', ''))
    cast = str(row.get('cast', ''))
    combined = f"{description} {genre} {cast}"
    return clean_text(combined)


def get_wordcloud_text(df, genre='All Genres'):
    if genre == 'All Genres':
        texts = df['description'].dropna()
    else:
        mask = df['listed_in'].str.contains(genre, na=False)
        texts = df[mask]['description'].dropna()

    combined = ' '.join(texts.tolist())
    cleaned = clean_for_nlp(combined)
    return cleaned