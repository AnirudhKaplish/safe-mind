"""
extract_features.py
====================
Feature extraction module - implements the "Feature Extraction"
stage of the approved design:
  - TF-IDF
  - Sentiment scores (VADER - lexicon-based, no external download)
  - Linguistic markers:
      * first-person singular pronoun ratio (known depression signal
        - depressive language shows increased self-focus, per
        Rude, Gortner & Pennebaker 2004 and related work)
      * negation word count (anxiety/depression often co-occurs with
        negation-heavy phrasing - "can't", "won't", "never")
      * average word length
      * post length (word count)
      * exclamation/question mark counts (arousal indicator)
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.features.clinical_lexicon import phrase_match_features

_analyzer = SentimentIntensityAnalyzer()

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
NEGATIONS = {"not", "no", "never", "cant", "can't", "wont", "won't", "dont", "don't", "n't"}


def sentiment_scores(text: str) -> dict:
    """Returns VADER compound/pos/neg/neu sentiment scores."""
    return _analyzer.polarity_scores(text)


def linguistic_features(text: str) -> dict:
    tokens = text.lower().split()
    n = max(len(tokens), 1)
    first_person_ratio = sum(1 for t in tokens if t in FIRST_PERSON) / n
    negation_count = sum(1 for t in tokens if t in NEGATIONS)
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
    exclam_count = text.count("!")
    question_count = text.count("?")
    return {
        "word_count": len(tokens),
        "first_person_ratio": first_person_ratio,
        "negation_count": negation_count,
        "avg_word_len": avg_word_len,
        "exclam_count": exclam_count,
        "question_count": question_count,
    }


from src.preprocessing.clean_text import light_clean


def build_feature_table(cleaned_texts: list, raw_texts: list = None) -> "pd.DataFrame":
    """Builds the sentiment + linguistic + clinical phrase-lexicon
    feature table (non-TF-IDF features) for a list of cleaned text
    strings.

    `raw_texts`, if given, should be the ORIGINAL (pre-stopword-removal)
    text - phrase matching needs connector words like "out"/"of" intact
    ("can't get out of bed"), which the fully-cleaned TF-IDF text has
    already stripped. Falls back to matching against `cleaned_texts`
    if `raw_texts` isn't provided (will miss some connector-word
    phrases in that case)."""
    import pandas as pd
    if raw_texts is None:
        raw_texts = cleaned_texts
    rows = []
    for t, raw in zip(cleaned_texts, raw_texts):
        s = sentiment_scores(t)
        l = linguistic_features(t)
        p = phrase_match_features(light_clean(raw))
        row = {**s, **l, **p}
        rows.append(row)
    return pd.DataFrame(rows)


def build_tfidf(cleaned_texts: list, max_features: int = 3000):
    """Fits and returns a TF-IDF vectorizer + the transformed matrix."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(cleaned_texts)
    return vectorizer, matrix


if __name__ == "__main__":
    sample = "i can't sleep all lately too much stressed out about everything"
    print("Sentiment  :", sentiment_scores(sample))
    print("Linguistic :", linguistic_features(sample))