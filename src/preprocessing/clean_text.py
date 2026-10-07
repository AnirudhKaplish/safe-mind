"""
clean_text.py
=============
Text preprocessing module for the Mental Health Detection pipeline.

Implements the preprocessing steps listed in the approved design
(Design Specifications - Data Processing):
  - URL removal
  - emoji removal
  - normalization (lowercasing, punctuation handling)
  - tokenization
  - stop-word removal (while preserving negation/contrast words,
    since "not happy" / "can't sleep" carry real clinical signal
    and stripping "not" would flip the meaning of the sentence)
"""

import re
import string

# Words we deliberately KEEP even though a generic stop-word list
# would remove them - they carry meaningful signal for mental
# health text (negation, contrast, intensifiers).
KEEP_WORDS = {
    "not", "no", "nor", "never", "neither",          # negation
    "but", "however", "although", "though", "yet",    # contrast
    "still", "despite", "whereas", "while",           # concession
    "very", "too", "extremely", "quite",              # intensifiers
    "i", "me", "my", "myself",                        # first-person pronouns
                                                       # (known clinical signal -
                                                       # depressive text uses more
                                                       # first-person singular pronouns)
}

# A compact stop-word list (kept self-contained so this module has
# no external NLTK download dependency - NLTK's stopword corpus
# requires an internet download via nltk.download() which may not
# be available in restricted environments).
STOP_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "and", "or", "as", "at", "by", "for", "from", "in", "into", "of", "on",
    "to", "with", "this", "that", "these", "those", "it", "its", "he", "she",
    "him", "her", "his", "hers", "they", "them", "their", "we", "us", "our",
    "you", "your", "do", "does", "did", "doing", "have", "has", "had", "having",
    "will", "would", "can", "could", "should", "there", "here", "than", "then",
    "so", "if", "about", "just", "up", "down", "out", "over", "under", "again",
} - KEEP_WORDS

URL_RE = re.compile(r"http\S+|www\.\S+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
MULTISPACE_RE = re.compile(r"\s+")


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)


def normalize(text: str) -> str:
    text = text.lower()
    # keep apostrophes inside words (don't -> don't) but strip other punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list:
    return text.split()


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS]


def light_clean(raw_text: str) -> str:
    """Lighter cleaning pass for PHRASE matching: removes URLs/emojis
    and normalizes case/punctuation, but deliberately SKIPS stop-word
    removal. Multi-word idioms like "can't get out of bed" need their
    connector words ("out", "of") intact to be recognized as a phrase
    - the full clean_pipeline() strips those words for TF-IDF purposes,
    which silently breaks phrase-lexicon matching if used for that too."""
    text = remove_urls(raw_text)
    text = remove_emojis(text)
    text = normalize(text)
    return text


def clean_pipeline(raw_text: str) -> str:
    """Full cleaning pipeline: URL/emoji removal -> normalize ->
    tokenize -> stop-word removal (with negation/contrast preserved)
    -> rejoin into a cleaned string ready for feature extraction."""
    text = remove_urls(raw_text)
    text = remove_emojis(text)
    text = normalize(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "I can't sleep at all lately :( check this out https://example.com/thread it's just too much!!"
    print("RAW    :", sample)
    print("CLEANED:", clean_pipeline(sample))