
"""
input_type_filter.py
=====================
A PRE-FILTER that runs BEFORE the risk classifier. Its job is narrow
and specific: decide whether the input text is actually a first-person
personal disclosure at all - because the classifier itself has no
concept of "who is this about." It reacts to word presence, so
"What are the signs of depression?" and "My sister has been
depressed" both light up the same "depression" signal a genuine
first-person disclosure would, even though neither is actually a
personal risk statement from the speaker.

This is a genuine, explainable engineering fix for a real, DEMONSTRATED
blind spot (see manual testing: cases #6, #7, #10 in the 10-category
stress test) - not a claim that it catches every non-personal input.
It is rule-based, not learned, and will have its own edge cases and
false positives/negatives. Treat it the same way as clinical_lexicon.py:
a targeted patch for a known gap, not a general solution.

Categories returned:
  - "too_short"           -> fewer than MIN_WORDS words; the models are not
                             reliable on very short text (e.g. "i am happy"
                             was read as depression because of "i am" alone)
  - "personal_disclosure" -> proceed to normal risk classification
  - "question"            -> interrogative sentence, not a disclosure
  - "third_party"         -> appears to be about someone else, not the speaker
  - "general_statement"   -> appears to be a statistic/academic/news statement
"""

import re

FIRST_PERSON_TOKENS = {"i", "im", "ive", "ill", "id", "me", "my", "myself", "mine"}
THIRD_PERSON_PRONOUNS = {"he", "she", "they", "him", "her", "his", "their", "them",
                          "hes", "shes", "theyre"}

# Relation nouns that, when preceded by "my", shift the subject to a
# third party rather than the speaker ("my sister" is about the
# sister, not the speaker, even though "my" is technically first-person).
RELATION_NOUNS = {
    "sister", "brother", "mother", "mom", "father", "dad", "friend",
    "husband", "wife", "partner", "son", "daughter", "colleague",
    "classmate", "roommate", "boyfriend", "girlfriend", "cousin",
    "aunt", "uncle", "grandma", "grandpa", "grandmother", "grandfather",
    "coworker", "teammate", "neighbor",
}

# Phrases suggesting a general/statistical/academic statement rather
# than a personal disclosure.
GENERAL_STATEMENT_MARKERS = [
    "according to", "studies show", "study shows", "research shows",
    "statistics show", "rates have", "rates of", "prevalence of",
    "has risen", "have risen", "increased significantly",
    "decreased significantly", "study found", "researchers found",
    "survey found", "data shows", "report shows",
]

# Texts shorter than this are not screened. With only a few words, generic
# tokens such as "i" / "am" dominate and both models become unreliable.
MIN_WORDS = 6

QUESTION_WORDS = {"what", "how", "why", "when", "where", "who", "which",
                   "is", "are", "does", "do", "can", "could", "should", "would"}


def _tokenize_lower(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def classify_input_type(raw_text: str) -> dict:
    """Returns {'category': ..., 'reason': ...}."""
    text = raw_text.strip()
    tokens = _tokenize_lower(text)

    # --- 0. Length check ---
    if len(tokens) < MIN_WORDS:
        return {"category": "too_short",
                "reason": f"Only {len(tokens)} word{'s' if len(tokens) != 1 else ''}. Please write at least "
                          f"{MIN_WORDS} words so the models have enough to go on."}

    # --- 1. Question check ---
    if text.rstrip().endswith("?"):
        return {"category": "question",
                "reason": "Sentence is phrased as a question, not a personal statement."}
    first_word = tokens[0] if tokens else ""
    if first_word in QUESTION_WORDS and len(tokens) > 2:
        return {"category": "question",
                "reason": f"Sentence opens with a question word ('{first_word}')."}

    # --- 2. Third-party check ---
    first_person_count = 0
    third_person_count = 0
    matched_relation = None
    for i, tok in enumerate(tokens):
        tok_clean = tok.replace("'", "")
        if tok_clean == "my" and i + 1 < len(tokens):
            next_tok = tokens[i + 1].replace("'", "")
            if next_tok in RELATION_NOUNS:
                third_person_count += 1
                matched_relation = f"my {next_tok}"
                continue  # don't also count this "my" as first-person
        if tok_clean in FIRST_PERSON_TOKENS:
            first_person_count += 1
        elif tok_clean in THIRD_PERSON_PRONOUNS:
            third_person_count += 1

    if third_person_count > first_person_count:
        reason = f"Third-party subject detected ('{matched_relation}')" if matched_relation \
            else "Third-person pronouns outweigh first-person pronouns."
        return {"category": "third_party", "reason": reason}

    # --- 3. General/statistical statement check ---
    text_lower = " ".join(tokens)
    for marker in GENERAL_STATEMENT_MARKERS:
        marker_tokens = marker.split()
        if " ".join(marker_tokens) in text_lower:
            if first_person_count == 0:
                return {"category": "general_statement",
                        "reason": f"Statistical/academic phrasing detected ('{marker}'), no first-person subject."}

    # --- Default: treat as a personal disclosure ---
    return {"category": "personal_disclosure", "reason": "First-person personal statement."}


if __name__ == "__main__":
    tests = [
        "I have been feeling hopeless and empty for weeks and nothing brings me joy anymore.",
        "My sister has been really depressed lately and I don't know how to help her.",
        "What are the early warning signs of depression in teenagers?",
        "Depression rates among college students have risen significantly over the past decade according to recent studies.",
        "I haven't left my room in three days and I don't see the point in getting up anymore.",
        "I'm just so tired.",
        "i am happy",
    ]
    for t in tests:
        result = classify_input_type(t)
        print(f"[{result['category']:20s}] {t[:60]}...")
        print(f"    reason: {result['reason']}")