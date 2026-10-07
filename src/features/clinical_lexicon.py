
"""
clinical_lexicon.py
====================
Adds a targeted phrase-lexicon feature to catch depression IDIOMS
that a plain TF-IDF/bag-of-words model tends to miss - exactly the
failure mode found during manual testing: "I can't get out of bed...
I want to disappear" scored only ~19% depression probability, because
individual words like "bed" and "get" are common in totally unrelated
(non-depression) contexts, diluting the real signal.

This module checks for a curated set of MULTI-WORD PHRASES commonly
associated with depressive symptom language (inspired by clinical
screening instruments like PHQ-9 item wording, and common phrasing
patterns documented in mental-health NLP literature - e.g. Coppersmith
et al. 2014, De Choudhury et al. 2013), and returns a phrase-match
COUNT as an additional feature - so "can't get out of bed" is caught
directly as a phrase, regardless of how the surrounding words score
individually.

IMPORTANT HONESTY NOTE: this lexicon is a hand-curated heuristic, not
learned from data. It's a legitimate, explainable engineering fix for
a known blind spot - not a way to reach "perfect" classification. Some
real depression posts won't use any of these exact phrases, and it can
also fire on phrases used sarcastically or in unrelated contexts. It
is one more signal for the model, not a rule that overrides the model.
"""

import re

# Curated multi-word phrases associated with depressive symptom
# language. Grouped by theme for maintainability - add to these lists
# as more real blind-spot cases are found during testing.
DEPRESSION_PHRASES = [
    # anhedonia / loss of interest
    "nothing feels worth", "don't enjoy", "lost interest", "nothing matters",
    "no point in", "what's the point",
    # hopelessness
    "no hope", "never get better", "nothing will change", "give up",
    "giving up", "no way out",
    # worthlessness / burden
    "i'm worthless", "im worthless", "such a burden", "everyone would be better",
    "hate myself", "not good enough",
    # low energy / incapacitation (the exact blind spot found in testing)
    "can't get out of bed", "cant get out of bed", "can't get up",
    "no energy", "too tired to", "can't function", "can't do anything",
    # passive suicidal ideation / withdrawal (NOT active ideation -
    # active ideation/self-harm content should be escalated separately,
    # outside the scope of this classifier)
    "want to disappear", "wish i could disappear", "want to sleep forever",
    "don't want to exist", "dont want to exist", "want it to end",
    # isolation
    "stopped talking to", "don't want to see anyone", "isolate myself",
    "isolated myself", "cancelled on everyone",
    # numbness
    "feel numb", "feel empty", "feel nothing", "empty inside",
]

# Pre-compile as simple substring patterns (phrases, not full regex,
# since we want literal multi-word matches after normalization -
# clean_text.py already lowercases and strips most punctuation).
# Deduplicated after apostrophe-stripping, since e.g. "can't get out
# of bed" and "cant get out of bed" normalize to the same string.
_NORMALIZED_PHRASES = sorted(set(p.replace("'", "") for p in DEPRESSION_PHRASES))


def phrase_match_features(cleaned_text: str) -> dict:
    """Returns a count of how many curated depression phrases appear
    in the text, plus a binary flag (any match at all)."""
    text = cleaned_text.replace("'", "")
    matches = [p for p in _NORMALIZED_PHRASES if p in text]
    return {
        "depression_phrase_count": len(matches),
        "depression_phrase_present": 1 if matches else 0,
    }


def matched_phrases(cleaned_text: str) -> list:
    """Returns which specific phrases matched - useful for the
    explanation/report layer to show doctors exactly what was caught."""
    text = cleaned_text.replace("'", "")
    return [p for p in _NORMALIZED_PHRASES if p in text]


if __name__ == "__main__":
    tests = [
        "i cant get out of bed everything feels pointless and i just want to disappear",
        "had a great time at the party last night with my friends",
        "nothing feels worth doing anymore i just lie in bed most of the day",
    ]
    for t in tests:
        print(t[:60], "...")
        print("  ", phrase_match_features(t), matched_phrases(t))