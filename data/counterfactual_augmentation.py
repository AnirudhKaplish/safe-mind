
"""
counterfactual_augmentation.py
================================
Generates a small set of COUNTERFACTUAL training examples specifically
designed to break a suspected shortcut in the fine-tuned BERT model:
it appears to associate the literal word "depressed"/"depression"
almost one-to-one with the depression class (likely because the
source dataset's depression class was scraped from r/depression,
making vocabulary itself a near-perfect proxy label during training,
rather than actual symptom content).

Two groups of counterexamples, each deliberately DIVERSE (not just
variations of the two sentences that happened to fail) so the model
learns a generalizable pattern, not a memorized patch:

  A) RECOVERY / PAST-TENSE mentions of depression-related words,
     labeled not_depression - teaches that using the word "depressed"
     doesn't automatically mean a current depression disclosure.

  B) SYMPTOM-ONLY descriptions with NO depression-related vocabulary,
     labeled depression - teaches that genuine risk indicators
     (incapacitation, anhedonia, hopelessness) matter more than
     specific keywords.

HONESTY NOTE: this is a real, principled fix (counterfactual/
adversarial data augmentation is an established NLP technique - see
Kaushik et al. 2020), not a hack. But it only works if these examples
are diverse enough to generalize - adding just the 2 known failing
sentences back into training would be memorization, not learning.
Merge this into your real training data, don't fine-tune on it alone.

Usage:
    python counterfactual_augmentation.py
    # writes augmented_dataset.csv - merged real data + these examples
"""

import pandas as pd

RECOVERY_EXAMPLES = [
    "I used to feel depressed but honestly I'm doing so much better now, life is good.",
    "I went through a really depressed period last year, but therapy helped a lot and I feel like myself again.",
    "Looking back, I was definitely depressed in college, but these days I actually enjoy my life.",
    "I was diagnosed with depression a few years ago and it was rough, but I've made real progress since then.",
    "My depression used to control everything, but I've been in a good place for months now.",
    "I remember being so depressed after the breakup, but I've healed a lot and I'm genuinely happy now.",
    "Depression hit me hard after I lost my job, but starting a new career gave me my spark back.",
    "I battled depression for years; today I actually look forward to getting up in the morning.",
]

SYMPTOM_ONLY_EXAMPLES = [
    "I haven't left my room in three days and I don't see the point in getting up anymore.",
    "I can't remember the last time I actually wanted to do anything, everything just feels grey.",
    "Getting out of bed has become the hardest part of my day, and I don't know why I bother most mornings.",
    "I've stopped answering texts because I just don't have it in me to talk to anyone right now.",
    "Nothing brings me any joy anymore, not even the things I used to love doing every day.",
    "I keep canceling plans because I can't find the energy to leave the house at all.",
    "I feel like I'm just going through the motions, nothing really matters to me these days.",
    "I've been sleeping twelve hours a day and I still wake up feeling completely empty inside.",
]


def build_augmented_dataset(real_data_path="data/seed_dataset.csv",
                             output_path="data/augmented_dataset.csv"):
    real_df = pd.read_csv(real_data_path)

    aug_rows = []
    for text in RECOVERY_EXAMPLES:
        aug_rows.append({"text": text, "label": "not_depression"})
    for text in SYMPTOM_ONLY_EXAMPLES:
        aug_rows.append({"text": text, "label": "depression"})
    aug_df = pd.DataFrame(aug_rows)

    combined = pd.concat([real_df[["text", "label"]], aug_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"Real data: {len(real_df)} rows")
    print(f"Counterfactual examples added: {len(aug_df)}")
    print(f"Combined dataset saved to {output_path}: {len(combined)} rows")


if __name__ == "__main__":
    build_augmented_dataset()