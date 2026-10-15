"""
build_seed_dataset.py
======================
Builds a small, labeled, Reddit-post-style seed dataset for the
Mental Health Detection pipeline.

WHY THIS EXISTS:
The training sandbox this was built in cannot reach Kaggle or
HuggingFace to pull the real dataset (network is restricted to
package registries only). So this script generates a realistic
SEED dataset (posts written in the style of r/depression,
r/Anxiety, r/stress, and neutral subreddits) so the rest of the
pipeline (preprocessing -> features -> models -> XAI -> report)
is fully runnable and produces real metrics today.

TO SWITCH TO THE REAL DATASET LATER:
Download either of:
  - Kaggle: "Depression: Reddit Dataset (Cleaned)"
    kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
  - Dreaddit (Turcan & McKeown, 2019) - stress dataset
Both have (or can be mapped to) a `text` + `label` column.
Save it as data/raw_dataset.csv and re-run this pipeline -
nothing else needs to change.

Labels used: normal, depression, anxiety, stress
"""

import pandas as pd
import random
import os

random.seed(42)

# ---------------------------------------------------------------
# Seed sentence banks per class. These are written to mimic the
# tone/vocabulary of real Reddit mental-health posts based on
# published examples in the literature (Dreaddit, CLPsych, etc.)
# with realistic hedging, first-person narration, and platform-
# style phrasing (line breaks, informal punctuation).
# ---------------------------------------------------------------

depression_posts = [
    "I don't even know why I'm posting this. Nothing feels worth doing anymore, I just lie in bed most of the day.",
    "I've been feeling numb for weeks now. Everything I used to enjoy just feels grey and pointless.",
    "Some days I can't even get up to eat. I know I should but I just don't have it in me.",
    "I keep telling everyone I'm fine but honestly I feel like a burden to everyone around me.",
    "I've lost interest in literally everything, my hobbies, my friends, even food doesn't taste like anything.",
    "I cried in the shower again this morning for no reason. I'm just so tired of feeling like this.",
    "Nothing matters anymore. I go through the motions of my day but inside I feel completely empty.",
    "I can't remember the last time I felt genuinely happy about anything, it's just been months of grey days.",
    "I isolated myself from all my friends this month. I just don't have the energy to pretend I'm okay.",
    "Every night I lie awake thinking about how tired I am of existing, not wanting to die, just tired.",
    "I used to love painting but I haven't touched my brushes in three months, I just don't care anymore.",
    "I feel like I'm a disappointment to my family no matter what I do, nothing I do is ever good enough.",
    "I've stopped replying to texts because I just don't have the energy to hold a conversation anymore.",
    "Waking up feels like the hardest part of my day, I just want to stay under the blanket forever.",
    "I feel so hopeless about the future, like nothing is ever going to get better for me.",
]

anxiety_posts = [
    "My heart has been racing all day and I can't shake this feeling that something bad is about to happen.",
    "I keep overthinking every conversation I had today, replaying it over and over in my head.",
    "I had a panic attack before my presentation and I couldn't breathe properly for almost ten minutes.",
    "I'm constantly on edge lately, jumping at every notification like it's bad news.",
    "I can't stop worrying about my exam tomorrow even though I've studied more than enough.",
    "My hands were shaking the whole meeting and I felt like everyone could tell I was panicking inside.",
    "I keep checking my phone every two minutes waiting for a reply, my chest feels tight the whole time.",
    "I've been getting these sudden waves of dread out of nowhere, my stomach just drops.",
    "I couldn't sleep at all last night because my mind wouldn't stop racing about tomorrow.",
    "Every time my phone rings unexpectedly I feel a jolt of fear before I even answer it.",
    "I avoided the party because just thinking about being around that many people made me feel sick.",
    "I keep imagining worst case scenarios about my health even though the doctor said I'm fine.",
    "My thoughts are spiraling again, I can't focus on anything because I'm so anxious about the interview.",
    "I feel like I'm constantly bracing for something to go wrong, I can't relax even at home.",
    "I get this tight feeling in my chest whenever I have to speak up in a group, it's exhausting.",
]

stress_posts = [
    "Work has been absolutely brutal this month, I'm juggling three deadlines and barely sleeping.",
    "I feel completely overwhelmed trying to balance my job, my classes, and taking care of my family.",
    "My boss keeps piling on more tasks and I genuinely don't know how I'm going to get it all done.",
    "I've been running on four hours of sleep all week trying to keep up with everything on my plate.",
    "Between rent going up and my hours getting cut I'm stressed out of my mind about money right now.",
    "I have three assignments due this week and I haven't even started the first one, I'm panicking.",
    "Juggling my internship and finals has left me so burnt out I can barely think straight.",
    "My inbox has 200 unread emails and I feel like I'm drowning in work I can't keep up with.",
    "I snapped at my roommate today over something small, I think the pressure at work is getting to me.",
    "I've got back to back meetings all week and no time to actually finish the work they're about.",
    "The constant pressure from my manager to hit targets is wearing me down day after day.",
    "I feel stretched too thin between caregiving for my parent and holding down my full time job.",
    "I keep getting tension headaches from staring at spreadsheets all day trying to hit this deadline.",
    "Moving apartments while starting a new job at the same time has completely drained me.",
    "I haven't had a proper day off in a month and I can feel myself getting close to burning out.",
]

normal_posts = [
    "Had a really nice weekend hiking with some friends, the weather was perfect for it.",
    "Finally finished setting up my new desk and it's made working from home so much nicer.",
    "Tried a new recipe today, turned out way better than I expected honestly.",
    "Been getting back into reading lately, just finished a really good mystery novel.",
    "Went to the gym this morning and actually hit a new personal best on my squats.",
    "Caught up with an old friend over coffee today, it was really good to reconnect.",
    "Started learning guitar again after a long break, it's fun to be picking it back up.",
    "Spent the afternoon in the garden, the tomatoes are finally starting to come in nicely.",
    "Watched a great documentary last night about ocean life, really well made.",
    "Got a lot done at work today and actually left on time for once, feels good.",
    "My dog learned a new trick today and I'm honestly way too proud of her for it.",
    "Tried out a new coffee shop downtown, the espresso was really solid.",
    "Been enjoying the cooler weather lately, perfect for evening walks around the neighborhood.",
    "Finished a big project at work today, feeling pretty accomplished about how it turned out.",
    "Spent the day organizing my bookshelf and found a few books I forgot I owned.",
]

def expand(posts, label, n_variants=4):
    """Create light variations of each seed post to grow the dataset
    while keeping realistic sentence structure (simple prefix/suffix
    variation, not paraphrase-by-model, to avoid injecting bias)."""
    prefixes = ["", "So, ", "Honestly, ", "Update: ", "Not sure why I'm typing this but, "]
    suffixes = ["", " Anyway.", " Just needed to get that out.", " Not sure what to do.", " Thanks for reading if you did."]
    rows = []
    for p in posts:
        for i in range(n_variants):
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            text = f"{prefix}{p}{suffix}".strip()
            rows.append({"text": text, "label": label})
    return rows

rows = []
rows += expand(depression_posts, "depression", n_variants=5)
rows += expand(anxiety_posts, "anxiety", n_variants=5)
rows += expand(stress_posts, "stress", n_variants=5)
rows += expand(normal_posts, "normal", n_variants=5)

df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=42).reset_index(drop=True)

import argparse

parser = argparse.ArgumentParser(description="Build the SYNTHETIC 4-class seed dataset.")
parser.add_argument("--force", action="store_true",
                    help="overwrite data/seed_dataset.csv even if it already exists")
args = parser.parse_args()

# Repo-relative output path (was a hard-coded sandbox path).
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_dataset.csv")

print("=" * 66)
print("WARNING: SYNTHETIC DATA. This builds the hand-written 4-class seed")
print("set (normal/depression/anxiety/stress), NOT the real Kaggle dataset.")
print("For real results run: python data/prepare_real_dataset.py")
print("=" * 66)

if os.path.exists(out_path) and not args.force:
    print(f"NOTE: {out_path} already exists (likely your real dataset).")
    print("Refusing to overwrite it with synthetic seed data. Re-run with --force to overwrite.")
    raise SystemExit(0)

df.to_csv(out_path, index=False)

print(f"SYNTHETIC seed dataset built: {len(df)} rows")
print(df["label"].value_counts())
print(f"Saved to {out_path}")
