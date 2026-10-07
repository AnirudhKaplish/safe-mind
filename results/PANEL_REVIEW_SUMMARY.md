# Panel Review III — Implementation & Results Summary
## Interpretable Mental Health Detection from Social Media

**Dataset: REAL DATA** — Kaggle "Depression: Reddit Dataset (Cleaned)"
(`kaggle.com/datasets/infamouscoder/depression-reddit-cleaned`),
7,649 real Reddit posts after cleaning/deduplication, binary labels
(`depression` / `not_depression`), roughly balanced (3,760 / 3,889).

This document maps what's built directly to the four review criteria.
All results below come from actually running the code in `mhd_project/`
on this real dataset (see `README.md` for exact run commands) —
nothing here is projected, estimated, or run on synthetic data.

**Note on scope narrowing:** the real dataset's ground-truth labels are
binary (depression / not-depression) — it does not contain separate
anxiety/stress annotations. Rather than inventing labels the data
doesn't actually support, the classification task was narrowed to
match what's genuinely annotated. This is worth stating to the panel
proactively — it shows the scope decision was made deliberately, not
because the harder version wasn't attempted.

---

## 1. Implementation (working modules, ~half of approved scope)

Six of the approved design's modules are fully built and executable
end-to-end, chained together (text in → risk class + explanation →
consent-gated report out), now running on real data:

1. **Data pipeline** — 7,649 real, cleaned Reddit posts
2. **Preprocessing** — URL/emoji removal, normalization, tokenization,
   stop-word removal with negation preserved
3. **Feature extraction** — TF-IDF, VADER sentiment, linguistic markers
4. **Baseline classification** — Logistic Regression + Random Forest
5. **Explainable AI** — SHAP (LinearExplainer) + LIME (LimeTextExplainer)
6. **Blockchain dynamic consent** — a real Solidity smart contract,
   compiled and deployed to a local chain
7. **Report generation** — combines all of the above, with consent
   enforcement built in

Two modules are explicitly scoped as next-milestone work, not claimed
as done: BERT/RoBERTa fine-tuning (script is written and ready — see
`src/models/finetune_transformer.py` — run it on the same real CSV,
just needs a machine with internet/GPU access to pull pretrained
weights) and the doctor/therapist dashboard UI.

The physical ESP32 hardware firmware (`hardware/mhd_hardware.ino`) is
written and matches the approved hardware/response tables exactly, but
hasn't been flashed to a physical board yet — a virtual device console
(`hardware/virtual_device.py`) demonstrates the identical response
logic end-to-end without needing the board in the room.

---

## 2. Technical Accuracy

- **Preprocessing correctly preserves negation words** ("can't",
  "never") that a generic stop-word list would otherwise strip —
  important because negation flips clinical meaning in this domain.
- **Train/test split is stratified by label.** Each row here is an
  independent post (not augmented variants of the same post, as the
  earlier seed data was), so ordinary stratified splitting is
  appropriate; the code still documents that a *user/author*-level
  split would be the correct approach if the real dataset ever
  includes multiple posts per author.
- **SHAP is run against a proper background sample** drawn from the
  training set — not against the instance being explained itself
  (a real bug caught and fixed earlier: explaining a point against
  itself as its own background trivially zeroes out every
  contribution).
- **The blockchain component is a real deployed smart contract**
  (`ConsentManager.sol`), not a mocked log — compiled with the `solc`
  npm package and deployed to a local Ethereum-compatible chain via
  `ganache`, producing real transaction hashes and a real revert when
  an unauthorized processing attempt is made post-revocation.
- **The report generator enforces consent before generating a report**
  (a second bug caught and fixed: it was generating reports even
  when consent had been revoked) — it now correctly returns a
  `BLOCKED` status instead, matching "processing allowed only when
  valid consent exists."
- **A real pandas/CSV edge case was caught and fixed**: some cleaned
  posts became empty strings after stop-word removal, and pandas
  silently reads empty CSV fields back as `NaN` on reload — this broke
  the SHAP background sampler until explicitly handled
  (`fillna("").astype(str)`).

---

## 3. Results Obtained So Far (on the real dataset)

### Baseline model comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 96.2% | 96.5% | 96.2% | 96.2% |
| Random Forest | 96.6% | 96.8% | 96.5% | 96.6% |

See `results/baseline_comparison.png` for the visual comparison and
`results/confusion_matrix_*.png` for per-class breakdowns. The Random
Forest confusion matrix shows 879/940 depression posts correctly
caught (61 missed) and 969/973 not-depression posts correctly caught
(4 false alarms) — realistic, credible numbers with visible errors,
not a suspiciously perfect diagonal.

### Explainability example — a genuinely useful real case

Input: *"I don't even know why I'm posting this. Nothing feels worth
doing anymore, I just lie in bed most of the day."*
→ Model predicted: **not_depression** (only 66.9% confidence — a
weak, borderline call)
→ LIME shows most contributing words ("anymore", "worth", "Nothing",
"I") actually pushed *against* "not_depression" — i.e., toward
depression — but a couple of neutral words ("bed", "why") tipped the
final label the other way.

**This is a strong, real talking point, not a weakness to hide:** this
is exactly the scenario explainability exists for — the confidence
score (67%, not 99%) and the word-level breakdown both signal
"borderline, needs a second look," which is precisely the kind of
evidence a doctor reviewing the report would want before trusting a
low-confidence "not depression" call. See `results/lime_example_1.png`.

### Blockchain consent lifecycle (real transactions)

| Step | Result |
|---|---|
| Grant consent | ✅ Success, block confirmed |
| Check before processing | ✅ Allowed |
| Modify consent | ✅ Success |
| Check again | ✅ Still allowed |
| Revoke consent | ✅ Success |
| Check after revocation | ✅ **Reverted as expected** |

Full log with complete transaction hashes:
`results/blockchain_consent_demo_log.json`

### Report generation — both enforcement paths demonstrated

- `results/sample_assessment_report_allowed.json` — consent granted →
  full report generated (risk class, confidence, SHAP/LIME evidence)
- `results/sample_assessment_report_blocked.json` — consent revoked →
  report generation correctly refuses and returns `BLOCKED`

---

## 4. Presentation and Clarity — talking points for likely questions

- **"Are these real results?"** → Yes — 7,649 real Reddit posts from
  a published Kaggle dataset, not synthetic data. Say this early;
  don't wait to be asked.
- **"Why binary and not the original 4-class (depression/anxiety/
  stress/normal) design?"** → The real, publicly available labeled
  dataset only has binary depression/not-depression ground truth. We
  chose to report accurate results on real labels rather than
  fabricate anxiety/stress categories with no ground truth to
  validate against. Multi-class detection remains a next-step goal if
  a suitably labeled dataset (e.g. Dreaddit for stress) is added.
- **"96% seems high — is this legitimate?"** → Yes, and unlike an
  earlier synthetic-data test run (which hit an inflated 100% on
  templated data), this has visible, realistic errors in the
  confusion matrix (61 missed depression cases, 4 false alarms) — a
  healthy sign the number is real, not memorized.
- **"Is the blockchain real or simulated?"** → Real — a compiled,
  deployed Solidity contract on a local chain, with real transaction
  hashes you can show directly from `blockchain_consent_demo_log.json`.
- **"What's left for the remaining half of scope?"** → BERT/RoBERTa
  fine-tuning on this same real CSV (script ready, needs a GPU/
  internet run), the doctor dashboard UI, and flashing the firmware to
  the physical ESP32 board.
- **"What bugs did you actually catch while building this?"** → Three
  concrete ones, all fixed: (1) SHAP was explaining each post against
  itself as its own background, zeroing every contribution; (2) the
  report generator was ignoring its own consent check and generating
  reports even when consent had been revoked; (3) empty cleaned posts
  were silently becoming `NaN` on CSV reload, breaking the SHAP
  background sampler on the real (larger, messier) dataset. Mentioning
  these proactively is a strong signal of genuine, hands-on
  implementation rather than copied code.
