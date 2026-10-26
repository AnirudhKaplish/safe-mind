# SAFE-MIND

**A Blockchain-Governed, Explainable AI Framework for Privacy-Preserving Depression Risk Screening and Clinician Referral**

SAFE-MIND screens a person's own social media posts for signs of depression, but only with their permission. A smart contract records consent and blocks processing the moment it is revoked. Two models, a TF-IDF + Logistic Regression baseline and a fine-tuned BERT, must agree before a result is given. Every result shows the words that drove it. Results appear on a clinician dashboard, on a simulated robot digital twin and on a user check-in page.

> **This is a research prototype and a screening aid. It is not a medical diagnosis.** If you or someone you know needs support in India, Tele-MANAS is free and open 24/7: **14416** or **1800-891-4416**.

Project of VIT Chennai (School of Computer Science and Engineering). Guide: Dr. Jayanthi R.
Team: Anirudh Kaplish (23BCE1907), Archita Gupta (23BCE1902).

---

## How it works

```
 User page (/user)          Meridian clinician dashboard (/)
        |                               |
        +---------------+---------------+
                        v
        Flask app on port 5000  (src/dashboard)
        pre-filter -> baseline + BERT -> ensemble -> explanation
              |                                   |
              | HTTP, port 8546                   | robot bus (read-only API)
              v                                   v
   Node consent bridge                   Robot digital twin
   ganache + ConsentManager contract     GET /api/robot/state
```

For each post the system does the following:

1. **Pre-filter.** Skips questions, statements about other people, general statistics and anything under 6 words. These are texts the models were not trained for.
2. **Consent check.** Asks the `ConsentManager` smart contract whether permission is valid. If it was revoked, nothing is processed.
3. **Two models.** A baseline (TF-IDF, VADER sentiment, linguistic features and a 40-phrase lexicon, with Logistic Regression) and a fine-tuned `bert-base-uncased`.
4. **Ensemble rule.** If both agree, that label is returned with the averaged confidence. If they disagree, the case is marked NEEDS_REVIEW and a person decides.
5. **Explanation.** Exact word contributions for the baseline and LIME for both models. Red words pushed towards depression and green words pushed away.
6. **Report.** The result goes to the dashboard and the robot twin.

For many posts (a person report), the overall result is **red** when at least 2 posts are flagged and they are at least 25% of the checked posts, or when any post mentions disappearing. It is **amber** when at least 1 post is flagged or the models disagree on at least 2, and **green** otherwise.

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression (threshold 0.40) | 96.2% | Recall on depression rises from 0.928 to 0.947 when the threshold is lowered from 0.50 to 0.40 |
| Random Forest | 96.7% | Compared, not used in the app |
| Fine-tuned BERT | about 98.0% (F1 0.980) | 3 epochs on a Colab T4 GPU |

- Data: Kaggle "Depression: Reddit Dataset (Cleaned)", 7,649 posts, split 80/20 (1,913 test posts).
- A counterfactual-augmented set (16 extra rows, 7,665 in total) fixed a keyword shortcut in BERT: hard cases improved from 3/5 to 5/5.
- Edge-case tests: 10/10. Personal-disclosure examples accepted: 7/7. Robot digital twin tests: 110.
- Test accuracy on this dataset is an upper bound. Reddit depression posts are written very differently from everyday posts, so real-world performance will be lower. Labels describe a community, not a diagnosis.

## Repository layout

```
src/            screening engine (features, models, explanation, reports) and the dashboard
src/dashboard/  app.py (Meridian), user_api.py + user_app.html (user check-in page)
blockchain/     Solidity contract, compile and demo scripts, consent_server.js bridge
robot_sim/      robot digital twin (simulation)
data/           samples (fictional demo people); raw datasets are not included
results/        trained baseline files; the BERT model is not included (see below)
tests/          automated tests
```

## Setup (Windows)

**You need:** Python 3.11, Node.js 22 and Git.

```
git clone https://github.com/AnirudhKaplish/safe-mind.git
cd safe-mind
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-robot.txt
```

**Blockchain packages** (once):

```
cd blockchain
npm install ganache ethers solc
node scripts\compile.js
cd ..
```

**Dataset.** Download "Depression: Reddit Dataset (Cleaned)" from Kaggle and save it as `data\seed_dataset.csv`. It is needed only for retraining, not for running the demo.

**BERT model.** The model file is about 440 MB, too large for GitHub. Download the zip from **[BERT model (Google Drive)](https://drive.google.com/file/d/1_9C_LIgCCVWFXESA5apHwRwYu-CYe9gj/view?usp=sharing)**, unzip it, and place the files directly in `results\bert_model\` (`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`). Without it, the app falls back to a baseline stand-in and says so.

## Run

**Everything at once (blockchain bridge, dashboard, robot):**

```
run_user_app.bat
```

Then open:

- `http://127.0.0.1:5000/user` : user check-in page (permission, upload posts, live results, report)
- `http://127.0.0.1:5000` : Meridian clinician dashboard (single post and person report)

The Engine, Blockchain and Robot chips at the top of the user page should all be green. In the robot window, `q` stops the robot and the dashboard.

**Other commands:**

```
python -m robot_sim --manual          robot only (keys 1-9 show each screen, 0 auto-cycles)
python test_my_own_sentences.py       try your own sentences
cd blockchain && node scripts\deploy_and_demo.js     consent lifecycle demo
```

Linux and macOS: use `run_robot_demo.sh` and `run_all.sh`.

## Demo data

`data/samples/` has fictional people for trying the person report: `demo_person_struggling_posts.csv` with `demo_person_struggling_comments.csv` (red, with the urgent note) and `demo_person_doing_well_posts.csv` (green). To use your own data, request your Reddit data export at reddit.com/settings/data-request and upload `posts.csv` and `comments.csv`.

We deliberately do **not** collect posts by username. That would be profiling without consent and would go against the consent design.

## Known limitations

- Labels describe the community of origin, not a clinical diagnosis. Clinical validation has not been done.
- Binary scope (depression or not), English text from one platform.
- Warning themes come from a fixed phrase list. They are not learned.
- BERT can over-trust the words "I am". A minimum-length rule (6 words) is a patch. The proper fix is retraining with more counterfactual data.
- The question filter can wrongly skip a personal post.
- BERT's calibration figures in the paper come from the original model file and need recalibrating for the retrained model (`src/models/calibrate_bert.py`).
- The blockchain is a local test chain that resets when restarted. The robot is a simulation.
- "Log in with Reddit" is future work.

## Credits

- Dataset: Kaggle, "Depression: Reddit Dataset (Cleaned)"
- Libraries: scikit-learn, Hugging Face Transformers, PyTorch, LIME, SHAP, VADER, Flask, ganache, ethers.js

## Cite

> A. Kaplish, A. Gupta and R. Jayanthi, "SAFE-MIND: A Blockchain Governed, Explainable AI Framework for Privacy Preserving Depression Risk Screening and Clinician Referral," VIT Chennai, 2026.
