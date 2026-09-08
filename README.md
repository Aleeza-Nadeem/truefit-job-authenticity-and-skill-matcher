# TrueFit — Job Authenticity & Skill Match Analyzer

A machine learning project that combines **supervised** and **unsupervised** learning to help job seekers answer two questions when they encounter a job posting:

1. **Is this posting likely real or fake?** (Fraud/Authenticity Detection — Supervised)
2. **How well do my skills match this posting, and what's missing?** (Skill Clustering & Matching — Unsupervised)

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Who This Helps](#who-this-helps)
- [Architecture Overview](#architecture-overview)
- [Pipeline A: Fraud Detection (Supervised)](#pipeline-a-fraud-detection-supervised)
- [Pipeline B: Skill Clustering (Unsupervised)](#pipeline-b-skill-clustering-unsupervised)
- [Datasets Used](#datasets-used)
- [Key Decisions & Lessons Learned](#key-decisions--lessons-learned)
- [Known Limitations](#known-limitations)
- [Project Structure / Files](#project-structure--files)
- [How the App Works](#how-the-app-works)
- [Future Improvements](#future-improvements)
- [Honesty Note](#honesty-note)

---

## Project Motivation

While job hunting, two recurring frustrations stood out:

1. Not knowing whether a job posting is **authentic** — many postings are vague, scammy, or misleading.
2. Not knowing whether a posting's requirements **actually match** a person's own skills, or what's missing to qualify.

Rather than treating this as one problem, it was split into two well-defined ML problems that combine into a single tool:

> A user pastes a job posting's details → the app tells them **(1) how likely it is to be real**, and **(2) how well its required skills match their own profile**, including which skills they're missing.

This project intentionally uses **both halves of machine learning** — supervised classification where labeled ground truth exists (fraud/not-fraud), and unsupervised clustering where no labels exist (grouping jobs by skill similarity).

---

## Who This Helps

- **Job seekers** — get a market-intelligence style signal instead of guessing which postings are trustworthy or worth applying to.
- **Career switchers** — see which skill category they're closest to and what's missing to fully qualify.
- **Recruiters / reviewers of this project** — see evidence of end-to-end data science skills: messy real-world data wrangling, hypothesis-driven EDA, feature engineering, model tuning, and honest interpretation of results (including when hypotheses fail).

**Important scope honesty:** This tool is a *market signal / risk-flagging tool*, not a certainty machine and not personalized career advice. It does not know a user's interests, aptitude, or what skills should be learned in what order — it only reports what the data shows.

---

## Architecture Overview

```
User pastes a job posting
        │
        ├──> Pipeline A (Fraud Detection — Supervised)
        │     Input: title, description, salary, logo, screening Qs, etc.
        │     Output: Real / Fake probability (threshold = 0.2)
        │
        └──> Pipeline B (Skill Clustering — Unsupervised)
              Input: user's own skills + posting's required skills
              Output: Best-fit job cluster, % skill match, missing skills
```

Both pipelines run independently on the same posting and their results are shown together.

---

## Pipeline A: Fraud Detection (Supervised)

### Why Supervised
The fraud dataset has a ground-truth label (`fraudulent`: 0/1), so this is a **classification problem**, not clustering — an important distinction made explicit during the project (it would be inaccurate to call this part "unsupervised").

### Dataset
**"[Real or Fake] Fake Job Description Prediction"** by shivamb (Kaggle) — based on the Employment Scam Aegean Dataset (EMSCAD) from the University of the Aegean.
- 17,880 job postings total
- 866 fraudulent (~4.8%) → **heavily imbalanced target**
- 18 columns: `job_id, title, location, department, salary_range, company_profile, description, requirements, benefits, telecommuting, has_company_logo, has_questions, employment_type, required_experience, required_education, industry, function, fraudulent`

### Preprocessing Principles
- **Never fill missing values with mean/mode** — for this dataset, missingness itself is often a real signal (legitimate companies tend to fill in more fields; scam postings often skip them).
- Missing categoricals → filled with `"Unknown"` (preserves the missingness signal instead of erasing it).
- `salary_range` (text like `"$50,000-$60,000"`) parsed into numeric `salary_min` / `salary_max`, plus a `has_salary` binary flag.

### Hypothesis-Driven EDA (the core methodology)
Instead of guessing features, each hypothesis was **statistically tested** against the real data using group comparisons and chi-square significance tests — not just eyeballed percentages. Final results:

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| A | Remote + generic title (e.g. "data entry", "virtual assistant") | Fraud rate 4.7% → 53.8% (11x) | ✅ Strong signal |
| B | `missing_all_three` (no logo + no profile + no salary) | 22.7% fraud rate, p<0.001 | ✅ Strong signal |
| C | High salary for entry-level role | 4.8% → 15% fraud rate | ✅ Moderate signal |
| D | Off-platform contact (Telegram/WhatsApp/personal email) | 0% fraud rate on 102 matches | ❌ Rejected (no signal in this dataset) |
| 5 | Duplicate/copy-pasted descriptions | 4.0% → 7.4% fraud rate | ⚠️ Weak but real |
| 6 | Technical title + missing education field | Fraud rate *lower* (2.5% vs 5.2%), p<0.001 | ❌ Rejected — real pattern, reversed direction |
| 7 | Short description length | No meaningful difference (p=0.93) | ❌ No effect |
| 8 | Technical title but non-technical `function` field | Fraud rate *lower* (1.4% vs 4.9%), small sample (n=71) | ❌ Rejected — reversed & unreliable |
| — | `is_shadow_company` (long profile + missing logo) | p=0.27 | ❌ Inconclusive — not statistically significant |
| — | `lacks_screening` (no screening questions) | Strong signal, p<0.001 | ✅ Strong signal |

**Key lesson embedded in the process:** a lower fraud-rate percentage on a small sample (e.g. 71 or 276 rows) is not automatically a "real" finding — chi-square testing was used to separate genuine (if surprising) patterns from statistical noise before trusting any result.

### Final Engineered Features Used
- `is_high_risk_title_combo`
- `missing_all_three`
- `high_salary_for_entry`
- `is_duplicate_desc`
- `lacks_screening`
- `technical_no_education`
- Raw binary columns: `has_company_logo`, `telecommuting`, `has_questions`
- TF-IDF vectors (max 300 features) on combined text (`title + company_profile + description + requirements + benefits`, HTML-stripped, lowercased, cleaned)
- Low-cardinality categoricals (`employment_type`, `required_experience`, `required_education`) → one-hot encoded
- High-cardinality categoricals (`industry`, `function`) → frequency encoded
- `department` → reduced to a binary `has_department` flag rather than one-hot encoding names (65% missing)

### Modeling
- **Train/test split** done *before* fitting TF-IDF, to avoid data leakage (TF-IDF fit on train only).
- Two baseline models compared: **Logistic Regression** and **Random Forest**, both with `class_weight='balanced'` to address the ~5% fraud imbalance.
- Evaluated with **precision, recall, F1, ROC-AUC, and confusion matrix** — never plain accuracy, since a lazy "always predict real" model would already score ~95% accuracy while being useless.

**Baseline results (threshold = 0.5):**

| Model | Precision (fraud) | Recall (fraud) | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.13 | 0.60 | 0.22 | 0.727 |
| Random Forest | 0.96 | 0.47 | 0.64 | 0.991 |

**Threshold tuning** (adjusting the 0.5 cutoff on `predict_proba`) was used to rebalance precision vs. recall, since:
- **False alarm** (real posting flagged as fraud) → annoying, user might miss a legitimate opportunity.
- **Missed fraud** (fake posting flagged as real) → genuinely harmful, higher-stakes mistake.

Because this is a safety-relevant tool, **recall was prioritized without destroying precision** — the classic asymmetric-cost tradeoff.

**Hyperparameter tuning via `GridSearchCV`** (with `StratifiedKFold`, scoring on F1) was also tried — an important, honest finding was that **the tuned model did not beat the original untuned model once both were threshold-optimized**:

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Untuned RF | 0.5 | 0.96 | 0.47 | 0.64 |
| **Untuned RF** | **0.2** | **0.76** | **0.84** | **0.80** ← **Final model** |
| Tuned RF | 0.5 | 0.77 | 0.61 | 0.68 |
| Tuned RF | 0.3 | 0.55 | 0.89 | 0.68 |

**Final locked-in model: untuned Random Forest, decision threshold = 0.2** (Precision 0.76, Recall 0.84, F1 0.80). This was a deliberate, data-driven decision to prefer the simpler model after tuning failed to beat it — documented here as a legitimate finding, not an error.

> **Important mechanic to remember:** the threshold is applied *after* training, only at the final probability→label step (`(proba >= 0.2).astype(int)`). It does not require retraining the model.

---

## Pipeline B: Skill Clustering (Unsupervised)

### Why Unsupervised
There is no ground-truth "correct job category" label for postings — categories are discovered purely from which skills co-occur, which is the core definition of clustering.

### Datasets Used
From the **"LinkedIn Job Postings (2023-2024)"** dataset by arshkon (Kaggle):
- `postings.csv` (~124k rows, 31 columns — job details, salary, work type, location, timestamps)
- `job_skills.csv` (job_id ↔ skill code mapping)
- `skills.csv` (skill code ↔ skill name lookup, 35 categories)
- `companies.csv` (optional enrichment — company size code 1–7, country; parked as a v2 feature, not used in the core pipeline)

### Real-World Data Wrangling Challenges (and fixes)
This dataset caused significant, realistic engineering friction, which is documented here because it reflects real data science work:

- **CSV parsing failures**: `postings.csv` contains job descriptions with embedded newlines and unescaped quote characters, which broke the default C-engine parser (`EOF inside string` errors) and caused massive row-count discrepancies across different parsing attempts (from ~22k to ~1.3M "rows" depending on engine/flags).
- **Root cause diagnosis**: eventually traced back to an **incomplete file upload**, not a genuine parsing bug — confirmed by loading a properly-synced copy via Google Drive, which parsed cleanly with default `pd.read_csv()` at the correct ~123,849 rows.
- **Colab runtime disconnects** wiped in-memory variables multiple times, causing silent reuse of stale/wrong DataFrames during merges (diagnosed via job_id set-overlap checks). **Fix adopted going forward**: mount Google Drive and save intermediate cleaned files immediately, so a disconnect never costs full pipeline re-runs.
- **Salary outliers**: IQR-based outlier detection found ~952 out of 36,073 non-null `normalized_salary` values (~2.6%) that were obvious data-entry errors (e.g., "$286,000,000/year" for a Cloud Domain Architect). These were **nullified, not dropped** — the salary value was discarded but the row (with its valid title/skills) was kept, since the error was isolated to one column.

### Feature Scoping
Reduced from 31 raw columns down to the ones actually needed:
`job_id, title, formatted_work_type, remote_allowed, normalized_salary, formatted_experience_level, location, listed_time, skills_combined`

### Skill Aggregation
`job_skills.csv` + `skills.csv` were merged and grouped by `job_id` so each posting has one combined skill string (e.g. `"Sales, Business Development"`), then inner-joined to `postings.csv` (correctly dropping ~8% of postings that had no skill listed — those can't be clustered by skill anyway).

Final merged dataset: **109,392 rows**.

### Clustering Approach
- **TF-IDF vectorization** on `skills_combined` (small, dense vocabulary — only 35 possible skill categories, 1–3 per posting).
- **K-Means** with `k` selected using both the **elbow method (inertia)** and **silhouette score** (computed on a fixed 5,000-row sample for speed) — but critically, **the highest silhouette score was deliberately not chosen blindly**. It kept climbing toward `k=19` because with only 35 base categories, more clusters mechanically approach "one cluster per unique skill-string," which fragments meaning rather than adding it.
- **Final decision: k=10**, chosen by manually inspecting the actual top skill combinations inside clusters at k=8, k=10, and k=12 — k=8 merged categories that should be separate, k=12 fragmented categories that should stay together, k=10 was the interpretable middle ground.

### Final Clusters (k=10)

| Cluster | Size | Label |
|---|---|---|
| 0 | 14,947 | Healthcare |
| 1 | 15,223 | Manufacturing & Operations Management |
| 2 | 9,729 | Engineering & IT |
| 3 | 11,661 | Sales & Business Development |
| 4 | 9,257 | Other / Unclassified (employer didn't specify a detailed category) |
| 5 | 6,926 | Finance & Sales |
| 6 | 5,339 | Administrative & Legal |
| 7 | 4,401 | Accounting & Finance |
| 8 | 17,443 | HR, Education & Quality Assurance |
| 9 | 14,466 | Information Technology & Digital |

Clusters are reasonably balanced (~4,400–17,400 rows each), with the expected minor conceptual overlap between "Engineering & IT" and "Information Technology & Digital" (a genuine real-world category fuzziness, not a modeling error).

### User Matching & Skill Gap
- A user's typed skills are transformed with the **same fitted TF-IDF vectorizer**, then compared to each cluster's centroid via **cosine similarity** to find the best-fit cluster + confidence score.
- A separate function computes **missing skills**: the top skills of the matched cluster minus the skills the user already listed.

### Cross-Tabs (the "market snapshot" insight layer)
Since salary, work type, remote status, and experience level are all *downstream effects* of what skills a role requires (not independent columns), they were cross-tabbed against the discovered clusters:
- Average/median salary by cluster
- % remote-allowed by cluster
- Work type distribution (full-time/contract/etc.) by cluster

---

## Datasets Used

| Dataset | Source | Used For |
|---|---|---|
| Real or Fake Job Postings (EMSCAD) | Kaggle — shivamb | Pipeline A (fraud detection) |
| LinkedIn Job Postings 2023–2024 (`postings.csv`, `job_skills.csv`, `skills.csv`) | Kaggle — arshkon | Pipeline B (skill clustering) |
| `companies.csv` (same LinkedIn dataset) | Kaggle — arshkon | Optional/parked enrichment (company size, country) |

---

## Key Decisions & Lessons Learned

- **Missingness ≠ noise, always check first**: in the fraud dataset, missing fields were informative and preserved (`"Unknown"` category, presence/absence flags) rather than imputed away. In the skill dataset, missingness was mostly just inconsistent form usage, not a meaningful signal — the same instinct was correctly *not* over-applied there.
- **Statistical testing over eyeballing percentages**: several hypotheses that looked directionally promising on raw percentages were confirmed or rejected using chi-square significance testing and sample-size awareness, avoiding false confidence from small groups (e.g., n=71, n=276).
- **A rejected hypothesis is still a valid, useful result** — e.g., "technical roles missing an education field" turned out to correlate with *lower* fraud, not higher, which is a legitimate, real finding worth reporting rather than hiding.
- **Model tuning is not always an improvement** — GridSearchCV's best model (optimized for F1 at the default 0.5 threshold) underperformed the simpler untuned model once both were properly threshold-tuned. The simpler, better-performing model was chosen deliberately.
- **Clustering `k` should be chosen for interpretability, not just for the highest silhouette score** — a metric that keeps climbing isn't automatically "better" if it means clusters stop being human-meaningful.
- **Diagnose root causes, not symptoms**: repeated CSV parsing failures were eventually traced to an incomplete upload rather than a parser configuration problem — switching parsing engines wasn't the fix; re-uploading correctly was.
- **Cost-asymmetric errors matter**: for a fraud-detection tool, a missed scam is worse than a false alarm, which directly informed the decision to prioritize recall over raw precision via threshold tuning.

---

## Known Limitations

- **No time-trend analysis is possible with the current skill-clustering dataset.** 109,391 of 109,392 postings are from a single month (April 2024) — it's a snapshot, not a historical archive. The original idea of showing "which skill clusters are growing/shrinking" had to be honestly dropped from this version of the project; the LinkedIn dataset does not support it.
- The fraud dataset (EMSCAD) is from 2012–2014 — scam tactics and today's job market conditions (e.g., AI-driven hiring processes) are not reflected in it. Feature validity should be understood as historically grounded, not necessarily current.
- Flagging a posting as "unusual" is a **risk signal, not a fraud verdict** — legitimate postings (e.g., from small startups) can also look unusual by these metrics. The app should be framed honestly as a flagging/signal tool, not a certainty machine.
- `company_size` from `companies.csv` is a coded LinkedIn size bucket (1 = 1–10 employees ... 7 = 5,001–10,000+), not a literal headcount — not currently used in the core pipeline.
- Skill categories are coarse (only 35 possible tags, 1–3 per posting), so clusters represent broad fields rather than fine-grained job roles.

---

## Project Structure / Files

```
TrueFit_App/
│
├── app.py                          # Streamlit app tying both pipelines together
├── requirements.txt                # Python dependencies
│
├── fraud_model_rf.pkl              # Trained Random Forest fraud classifier
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer fit on fraud dataset text
├── feature_cols.pkl                # Ordered structured feature column list
├── industry_freq_map.pkl           # Frequency encoding map for `industry`
├── function_freq_map.pkl           # Frequency encoding map for `function`
├── onehot_cols.pkl                 # One-hot column names used at train time
├── entry_75th_salary.pkl           # 75th-percentile salary threshold (entry-level)
│
├── skill_cluster_model.pkl         # Trained K-Means model (k=10)
├── skill_tfidf_vectorizer.pkl      # TF-IDF vectorizer fit on skills_combined
├── cluster_labels.pkl              # Cluster ID → human-readable label mapping
└── final_df_with_clusters.csv      # Full skill-clustering dataset with assigned clusters
```

---

## How the App Works

1. User pastes/enters a job posting's details (title, description, salary, logo presence, screening questions, etc.).
2. **Pipeline A** transforms the input using the same TF-IDF vectorizer, frequency maps, and one-hot columns from training, then outputs a fraud probability. At **threshold 0.2**, anything scoring ≥ 0.2 is flagged as "Likely Fake / Verify Before Applying."
3. User enters their own skills as free text (e.g., `"Python, SQL, Excel"`).
4. **Pipeline B** vectorizes those skills with the same TF-IDF vectorizer used for clustering, computes cosine similarity to each of the 10 cluster centroids, returns the best-fit cluster + confidence score, and lists missing top skills for that cluster.
5. Both results are displayed together for the same posting.

---

## Future Improvements

- Re-introduce time-trend analysis using a dataset spanning multiple months (either an older LinkedIn snapshot for comparison, or freshly scraped current postings).
- Fold in `companies.csv` (company size, country) as an additional cross-tab dimension.
- Explore HDBSCAN as an alternative to K-Means for handling irregular cluster shapes/noise.
- Expand skill granularity beyond the 35 broad LinkedIn categories (e.g., using richer NLP embeddings on free-text `skills_desc` or job descriptions).
- Validate the fraud model against more recent scam patterns, since EMSCAD is from 2012–2014.

---

## Honesty Note

This project deliberately documents **negative and inconclusive results** (rejected hypotheses, a tuning attempt that didn't help, a dropped project feature due to data limitations) alongside the positive ones. That's intentional — the goal was to demonstrate genuine data-driven decision-making, not to present a curated success story. Every claim in this README is backed by a specific number produced during the actual analysis described above.
