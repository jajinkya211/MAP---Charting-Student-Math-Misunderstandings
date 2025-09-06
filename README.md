# MAP (Misconception Annotation Project) — Offline TF-IDF Baseline

A simple, **fully offline** starter/benchmark for the Kaggle MAP competition.

* **Goal**: From a student’s free-text explanation + question context, predict up to **3** `Category:Misconception` tokens per row.
* **Metric**: MAP\@3
* **Constraints**: Kaggle **Internet OFF**. This repo avoids any downloads and runs with scikit-learn only.
* **What you get**: A robust TF-IDF + linear pipeline with QuestionId priors, gated decoding, and a **format-correct** `submission.csv`.


## Approach

1. **Text normalize** (lowercase, trim, whitespace).
2. **Features**
   • Separate blocks for **Question+MC** and **Explanation**
   • **Word** (1–3 n-grams) + **Character** (3–6 n-grams) TF-IDF
3. **Two heads**
   • **Category** head trained on exactly **3 canonical labels**: `True_Correct`, `False_Neither`, `False_Misconception`
   • **Misconception** head trained on all misconception labels (including `"None"`)
   • Linear models only: Calibrated LinearSVC for Category (fallback to SGD logistic), SGD logistic for Misconception
   • **Single-class safeguards**: if a head sees one class, it becomes a constant predictor (prevents crashes)
4. **QuestionId priors**
   • Compute Laplace-smoothed label distributions per `QuestionId`
   • Blend model probabilities with priors for test rows (fall back to global priors if no `QuestionId`)
5. **Gated decoding**
   • Only `False_Misconception` may pair with a non-`None` misconception
   • Other categories always pair with `:None`
   • Confidence gates avoid low-probability spam; exactly **3 tokens** per row, deduped

---
## Tuning knobs (safe to change)

* **Block weights** (feature importance):
  `w_exp_word`, `w_exp_char`, `w_qmc_word`
* **Prior blending** (0–1):
  `cat_proba = blend_with_prior(..., w=0.20)`
  `mis_proba = blend_with_prior(..., w=0.12)`
  Increase `w` to trust **QuestionId priors** more; decrease to favor model predictions.
* **Decoder gates**:
  `mis_gate_thr` (allow misconception only if category is confident; try 0.30–0.45)
  `mis_min_conf` (minimum misconception prob; try 0.05–0.15)
  `mis_max_per_row` (usually 1)

## License

MIT — use freely; attribution appreciated.
