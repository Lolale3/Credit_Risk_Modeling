# Credit Risk Modeling — PD Model and Application Scorecard

A probability-of-default model on Lending Club consumer loans, turned into a
FICO-style scorecard and, finally, into an approval decision.

The model was the easy part. The decision was the point: moving the approval
cutoff from the default 0.5 to a cost-aware 0.19 cut the rejection rate from
**27.6% to 7.4%** while reducing missed defaults, on the same model.

---

## Results

| Metric | Value |
|---|---|
| AUROC (test) | 0.866 |
| Gini | 0.732 |
| AUC-PR | 0.976 |
| Cross-validated AUROC (5-fold × 3 repeats) | 0.866 |
| Scorecard range | 300–850 |
| Approval cutoff score | 488 |

**Data:** 466,285 loans issued 2007–2014, split 373,028 train / 93,257 test,
stratified. 75 raw features reduced to 51, then to ~20 WoE-binned predictors.

### The threshold decision

The default 0.5 cutoff quietly assumes a false approval and a false rejection
cost the same. In lending they don't — a charged-off loan costs far more than
the margin forgone by declining a good applicant, and the class balance is
roughly 9:1 good to bad.

Selecting the cutoff by Youden's J instead of convention:

| | Threshold 0.50 | Threshold 0.19 |
|---|---|---|
| Rejection rate | 27.6% | 7.4% |
| Approved | 67,561 | 86,386 |
| False negatives (good loans rejected) | 18.7% | 2.6% |
| False positives (bad loans approved) | 2.1% | 6.3% |

The 0.5 cutoff rejects more than a quarter of applicants — a large,
unnecessary loss of business — and rejects good borrowers seven times more
often than the tuned cutoff does. The tuned cutoff accepts more bad loans in
exchange, which is the trade a lender should be making explicitly rather than
inheriting from a library default.

---

## Method

**Target definition.** `loan_status` collapsed to binary: Charged Off,
Default, Late (31–120 days), and "Does not meet the credit policy: Charged
Off" are bad (0); everything else good (1). Roughly 11% bad.

**Leakage removal.** Features populated only *after* a borrower defaults —
`recoveries`, `collection_recovery_fee`, `total_rec_prncp`,
`total_rec_late_fee` — were dropped before modeling. They predict default
almost perfectly and are worthless at application time.

Information Value doubled as a leakage detector: `last_pymnt_amnt` and
`mths_since_last_pymnt_d` were discarded for *abnormally high* IV, which in a
credit scorecard is a symptom rather than a win.

**Feature selection.** Chi-squared tests for categorical features, ANOVA
F-statistic for numerical ones. Top 4 categorical and top 20 numerical
retained; `out_prncp_inv` and `total_pymnt_inv` dropped for multicollinearity
with their non-`_inv` counterparts. Features with negligible IV and flat WoE
(`emp_length`, `total_acc`, `tot_cur_bal`) were dropped as having no
discriminatory power.

**WoE binning.** Every predictor was binned by Weight of Evidence, with bins
inspected individually and sparse bins merged into neighbours. Long-tailed
variables (`annual_inc` up to $75M, `revol_util` above 1.0) were capped into
a single top bin before binning the rest.

WoE binning is implemented as a custom scikit-learn transformer
(`WoE_Binning`) inside a `Pipeline`, so the binning is fit on training folds
only and cannot leak across cross-validation splits.

**Model.** Logistic regression with `class_weight='balanced'`, evaluated by
`RepeatedStratifiedKFold` (5 splits × 3 repeats) on AUROC.

Logistic regression is not the highest-AUROC option available, and that is
deliberate. A credit model has to be explainable and defensible: WoE bins
produce monotonic, readable relationships, missing values become their own
bin, and the coefficients convert directly into a scorecard a credit officer
or a regulator can read line by line. In lending, a model everyone can audit
beats a black box that scores two points higher.

**Scorecard.** Coefficients scaled linearly to the FICO range (300–850), with
reference categories fixed at zero. Scores computed on the test set by matrix
multiplication of the WoE-transformed features against the scorecard.

---

## Honest limitations

- **Some behavioural features remain.** `out_prncp`, `total_pymnt` and
  `total_rec_int` describe loan performance *after* origination. They're
  legitimate for a behavioural scorecard on an existing book, but for an
  application scorecard — scoring someone before you lend — they would not be
  available. Removing them would lower the AUROC and make the model a truer
  application model.
- **No out-of-time validation.** The split is random, not chronological. A
  production credit model should be validated on a later time period than it
  was trained on, since economic conditions shift.
- **Youden's J is not a cost function.** It weights sensitivity and
  specificity equally. A real cutoff would come from the actual cost ratio —
  expected loss given default against the margin on a good loan — which would
  likely land somewhere different. J is a defensible stand-in when those
  numbers aren't available, not a substitute for them.
- **PD only.** Expected loss is PD × LGD × EAD. This covers the first term.
- **No PSI / stability monitoring.** A deployed scorecard needs population
  stability tracking to detect drift between the development sample and
  incoming applicants.

---

## Running it

`Credit_Risk_Modelling.ipynb` was written in Google Colab and reads the
dataset from Google Drive. To run it elsewhere, replace the `drive.mount`
cell with a local path to `loan_data_2007_2014.csv`
([Lending Club data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)).

Dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`,
`seaborn`.

---

## Why this project exists

The interesting question in risk modeling isn't how well a model ranks
borrowers — it's where you draw the line, and what you decided an error was
worth when you drew it. That question shows up everywhere: in credit
approvals, in whether to trust a model's output or escalate it to a human, in
whether a forecast's confidence interval deserves to be believed. This is the
version of it that's easiest to read.
