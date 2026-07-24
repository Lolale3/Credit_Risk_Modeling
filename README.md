# Credit Risk Modeling — Probability of Default Scorecard

A probability-of-default (PD) model built on Lending Club consumer loan data, turned into an interpretable, FICO-style credit scorecard. The focus is on the parts that matter in a real lending context: **explainability, leakage control, and choosing a decision cutoff by its cost - not by default.**

## Problem

Given a consumer loan application, estimate the probability that the borrower will go bad (charge-off / serious delinquency), and translate that into an approve/decline decision that a business user can read and defend.

## Data

- **Source:** Lending Club loans (Kaggle), ~466K loans issued 2007–2014, 74 raw features.
- **Target:** `loan_status` collapsed to binary - *bad* = Charged Off, Default, and Late (31–120 days); *good* = everything else. ~11% bad rate.
- **Split:** 80/20 stratified (≈373K train / 93K test), split **before** cleaning and binning so no test information leaks into feature decisions.

## Approach

**1. Leakage control (the part I spent the most time on)**
- Dropped forward-looking fields that only populate *after* a loan goes bad (`recoveries`, `collection_recovery_fee`, `total_rec_prncp`, etc.).
- Used **Information Value (IV) as a leakage detector**: features with abnormally high IV (>~1.0, e.g. `last_pymnt_amnt`, `mths_since_last_pymnt_d`) were dropped as suspicious rather than kept as "strong predictors."

**2. Feature selection**
- Chi-square for categorical features, ANOVA F-test for numeric features.
- Correlation check to remove multicollinear pairs.
- Dropped near-zero-IV features (e.g. `emp_length`, `total_acc`) as noise.

**3. Weight of Evidence (WOE) binning**
- Every feature binned and converted to WOE, so all inputs enter the model on a common log-odds scale with monotonic, interpretable relationships.
- Sparse categories merged; WOE curves inspected for monotonicity.
- Implemented as a **custom scikit-learn transformer** (`BaseEstimator` / `TransformerMixin`) inside a `Pipeline`, so the exact same binning is applied to any new data at scoring time.

**4. Model**
- **Logistic regression** on WOE-transformed features - chosen for interpretability and regulatory defensibility over a marginally higher-AUC black box.
- `class_weight='balanced'` for cost-sensitive learning on the imbalanced target.

## Results

| Metric | Value |
|---|---|
| Test AUROC | 0.866 |
| Test Gini | 0.73 |
| Cross-val AUROC (RepeatedStratifiedKFold, 5×3) | 0.866 |

Train and test performance agree closely - no meaningful overfitting.

## Scorecard & Decision Cutoff

- Model coefficients scaled to a **FICO-style 300–850 score**.
- **Cutoff chosen by cost, not the default 0.5.** The 0.5 threshold rejected ~28% of applicants — most of them actually good borrowers. Using Youden's J, the operating point moved to ~0.184, which:
  - dropped the rejection rate from ~28% to ~7%, and
  - cut false rejections of good customers from ~19% to under 3%,
  - at the cost of approving somewhat more bads.

The point: the "right" cutoff is a business decision about the cost of a bad loan vs. the margin lost declining a good applicant. Youden's J is a reasonable statistical stand-in when those costs aren't given.

## What I'd add for production

- **Out-of-time validation** on a later vintage rather than a random split - 2007–2014 spans the financial crisis, and population shift matters.
- **PSI monitoring** to detect when the applicant population drifts away from the training distribution and the scorecard needs a refresh.

## Stack

Python · pandas · scikit-learn · NumPy · matplotlib

---

*Built as a personal project to go deeper into credit risk modeling and scorecard development.*
