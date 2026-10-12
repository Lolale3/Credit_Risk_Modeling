# Credit Risk Modelling: Application Scorecard and Profit-Based Cutoff

This project builds a probability-of-default (PD) model for consumer loans.
The model uses only data that a lender has at the time of approval.
The project then converts the model into a 300–850 scorecard.
Finally, it finds the cutoff that gives the most profit, not the best statistic.

**Data:** Lending Club loans, 2007–2014 (466,285 loans, 75 columns).
**Method:** WoE binning and logistic regression.
**Validation:** out-of-time. The model trains on 2007–2011 loans and is tested on 2012 loans.

---

## Main results

| Result | Value |
|---|---|
| Out-of-time Gini (2012 loans) | **0.291** (95% CI 0.277–0.305) |
| Gini of LendingClub's own sub-grade | 0.240 |
| KS | 0.213 |
| AUROC | 0.645 |
| Score stability (PSI), 2012 and 2013–14 | 0.075 and 0.020 (stable) |

1. The model ranks borrowers better than LendingClub's own sub-grade.
2. The standard statistical cutoff (Youden's J) catches 55% of defaults.
   But it rejects 34% of good borrowers and loses 37% of the profit.
3. A cutoff that changes with the loan's interest rate is the only rule that increases profit.

---

## Why this is version 2

Version 1 of this project had a Gini of 0.73.
That result was not correct, because the model used data from after approval.
Version 2 corrects these problems:

| Problem in v1 | Correction in v2 |
|---|---|
| The model used fields that exist only after the loan starts (`out_prncp`, `total_pymnt`, `total_rec_int`, payment dates). These fields leak the outcome. | v2 removes these fields. It keeps them in a separate `outcomes` table. This table defines the target and measures profit. The model never sees it. |
| `Current` loans counted as good loans. Many of them had no time to default. | v2 uses only 36-month loans that ended before the data snapshot (January 2016). Each outcome is final. |
| A random train/test split. | An out-of-time split: train on 2007–2011, test on 2012. |
| `class_weight='balanced'` made the probabilities too high. | v2 removes it. The predicted PD is now a real probability. |
| `revol_util` is in percent, but v1 binned it as a fraction. About 99% of loans went into one bin. | v2 bins `revol_util` in percent. |
| The "Youden" cutoff (0.184) came from the wrong array, and v1 selected it on the test data. | v2 selects the Youden cutoff on the training data only. |
| Some cells did not run on pandas 2 or later. | v2 runs on current pandas. |

The v2 Gini is lower.
It is the correct number for a model that scores applicants before approval.

---

## Data and population

| Group | Loans issued | Number of loans | Bad rate | Use |
|---|---|---|---|---|
| Train | 2007–2011 | 31,534 | 12.3% | Fit the bins and the model |
| Test | 2012 | 43,425 | 13.7% | Out-of-time results |
| Recent | 2013–2014 | 262,949 | not final | Stability check (PSI) only |

A "bad" loan has the status Charged Off, Default, or Late (31–120 days).

---

## The model

The model uses 10 features. All of them are known at the time of application:

- `annual_inc`, `loan_to_income` (loan amount / annual income)
- credit history length, in months, at the issue date
- `inq_last_6mths`, `mths_since_last_delinq`, `pub_rec`
- `revol_util`, `total_acc`
- `grade`, `purpose`

**Steps:**

1. The `WoE_Binning` transformer fits the bins on the training data only.
2. The bins are monotonic. Small bins merge into the adjacent bin.
3. The transformer removes features with low Information Value (IV < 0.02).
4. It also removes one feature from each pair with a correlation above 0.75.
5. A logistic regression uses the binned features.
6. The coefficients become a 300–850 scorecard.

**Calibration.** The model ranks the loans correctly in all 10 score bands.
But the mean predicted PD (11.2%) is less than the actual bad rate (13.7%).
The 2012 loans were riskier than the earlier loans.
To correct this, recalibrate the intercept on a recent group of loans.

**In-time and out-of-time results.** The cross-validated Gini on the training data is 0.381.
The out-of-time Gini is 0.291.
The model does not overfit (the train Gini is 0.390).
The decrease comes from the change in the borrower population over time.

---

## The cutoff decision

A lower bad rate is not the goal. Profit is the goal.

The project measures two values from the real cash flows of the training loans:

- **g = 15.6%:** the return on a loan that the borrower repays.
- **l = 44.7%:** the net loss on a loan that defaults (EAD 62%, LGD 93%, less the interest and fees received before the default).

The bank should approve a loan when its expected profit is more than zero.
This is true when **PD < g / (g + l) = 25.8%**.

A loan with a higher interest rate earns more.
So the **price-aware rule** lets the cutoff increase with the rate:
from 14.4% at a 5.8% rate to 36.0% at a 17.9% rate.
The rule uses the interest rate only for the decision, not as a model feature.

### Results on the 2012 loans (real cash flows)

| Cutoff rule | Loans rejected | Good loans rejected | Defaults caught | Bad rate of approved loans | Profit change |
|---|---|---|---|---|---|
| Approve all ($50.95M profit) | 0% | 0% | 0% | 13.7% | — |
| Youden's J | 37.1% | 34.3% | 54.9% | 9.8% | **−37.2%** |
| Flat profit rule (PD < 25.8%) | 1.9% | 1.5% | 4.4% | 13.3% | −0.3% |
| Price-aware rule | 0.3% | 0.3% | 0.9% | 13.6% | **+0.24%** (+$0.12M) |

- Youden's J gives the lowest bad rate, but it loses the most money.
- The price-aware rule is the only rule that increases profit.
  The 95% confidence interval of the gain (+$0.01M to +$0.25M) is above zero.
- The gain is small because LendingClub's prices already covered the risk.
  Each grade gave a positive return, from +6.7% (grade A) to +22.8% (grade G).

---

## Monitoring and reason codes

**Stability.** The score is stable (PSI 0.075 for 2012, 0.020 for 2013–14).
Two features changed more:

- `purpose`: CSI 0.34 for 2013–14. This change is large.
- `revol_util` and credit history length: CSI about 0.19. Monitor these features.

**Reason codes.** For each declined applicant, the scorecard gives the features that cost the most points.
This is similar to the reasons in an adverse-action notice.
The first reason is `purpose` in 76% of declines,
`annual_inc` in 13%, and `inq_last_6mths` in 11%.

---

## Limitations

- **No reject inference.** The data contains only funded loans.
- **No FICO score.** The data does not include it.
- **Gross profit only.** The profit does not include funding and servicing costs.
- **36-month loans only.** The 60-month loans have no final outcomes in this data.
- **No fair-lending review.** Do a fair-lending review before you use this model on real applicants.

---

## How to run

The notebook runs in Google Colab or in a local Jupyter environment.

1. Download the data from Kaggle:
   [Loan data 2007–2014 (devanshi23)](https://www.kaggle.com/datasets/devanshi23/loan-data-2007-2014).
   The file name is `loan_data_2007_2014.csv`.
2. **In Colab:** put the file in `My Drive/Colab Notebooks/`.
   The notebook connects to Google Drive and reads the file from there.
3. **On your computer:** put the file in the same folder as the notebook.
   Or set the `LOAN_DATA_PATH` environment variable to the file path.
4. Open `Credit_Risk_Modelling.ipynb` and run all cells.

`credit_risk_modelling.py` contains the same code as a Python script.

**Requirements:** Python 3.10+, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn.



