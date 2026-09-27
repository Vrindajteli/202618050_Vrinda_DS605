# DS605 — Lab 5: Garment Worker Productivity
## Name : Vrinda Janak Teli
## Student ID : 202618050

Predicting garment factory team productivity two ways: once using Scikit-learn, and once by rebuilding the same steps from scratch with only NumPy and Pandas — then comparing the two and improving the from-scratch version.

## What is done

1. **Part A — Scikit-learn.** Cleans and encodes the data with `ColumnTransformer` + `Pipeline`, trains a `LinearRegression` on `actual_productivity` and a `LogisticRegression` on a `MeetsTarget` label built from `actual_productivity >= targeted_productivity`. Records training time, prediction time, and MAE/RMSE/R² (regression) and accuracy/precision/recall/F1 (classification).
2. **Part B — From scratch.** Rebuilds median imputation, one-hot encoding, standard scaling, closed-form linear regression, and gradient-descent logistic regression using only NumPy and Pandas, on the exact same train/test rows, and evaluates with hand-written metric functions.
3. **Part C — Comparison and optimization.** Puts both versions side by side, then speeds up the manual logistic regression and tunes the ridge penalty for the manual linear regression using a validation split carved out of the training data (never the test set). Ends with a final three-way comparison table and a plain-language write-up of what changed and why.

## How to run it

```bash
pip install -r requirements.txt
jupyter notebook Lab5_Garment_Productivity.ipynb
```
## Key observations

- Both implementations land on very similar predictive scores for both tasks — expected, since they solve the same equations.
- The main gap in the baseline versions is **training time** for logistic regression: gradient descent needs thousands of small steps, while Scikit-learn's solver converges much faster.
- Switching the manual logistic regression to our optimization method closed that gap, because it uses the curvature of the loss to jump toward the answer instead of taking small steps.
- Tuning the ridge penalty for linear regression on a validation slice gave a small, honest improvement over the untuned version, without touching the test set.
- R² for the regression task stays fairly low across all versions — the dataset doesn't record several things that likely drive day-to-day productivity (worker experience, machine downtime, order-specific issues), so this is a limit of the available features rather than of any particular implementation.


## Dataset

Al Imran, A. (2020). *Productivity Prediction of Garment Employees* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51S6D

