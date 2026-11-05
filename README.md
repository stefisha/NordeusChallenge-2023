# Nordeus Data Science Challenge 2023

**Predicting Top Eleven league positions**

Predict each club's final position in a league of 14 teams using player quality,
activity, engagement, and resource features from the previous season. The challenge
supports the broader goal of creating fair and competitive league matchmaking.

[Notebook](NordeusChallenge.ipynb) · [Submission](league_rank_predictions.csv) · [Challenge brief](Challenge/1.%20Data%20Science%20challenge-%20JobFair%202023.pdf)

## Task and data

| Dataset | Season | Clubs | Leagues | Purpose |
| --- | ---: | ---: | ---: | --- |
| `Challenge/jobfair_train.csv` | 173 | 55,314 | 3,951 | Labeled training data |
| `Challenge/jobfair_test.csv` | 174 | 60,270 | 4,305 | Prediction inputs |

Every league contains 14 clubs. The target, `league_rank`, ranges from 1 for first
place to 14 for last place. Test labels are not included. Predictions are evaluated
using **mean absolute error (MAE)**, measured in league positions; lower is better.

The inputs include team age and quality, previous activity and match outcomes,
training, spending and resource balances, and registration categories. `club_id`
and `league_id` are retained for identification and excluded from model inputs.

## Models and saved results

The notebook compares classification approaches with numerical imputation and
categorical one hot encoding. It also includes exploratory parameter searches and
feature importance analysis.

| Model | Saved five fold MAE |
| --- | ---: |
| Random forest | 2.8027 |
| XGBoost | 2.7959 |
| LightGBM | 2.8163 |
| Random forest and XGBoost stacking | **2.5997** |

These values come from saved notebook outputs on the training season. They are
not independently reproduced benchmarks or scores on the hidden test labels.
The stacking classifier uses logistic regression to combine the base models and
produces the checked in submission.

## Run locally

Use Python 3.9 to 3.12 and start Jupyter from the repository root:

```sh
git clone https://github.com/stefisha/NordeusChallenge-2023.git
cd NordeusChallenge-2023
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt jupyterlab
jupyter lab NordeusChallenge.ipynb
```

On Windows, activate with `.venv\Scripts\activate`. The notebook defaults to the
included `Challenge/` directory. For another data location, set `NORDEUS_DATA_DIR`
before running its loading cell. In Colab, download the two CSV files or mount your
own storage and point that variable to their directory.

The requirements pin the notebook dependencies, not the original Colab environment.
The notebook contains repeated model fitting, nested stacking validation, and
parameter searches. Review those cells before running the entire notebook; the
complete training procedure has not been rerun as part of repository validation.

## Submission

`league_rank_predictions.csv` contains **60,270 rows**, one per test club:

| Column | Meaning |
| --- | --- |
| `club_id` | Club identifier in the test dataset |
| `predicted_league_rank` | Predicted integer position from 1 to 14 |

Check the file contract and current notebook link without installing model libraries:

```sh
python scripts/validate_submission.py
```

The validator checks unique clubs, test ordering, required columns, and prediction
bounds. GitHub Actions runs the same check on pushes and pull requests. It does
not evaluate predictive accuracy or enforce a complete league ordering.

## Evaluation considerations

The notebook fits preprocessing before cross validation and splits clubs rather
than holding out complete leagues. A stronger evaluation should fit transformations
inside each fold and assess generalization to unseen leagues and later seasons.

The classifiers predict clubs independently, so positions can repeat within a
league. They do not enforce a permutation of positions 1 to 14. Regression or an
ordinal model with an explicit league ranking step would be useful alternatives.
The MAE objective also deserves consideration when choosing how class probabilities
become a final predicted position.

## Repository contents

```text
NordeusChallenge.ipynb        Modeling, exploration, and saved outputs
league_rank_predictions.csv  Submitted club predictions
Challenge/                   Task brief and supplied datasets
scripts/                     Lightweight submission validation
requirements.txt             Notebook dependencies
.github/workflows/           Automated file validation
```

The challenge brief and datasets are supplied assessment materials. Their inclusion
does not assert a separate redistribution license.

**Author:** [Stefan Veličković](https://github.com/stefisha)
