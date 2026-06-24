# Semiconductor Manufacturing Fault Detection using the SECOM Dataset

Final Machine Learning project for predicting semiconductor manufacturing process failures from high-dimensional process parameter measurements.

![Electronics manufacturing line](reports/figures/electronics_manufacturing.png)

## Contents

1. [Project Objective](#1-project-objective)
2. [Vocabulary](#2-vocabulary)
3. [Dataset Description](#3-dataset-description)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Notebook Structure](#5-notebook-structure)
6. [Final Results](#6-final-results)
7. [Setup and Usage](#7-setup-and-usage)
8. [Experiment Tracking](#8-experiment-tracking)
9. [Model Persistence](#9-model-persistence)
10. [Limitations](#10-limitations)
11. [Sources and References](#11-sources-and-references)

## 1. Project Objective

The goal of this project is to build a classical machine learning workflow that predicts whether a semiconductor manufacturing process run passes or fails quality control.

The project focuses on a realistic and reviewable pipeline:

- exploratory data analysis;
- missing-value and data-quality checks;
- leakage-safe preprocessing;
- baseline classical machine learning models;
- imbalance-aware evaluation;
- clear discussion of results and limitations.

## 2. Vocabulary

This project uses terminology from manufacturing and machine learning. The most important terms are:

- **Process run**: one row in the SECOM dataset. It represents one manufacturing observation collected during production. For example, it can be thought of as one item or batch passing through a monitored manufacturing process.
- **Process parameter**: one measured numerical variable from the production process. For a non-microelectronics example, imagine a factory producing smartphones: temperature, pressure, machine speed, alignment measurement, or electrical test reading could all be process parameters.
- **Target label**: the output value the model tries to predict. In this dataset, the label indicates whether the process outcome is normal or faulty.
- **SECOM label convention**: `1` means fail/faulty and `-1` means pass/normal.
- **Pass sample**: a process run that passed quality control.
- **Fail sample**: a process run that failed quality control.
- **Class imbalance**: a situation where one target class is much more common than the other. In SECOM, failures are rare compared with passes.

## 3. Dataset Description

The project uses the SECOM dataset, a public semiconductor manufacturing dataset containing hundreds of anonymized process parameter measurements collected during production.

Each row is one process run. Each parameter column contains one numerical measurement. The target label indicates whether the process run passed or failed quality control.

A simplified example of the dataset structure looks like this:

| process_run | parameter_0 | parameter_1 | parameter_2 | ... | target |
|---:|---:|---:|---:|---|---:|
| 1 | 3030.93 | 2564.00 | 2187.73 | ... | -1 |
| 2 | 3095.78 | 2465.14 | NaN | ... | -1 |
| 3 | 2932.61 | 2559.94 | 2165.20 | ... | 1 |

In this project, `-1` represents a pass sample and `1` represents a fail sample. `NaN` indicates a missing process parameter value.

The dataset is useful for this project because it contains realistic industrial data challenges:

- high dimensionality, with hundreds of process parameters;
- missing values across many parameters;
- constant or low-information columns;
- a strongly imbalanced target, because failures are rare;
- anonymized variables, which require statistical rather than domain-specific interpretation.

Raw data files are expected under `data/raw/` but are excluded from version control. Use `python scripts/download_data.py` to download and validate the official UCI files.

## 4. Mathematical Formulation

This is a binary classification problem.

Let:

- $n$ be the number of process runs;
- $p$ be the number of process parameters;
- $X \in \mathbb{R}^{n \times p}$ be the feature matrix;
- $x_i \in \mathbb{R}^p$ be the parameter vector for process run $i$;
- $y_i \in \{-1, 1\}$ be the target label for process run $i$, where $1$ means fail and $-1$ means pass.

The model learns a function:

$$
f(x_i) \rightarrow y_i
$$

or, for probabilistic models:

$$
P(y_i = \text{fail} \mid x_i)
$$

The main modeling challenge is not only to predict the majority class correctly, but also to identify rare failures. Because of class imbalance, accuracy alone may be misleading. Metrics such as recall, precision, F1-score, ROC-AUC, and Average Precision, which summarizes precision-recall behavior, are more informative.

Preprocessing must also be handled mathematically carefully. For example, if a parameter $x_j$ has missing values, an imputation value such as the training median should be computed only from the training split:

$$
\text{median}_{train}(x_j)
$$

and then applied to validation or test data. Computing this value using the full dataset would leak information from the test set into training.

Before imputation, parameters with more than `50%` missing values are removed.
The retained columns are learned from each training split or cross-validation
fold, so validation and test rows do not influence this decision.
The full-dataset EDA finds `28` such parameters, while the fixed training split
used by the final model removes `24`; this difference is expected because the
filter is learned without inspecting the held-out test rows.

## 5. Notebook Structure

The project is organized into four notebooks:

| Notebook | Purpose |
|---|---|
| `notebooks/01_secom_fault_detection.ipynb` | Main analysis, EDA, preprocessing, model comparison, final model selection. |
| `notebooks/02_uci_baseline_reproduction.ipynb` | Approximate reproduction attempt for the UCI feature-selection baseline. |
| `notebooks/03_mlflow_experiment_tracking.ipynb` | Manual MLflow logging for representative model configurations and artifacts. |
| `notebooks/04_model_persistence.ipynb` | Final fitted model artifact persistence and pickle round-trip validation. |

Run the notebooks in this order if regenerating results from scratch.

For review, start with `notebooks/01_secom_fault_detection.ipynb`. The other notebooks are supporting evidence: benchmark reproduction, experiment tracking, and persistence validation.

## 6. Final Results

The final selected model is:

| Item | Value |
|---|---|
| Model name | `RF + SelectKBest tuned` |
| Model type | Random Forest classifier |
| Preprocessing | remove parameters with >50% training missingness, median imputation, constant-parameter removal |
| Feature selection | `SelectKBest(f_classif, k=200)` |
| Hyperparameters | `n_estimators=100`, `class_weight="balanced"`, `max_depth=None`, `min_samples_leaf=5`, `random_state=42` |
| Positive class | `1 = Fail` |
| Negative class | `-1 = Pass` |
| Selected threshold | `0.35` |
| Threshold criterion | maximize Fail F1 on training-set out-of-fold predictions |

Held-out test metrics:

| Metric | Value |
|---|---:|
| Accuracy | `0.901` |
| Balanced accuracy | `0.638` |
| Fail precision | `0.292` |
| Fail recall | `0.333` |
| Fail F1 | `0.311` |
| ROC-AUC | `0.771` |
| Average Precision | `0.238` |

The threshold is selected using training-set out-of-fold predictions, not the held-out test set. Candidate selection is based on training-set CV/OOF results. The held-out test set is used only for final evaluation after the model and threshold are fixed.

## 7. Setup and Usage

From the project root, create and activate an environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download and validate the official UCI SECOM data:

```powershell
python scripts/download_data.py
```

Expected local raw files after download:

```text
data/raw/secom.data
data/raw/secom_labels.data
data/raw/secom.names
```

These raw data files are intentionally excluded from Git. The `data/raw/.gitkeep` file only preserves the empty directory structure. See `data/README.md` for source and validation details.

Run the notebooks in order:

```text
notebooks/01_secom_fault_detection.ipynb
notebooks/02_uci_baseline_reproduction.ipynb
notebooks/03_mlflow_experiment_tracking.ipynb
notebooks/04_model_persistence.ipynb
```

Notebook 01 performs candidate comparison and model selection using only training-set cross-validation and out-of-fold threshold tuning. The held-out test set is used only after the final model and threshold are fixed; final test metrics are evaluation metrics, not model-selection metrics.

## 8. Experiment Tracking

MLflow tracking is implemented in `notebooks/03_mlflow_experiment_tracking.ipynb`.

From the project root, open the local MLflow UI with:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri file:.\mlruns --port 5000
```

Then open `http://127.0.0.1:5000` in a browser and select the `secom-fault-detection` experiment.

The generated `mlruns/` directory is local experiment state and is excluded from Git. Notebook 03 logs held-out test metrics only for the selected final model.

## 9. Model Persistence

The final fitted artifact is saved by `notebooks/04_model_persistence.ipynb` under:

```text
models/secom_best_pipeline.pkl
```

The pickle contains a dictionary with the fitted pipeline and the selected decision threshold:

```python
artifact = {
    "pipeline": fitted_pipeline,
    "fail_threshold": 0.35,
    "max_missing_fraction": 0.50,
    "positive_label": 1,
    "negative_label": -1,
    "model_name": "RF + SelectKBest tuned",
    "selection_metric": "Fail F1",
}
```

Use the persisted threshold when converting Fail probabilities into predictions:

```python
from src.evaluation import predict_with_threshold
from src.final_model import load_final_model_artifact

artifact = load_final_model_artifact("models/secom_best_pipeline.pkl")
y_pred, y_fail_proba = predict_with_threshold(
    artifact["pipeline"],
    X_new,
    artifact["fail_threshold"],
    positive_label=artifact["positive_label"],
    negative_label=artifact["negative_label"],
)
```

Only load this pickle from a trusted source. Pickle deserialization can execute code, and loading should use a compatible Python and library environment. The package versions used for the saved artifact are recorded in `environment_versions.txt`.

## 10. Limitations

- The dataset is small for the Fail class: only `104` failed process runs are available.
- Only `83` Fail samples are present in the training split used for model and threshold selection. Out-of-fold predictions prevent direct training-row leakage, but hyperparameter and threshold comparisons can still be noisy or optimistic because the same limited training data guides those choices.
- The process parameters are anonymized, so causal or physical interpretation is limited.
- The dataset is high-dimensional and contains missing values.
- The final model is best viewed as a risk-screening model, not a fully reliable automatic fault detector.

## 11. Sources and References

Main dataset and benchmark source:

- McCann, M. & Johnston, A. (2008). **SECOM**. UCI Machine Learning Repository. DOI: [10.24432/C54305](https://doi.org/10.24432/C54305). Dataset page: [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom).
- The UCI SECOM page is the source for the dataset size, label convention, missing-value description, and the published 10-fold baseline table used in `notebooks/02_uci_baseline_reproduction.ipynb`.

Implementation references used in the notebooks:

- scikit-learn: [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [common pitfalls / data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- scikit-learn: [`SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) and [`f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html).
- scikit-learn: [`KernelRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html), used in notebook 02 through a small classifier wrapper for the approximate UCI baseline reproduction.
- scikit-learn: [`balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) and [`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- scikit-learn: [Precision-Recall example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html), used as background for evaluating imbalanced classification.
