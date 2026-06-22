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

Raw data files are expected to be placed locally under `data/raw/` and are excluded from version control when large.

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

The main modeling challenge is not only to predict the majority class correctly, but also to identify rare failures. Because of class imbalance, accuracy alone may be misleading. Metrics such as recall, precision, F1-score, ROC-AUC, and precision-recall AUC are more informative.

Preprocessing must also be handled mathematically carefully. For example, if a parameter $x_j$ has missing values, an imputation value such as the training median should be computed only from the training split:

$$
\text{median}_{train}(x_j)
$$

and then applied to validation or test data. Computing this value using the full dataset would leak information from the test set into training.

## 5. Notebook Structure

The project is organized into four notebooks:

| Notebook | Purpose |
|---|---|
| `notebooks/01_secom_fault_detection.ipynb` | Main analysis, EDA, preprocessing, model comparison, final model selection. |
| `notebooks/02_uci_baseline_reproduction.ipynb` | Approximate reproduction attempt for the UCI feature-selection baseline. |
| `notebooks/03_mlflow_experiment_tracking.ipynb` | Manual MLflow logging for representative model configurations and artifacts. |
| `notebooks/04_model_persistence.ipynb` | Final fitted model artifact persistence and pickle round-trip validation. |

Run the notebooks in this order if regenerating results from scratch.

## 6. Final Results

The final selected model is:

| Item | Value |
|---|---|
| Model name | `RF + SelectKBest tuned` |
| Model type | Random Forest classifier |
| Preprocessing | median imputation, constant-parameter removal |
| Feature selection | `SelectKBest(f_classif, k=200)` |
| Hyperparameters | `n_estimators=100`, `class_weight="balanced"`, `max_depth=8`, `min_samples_leaf=5`, `random_state=42` |
| Positive class | `1 = Fail` |
| Negative class | `-1 = Pass` |
| Selected threshold | `0.35` |
| Threshold criterion | maximize Fail F1 on training-set out-of-fold predictions |

Held-out test metrics:

| Metric | Value |
|---|---:|
| Accuracy | `0.869` |
| Balanced accuracy | `0.665` |
| Fail precision | `0.237` |
| Fail recall | `0.429` |
| Fail F1 | `0.305` |
| ROC-AUC | `0.774` |
| PR-AUC / Average Precision | `0.284` |

The threshold is selected using training-set out-of-fold predictions, not the held-out test set. The held-out test set is used only for final evaluation.

## 7. Setup and Usage

Install dependencies from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If creating a new environment, install the requirements into that environment first, then register or select the notebook kernel used by the project.

Expected local raw files:

```text
data/raw/secom.data
data/raw/secom_labels.data
data/raw/secom.names
```

These raw data files are intentionally excluded from Git.

## 8. Experiment Tracking

MLflow tracking is implemented in `notebooks/03_mlflow_experiment_tracking.ipynb`.

From the project root, open the local MLflow UI with:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri file:.\mlruns --port 5000
```

Then open `http://127.0.0.1:5000` in a browser and select the `secom-fault-detection` experiment.

The generated `mlruns/` directory is local experiment state and is excluded from Git.

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
- The process parameters are anonymized, so causal or physical interpretation is limited.
- The dataset is high-dimensional and contains missing values.
- The final model is best viewed as a risk-screening model, not a fully reliable automatic fault detector.
