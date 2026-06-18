# Semiconductor Manufacturing Fault Detection using the SECOM Dataset

Final Machine Learning project for predicting semiconductor manufacturing process failures from high-dimensional process parameter measurements.

![Electronics manufacturing line](reports/figures/electronics_manufacturing.png)

## Contents

1. [Project Objective](#1-project-objective)
2. [Vocabulary](#2-vocabulary)
3. [Dataset Description](#3-dataset-description)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Planned Methodology](#5-planned-methodology)
6. [Expected Deliverables](#6-expected-deliverables)

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

## 5. Planned Methodology

1. Load and inspect the SECOM feature and label files.
2. Define clear terminology and target-label meaning.
3. Perform exploratory data analysis focused on shape, target imbalance, missing values, and data quality.
4. Split the data into training and test sets using stratification.
5. Build `scikit-learn` Pipelines for preprocessing and modeling.
6. Fit learned preprocessing steps only on training data to avoid data leakage.
7. Train classical baseline models such as Logistic Regression, Random Forest, and Support Vector Machines.
8. Evaluate models using metrics suitable for imbalanced classification.
9. Explore dimensionality reduction with PCA as an optional later comparison.
10. Discuss limitations and possible future improvements.

## 6. Expected Deliverables

- A GitHub repository with a clean project structure.
- One main Jupyter notebook in English: `notebooks/01_secom_fault_detection.ipynb`.
- Reusable helper modules under `src/`.
- A README with project context, vocabulary, methodology, and mathematical framing.
- Figures and evaluation outputs saved under `reports/figures/` when useful.
