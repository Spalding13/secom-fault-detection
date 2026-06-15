# Semiconductor Manufacturing Fault Detection using the SECOM Dataset

Final Machine Learning project for predicting semiconductor manufacturing process failures from sensor measurements.

## Objective

The goal of this project is to build a classical machine learning workflow that predicts whether a semiconductor manufacturing process run, represented by a wafer, passes or fails. The project focuses on a realistic and reviewable pipeline: exploratory data analysis, leakage-safe preprocessing, baseline modeling, class imbalance handling, evaluation, and interpretation of limitations.

## Dataset Description

The project uses the SECOM dataset, a public semiconductor manufacturing dataset containing hundreds of anonymized sensor measurements collected during production. Each row represents one process run or wafer, and the target label indicates whether the run passed or failed quality control.

The dataset is high dimensional, contains missing values, and is strongly imbalanced because failures are relatively rare. These properties make it suitable for studying practical issues in industrial fault detection.

Raw data files are expected to be placed locally under `data/raw/` and are excluded from version control when large.

## Planned Methodology

1. Load and inspect the SECOM features and labels.
2. Perform exploratory data analysis focused on missing values, feature distributions, constant columns, and target imbalance.
3. Split the data into training and test sets using stratification.
4. Build scikit-learn Pipelines for preprocessing and modeling.
5. Fit preprocessing steps only on training data to avoid data leakage.
6. Train classical baseline models such as logistic regression, random forest, and support vector machines.
7. Evaluate models using metrics suitable for imbalanced classification, including recall, precision, F1-score, ROC-AUC, and PR-AUC.
8. Explore dimensionality reduction with PCA as an optional comparison.
9. Discuss limitations and possible future improvements.

## Expected Deliverables

- A GitHub repository with clean project structure.
- One main Jupyter notebook in English: `notebooks/01_secom_fault_detection.ipynb`.
- Reusable helper modules under `src/`.
- A concise README describing the project and workflow.
- Figures and evaluation outputs saved under `reports/figures/` when useful.

## Timeline

Deadline: 23 June 2026.

The first milestone is a working notebook skeleton and repository structure. Later milestones will add data loading, exploratory analysis, preprocessing pipelines, model training, and evaluation.

