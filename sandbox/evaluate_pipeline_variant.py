import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import balanced_accuracy_score \
                            ,accuracy_score \
                            ,precision_score\
                            ,recall_score \
                            ,f1_score \
                            ,confusion_matrix

DEFAULT_THRESHOLDS = np.arange(0.05, 0.96, 0.05)

# Use OOF to tune the threshold
def _threshold_search(oof_fail_proba, y_train, metric, thresholds = None):

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    rows = []

    # Iterate over each threshold value
    for threshold in thresholds:
        # Evaluate class prediction, based on current threshold
        y_pred = np.where(oof_fail_proba >= threshold, 1, -1)

        # Calculate metrics
        precision = precision_score(
            y_train,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    
        recall = recall_score(
            y_train,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    
        f1 = f1_score(
            y_train,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
        # Store results for current threshold val
        rows.append({
            "threshold": threshold,
            "fail_precision": precision,
            "fail_recall": recall,
            "fail_f1": f1,
        })

    threshold_results = pd.DataFrame(rows)

    return threshold_results.sort_values(metric, ascending=False).iloc[0]

def evaluate_pipeline_variant(pipeline, variant_name, 
                              X_train, y_train, X_test, 
                              y_test, cv, metric, thresholds=None):
    
    # 1. OOF probabilities
    # 2. threshold search on OOF
    # 3. pick best threshold
    # 4. fit final pipeline on full train
    # 5. predict probabilities on test
    # 6. apply selected threshold
    # 7. compute test metrics
    # 8. build confusion matrix
    # 9. return all results

    print(f"Computing variant: {variant_name}...")
    
   
    # Generate Out-of-fold-probabilities
    oof_proba = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )

    # Collect OOF prob. for the failing class
    # Check indexes
    fail_class_index = list(pipeline.named_steps["classifier"].classes_).index(1)
    oof_fail_proba = oof_proba[:, fail_class_index]
    
    # Pick the best threshold
    best_row = _threshold_search(oof_fail_proba, y_train, metric, thresholds)
    best_threshold = best_row["threshold"]


    # Train a final model on the entire training set
    pipeline.fit(X_train, y_train)
    
    test_proba = pipeline.predict_proba(X_test)
    test_fail_proba = test_proba[:, 1]
    
    # Collect predictions, based on model probabilities on test set 
    # by using the best rated fail_f1 threshold value
    y_test_pred = np.where(test_fail_proba >= best_threshold, 1, -1)
    
    # Compute test metrics
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "fail_precision": precision_score(y_test, y_test_pred, pos_label=1, zero_division=0),
        "fail_recall": recall_score(y_test, y_test_pred, pos_label=1, zero_division=0),
        "fail_f1": f1_score(y_test, y_test_pred, pos_label=1, zero_division=0),
    }

    # Build the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=[-1, 1])
    
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Pass (-1)", "Actual Fail (1)"],
        columns=["Predicted Pass (-1)", "Predicted Fail (1)"],
    )
            
    return cm.ravel()


    