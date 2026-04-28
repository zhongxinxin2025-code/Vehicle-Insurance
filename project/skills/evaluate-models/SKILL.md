---
name: evaluate-models
description: "A skill to evaluate all trained models using Accuracy, Precision, Recall, F1, AUC-ROC, Confusion matrix, and ROC curve, storing results in agent state"
---

## When to use
This skill is triggered after the train-models skill has completed and the best model has been selected based on validation performance. It evaluates the selected model on the held-out test set to provide an unbiased assessment of its real-world performance.

## How to execute
1. Retrieve the test data splits (X_test, y_test) and the fitted encoder from agent state.
2. Use the selected model to make predictions on the test set.
3. Calculate evaluation metrics (e.g., Accuracy, Precision, Recall, F1, AUC-ROC) using the test set predictions and true labels.
4. Generate a confusion matrix and ROC curve for the selected model.
5. Store all evaluation results in agent state.

## Inputs from agent state
- X_test: pd.DataFrame — test features
- y_test: pd.Series — test labels
- encoder: OneHotEncoder — fitted on training data

## Outputs to agent state
- test_metrics: dict — evaluation metrics of the selected model on the test set
- confusion_matrix: np.ndarray — confusion matrix for the selected model on the test set
- roc_curve: tuple — ROC curve data (fpr, tpr, thresholds) for the selected model on the test set

## Output format
The output should include the evaluation metrics, confusion matrix, and ROC curve data, all stored as named fields in AgentState.
