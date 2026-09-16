# Creditcard_fraud_detection

## Final Model Evaluation

The final CardGuard XGBoost model was evaluated on the raw test dataset
after applying the same feature engineering process used during model
development.
The classification threshold was selected using the validation dataset.
A threshold of 0.65 was selected based on F1-score and was locked before
the final test evaluation.

### Test Dataset
- Total transactions: 555,719
- Legitimate transactions: 553,574
- Fraudulent transactions: 2,145
### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.81% |
| Fraud Precision | 69.85% |
| Fraud Recall | 88.67% |
| Fraud F1-Score | 78.14% |
| PR-AUC | 90.43% |
| ROC-AUC | 99.73% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 552,753 | 821 |
| Actual Fraud | 243 | 1,902 |

The model correctly detected 1,902 of the 2,145 fraudulent transactions,
resulting in a fraud recall of 88.67%.
The model produced 821 false positives and missed 243 fraudulent
transactions.
Because the dataset is highly imbalanced, precision, recall, F1-score,
PR-AUC, and ROC-AUC are considered more informative for evaluating the
fraud detection model than accuracy alone.
