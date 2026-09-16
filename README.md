# Creditcard_fraud_detection

## Final Model Performance

The final XGBoost model was evaluated on 555,719 unseen transactions,
including 2,145 fraudulent transactions.

The classification threshold of 0.65 was selected using the validation
dataset based on F1-score and locked before final testing.

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

| Actual Legitimate | 552,753 | 821 |
| Actual Fraud | 243 | 1,902 |
