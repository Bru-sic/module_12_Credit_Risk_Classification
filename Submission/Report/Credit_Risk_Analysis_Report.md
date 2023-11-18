# Module 12 Report Template

## Overview
### Purpose of the analysis
The purpose of the analysis was to assess the credit worthiness of borrowers comparing results from different machine learning algorithms. The challenge was attempting to make a binary decision from a dataset that had healthy loans significantly outnumbering the risky loans, as is typically the case in real world scenarios. This imbalance in the data causes some machine learning models to skew results towards the majority class, in this case the healthy loans. Adjusting the model to compensate for the skew then became part of the challenge.

### Information analysed and objectives
The dataset was provided in the file "lending_data.csv", containing data typically used in measuring and monitoring the credit risk of a loan.

The objective was to use machine learning to predict whether a loan was healthy or high-risk. Success was to be determined by comparing the predictions agaist known results (labels).

### Technical Information
The file consisted of 77,536 records in eight (8) columns as follows:
	* loan_size
  * interest_rate
  * borrower_income
  * debt_to_income
  * num_of_accounts
  * derogatory_marks
  * total_debt
  * loan_status (value of 0 means healthy loan; 1 means high risk)

Other than loan_status, a description for the other columns was not available, however these are fairly self-explanatory.

The loan_status was used as the "label" (known result). The remaining columns were used as "features" (input to the machine learning model)

The aggregated total number of records by loan_status was 75,036 for the Healthy loan group (loan_status=0) and  2,500 for the High-risk loan group  (loan_status=1), therefore significantly imbalanced with the majority being healthy, as would normally be expected in relation to lending and credit management overall.


### Machine learning process
1. Machine Learning Model 1: Using the original data on the **sklearn Logistic Regression** model with the Limited Memory Broyden–Fletcher–Goldfarb–Shanno algorithm ("lbfgs") solver as a first round of analysis. 

1. Machine Learning Model 2: Using the **imbalanced-learn RandomOverSampler** module to resample the data given the significant imbalance, and then training  **sklearn Logistic Regression** model with the "lbfgs" solver again but using the resampled data as input for a second round of analysis.1. 



## Results
* Machine Learning Model 1 (LogisticRegression) Results:
  * Healthy loan group (loan_group=0)
    * Precision score: 100% - all positive predictions that the loan was healthy were correct.
    * Recall score: 99% - of all the actual loan samples, the majority were correctly classified as having a healthy loan status.
  * High-risk loan group (loan_group=1)
    * Precision score: 85% - a high proportion of positive predictions that the loan was high-risk were correct.
    * Recall score: 91%  - of all the actual loan samples, the majority were correctly classified as having a high-risk loan status 
  * Balanced accuracy score: 95% - the Recall from both loan groups was high and quite well balanced.


* Machine Learning Model 2 (Random Oversampler + LogisticRegression) Results:
  * Healthy loan group (loan_group=0)
    * Precision score: 100% - all positive predictions that the loan was healthy were correct.
    * Recall score: 99% - of all the actual loan samples, the majority were correctly classified as having a healthy loan status.
  * High-risk loan group (loan_group=1)
    * Precision score: 85% - a high proportion of positive predictions that the loan was high-risk were correct.
    * Recall score: 91%  - of all the actual loan samples, the majority were correctly classified as having a high-risk loan status 
  * Balanced accuracy score: 99% - the Recall from both loan groups was high and quite well balanced.


## Summary
* Both machine learning models performed well in being able to predit and distinguish most healthy loans high risk loans. In terms of F1-Score, Accuracy, Precision and Recall both models performed equally. However, the balanced accuracy score improved from 95% to 99%.

* Given the use case of needing to correctly detect high-risk loans, and that the dataset is likely to generally be largely unbalanced, it is worth applying other analysis tools to determine if there is scope for improvement.

## Definitions
* Accuracy score: ratio of correctly predicted obersations to the total number of observations. ie `(TP + TN) / (TP + TN + FP + FN)`.
* Balanced accuracy score: average of recall obtained on each class.
* F1-score: harmonic mean of the precision and recall scores. ie `(2 * Precision * Recall) / (Precision + Recall)`.
* FP: False Positive.   
* FN: False Negative.   
* Logistic Regression: a statistical method for predicting binary outcomes from data.
* Precision score: ratio of correctly predicted positive observations to the total predicted positive observations. ie `TP / (TP + FP)`
* Recall score: ratio of correctly predicted positive observations for that class. ie `TP / (TP + FN)`.
* TN: True Negative.
* TP: True Positive.