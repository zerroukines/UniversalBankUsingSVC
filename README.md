# SVM Classification Model for Personal Loan Prediction
# **building an SVM classification model to predict if a customer is likely to accept a personal loan**  Dataset source: https://www.kaggle.com/datasets/vinod00725/svm-classification/

This repository contains code for building and evaluating a Support Vector Machine (SVM) classification model to predict whether a customer is likely to accept a personal loan offered by a bank. The model is trained on a dataset containing various customer attributes.

## Dataset

The dataset used for this project contains the following columns:

- `ID`: Unique identifier for each customer.
- `Age`: Age of the customer.
- `Experience`: Years of professional experience.
- `Income`: Annual income of the customer.
- `ZIP Code`: ZIP code of the customer.
- `Family`: Number of family members.
- `CCAvg`: Average credit card spending per month.
- `Education`: Education level (e.g., 1 for undergraduate, 2 for graduate).
- `Mortgage`: Mortgage value.
- `Personal Loan`: Target variable (1 if customer accepted the personal loan, 0 otherwise).
- `Securities Account`: Whether the customer has a securities account (1 for yes, 0 for no).
- `CD Account`: Whether the customer has a certificate of deposit (CD) account (1 for yes, 0 for no).
- `Online`: Whether the customer uses online banking services (1 for yes, 0 for no).
- `CreditCard`: Whether the customer has a credit card (1 for yes, 0 for no).

## Requirements

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install scikit-learn pandas matplotlib seaborn

