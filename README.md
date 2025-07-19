# Titanic Survival Prediction

This project focuses on predicting the survival of passengers from the Titanic dataset using various machine learning algorithms. It includes data preprocessing, model training, evaluation, and a comparison of different classification models.

## Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- File used: `train.csv`

## Objective

The goal is to build a model that can predict whether a passenger survived based on features like age, gender, ticket class, fare, and others.

## Features Used

- Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Sex: Gender (converted to numerical values)
- Age: Age in years (missing values handled)
- SibSp: Number of siblings or spouses aboard
- Parch: Number of parents or children aboard
- Fare: Ticket fare
- Embarked: Port of embarkation (converted to numerical)

## Models Used

The following machine learning models were used and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Linear Regression (used as a classifier with thresholding)

## Evaluation Metrics

Each model is evaluated using the following metrics:
- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)

## Visualizations

Model performance is compared using a bar chart showing the accuracy scores.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/shaktivishakan/titanic-survival-prediction.git
