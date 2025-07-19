import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset

df = pd.read_csv('train.csv')

# basic eda

print("First 5 rows of data :\n",df.head())
print("|n missing values:\n", df.isnull().sum())

# fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# convert categoriacl to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# knn
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# naive bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
y_pred_lin = [1 if i > 0.5 else 0 for i in y_pred_lin]

# Evaluation function
def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluation

evaluate(log_model, "Logistic Regression")
evaluate(knn_model, "KNN")
evaluate(nb_model, "Naive Bayes")

# evaluate linear regression seperately

print("\n--- Linear Regression(converted to classifier) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lin))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lin))
print("Classification Report:\n", classification_report(y_test, y_pred_lin))

# Visualize model performance
models = ['Logistic', 'Knn', 'Naive Bayes', 'Linear Regression']
accuracies = [
    accuracy_score(y_test, log_model.predict(X_test)),
    accuracy_score(y_test, knn_model.predict(X_test)),
    accuracy_score(y_test, nb_model.predict(X_test)),
    accuracy_score(y_test, y_pred_lin)  
]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance')
plt.ylim(0,1)
plt.show()
