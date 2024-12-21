import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('C:/codsoft_taskno1/Titanic-Dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("C:/codsoft_taskno1/Titanic-Dataset.csv")

print(df.columns)
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'Embarked']

df = df.drop(columns=columns_to_drop)
df["Age"] = df["Age"].fillna(df["Age"].mean())
df = pd.get_dummies(df, columns=['Sex'])

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X = df.drop(columns="Survived")
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))
y.value_counts()

cm = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred))
target_names = ['Not Survived', 'Survived']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
model_2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
model_2.fit(X_train, y_train)

y_pred = model_2.predict(X_test)
accuracy = model_2.score(X_test, y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

from xgboost import XGBClassifier
model_3 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_3.fit(X_train, y_train)

y_pred = model_3.predict(X_test)
accuracy = model_3.score(X_test, y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=1)
model_4 = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, learning_rate=1.0, random_state=42)
model_4.fit(X_train, y_train)

y_pred = model_4.predict(X_test)
accuracy = model_4.score(X_test, y_test)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))
