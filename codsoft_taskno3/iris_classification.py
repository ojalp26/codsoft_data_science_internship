
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "C:/codsoft_taskno3/IRIS.csv"
data = pd.read_csv(file_path)
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
else:
    print("Error: File not found. Please check the path.")

print(data.head())

print("\nMissing values in each column:")
print(data.isnull().sum())

sns.pairplot(data, hue='species', palette='Set1', diag_kind='kde')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = data.drop(columns=['species'])
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_importance = model.feature_importances_
plt.barh(X.columns, feature_importance, color='teal')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Iris Dataset")
plt.show()
