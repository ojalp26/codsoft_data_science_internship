import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

advertising = pd.read_csv("C:/codsoft_taskno4/advertising.csv")

print(advertising.head())
print(advertising.shape)
print(advertising.info())
print(advertising.describe())

print("Missing Values (%):\n", advertising.isnull().sum() * 100 / advertising.shape[0])

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(data=advertising, y='TV', ax=axs[0], color='lightblue')
axs[0].set_title('TV')
sns.boxplot(data=advertising, y='Newspaper', ax=axs[1], color='lightgreen')
axs[1].set_title('Newspaper')
sns.boxplot(data=advertising, y='Radio', ax=axs[2], color='lightcoral')
axs[2].set_title('Radio')
plt.tight_layout()
plt.show()

sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

sns.heatmap(advertising.corr(), cmap="YlGnBu", annot=True)
plt.title("Correlation Matrix")
plt.show()

X = advertising[['TV', 'Newspaper', 'Radio']]
y = advertising['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

print(lr.summary())

plt.scatter(X_train['TV'], y_train, label='Actual data')
plt.plot(X_train['TV'], lr.predict(X_train_sm), color='red', label='Regression line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.title('Regression Line for TV')
plt.show()

y_train_pred = lr.predict(X_train_sm)
residuals = y_train - y_train_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=15, kde=True)
plt.title('Error Terms Distribution')
plt.xlabel('Residuals')
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_train_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 Score): {r2}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test['TV'], y_test, label='Actual data')
plt.plot(X_test['TV'], y_test_pred, color='red', label='Regression line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.title('Regression Line for TV (Test Data)')
plt.show()
