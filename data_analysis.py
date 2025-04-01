import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv(r'C:\Users\onlys\Documents\GitHub\Machine-Failure-Model\data\data1.csv', skiprows=1)  # Skip the first row which contains the script name

# Clean column names to remove any extra commas
data.columns = [col.strip() for col in data.columns]

# Basic statistics
print("Dataset shape:", data.shape)
print("\nFailure rate:", data['fail'].mean())
print("\nBasic statistics for all variables:")
print(data.describe())

# Check correlation between variables
plt.figure(figsize=(12, 10))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()

# Analyze failures by temperature
plt.figure(figsize=(10, 6))
sns.countplot(x='Temperature', hue='fail', data=data)
plt.title('Failures by Temperature')
plt.xlabel('Temperature')
plt.ylabel('Count')

# Analyze VOC impact on failures
plt.figure(figsize=(10, 6))
sns.boxplot(x='fail', y='VOC', data=data)
plt.title('VOC Levels by Failure Status')

# Prepare data for modeling
X = data.drop('fail', axis=1)
y = data['fail']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Predicting Machine Failures')
plt.tight_layout()

# Analyze specific conditions leading to failures
failure_data = data[data['fail'] == 1]
print("\nStatistics for failed machines:")
print(failure_data.describe())

# Check if high VOC combined with high Temperature leads to more failures
plt.figure(figsize=(10, 6))
high_voc = data['VOC'] > 4
high_temp = data['Temperature'] > 8
sns.countplot(x=high_voc & high_temp, hue='fail', data=data)
plt.title('Failures when both VOC > 4 and Temperature > 8')
plt.xlabel('High VOC & High Temperature')
plt.xticks([0, 1], ['No', 'Yes'])

# Plot failure rate by VOC level
plt.figure(figsize=(10, 6))
failure_rates = data.groupby('VOC')['fail'].mean()
sns.barplot(x=failure_rates.index, y=failure_rates.values)
plt.title('Failure Rate by VOC Level')
plt.xlabel('VOC Level')
plt.ylabel('Failure Rate')

plt.tight_layout()