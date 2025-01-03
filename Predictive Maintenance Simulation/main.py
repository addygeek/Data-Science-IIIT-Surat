import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Simulating Sensor Data
np.random.seed(42)
n_samples = 1000

# Features: Random values simulating sensor readings
sensor_1 = np.random.normal(50, 10, n_samples)
sensor_2 = np.random.normal(100, 20, n_samples)
sensor_3 = np.random.normal(75, 15, n_samples)

# Target: Binary label indicating equipment failure
failure = (sensor_1 + sensor_2 + sensor_3) / 3 + np.random.normal(0, 5, n_samples)
failure = (failure > 120).astype(int)

# Creating a DataFrame
data = pd.DataFrame({
    'Sensor_1': sensor_1,
    'Sensor_2': sensor_2,
    'Sensor_3': sensor_3,
    'Failure': failure
})

# Splitting Data
X = data[['Sensor_1', 'Sensor_2', 'Sensor_3']]
y = data['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Random Forest Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Making Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
plt.bar(X.columns, importances, color='skyblue')
plt.title('Feature Importance')
plt.show()

# Visualizing Sensor Data and Failures
plt.scatter(data['Sensor_1'], data['Sensor_2'], c=data['Failure'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Sensor_1')
plt.ylabel('Sensor_2')
plt.title('Sensor Data and Failures')
plt.colorbar(label='Failure (0 = No, 1 = Yes)')
plt.show()
