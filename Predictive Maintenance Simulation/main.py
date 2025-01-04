import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib  # For saving the model

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

# Exploratory Data Analysis (EDA) - Visualizing Sensor Data
plt.figure(figsize=(10, 6))
plt.scatter(data['Sensor_1'], data['Sensor_2'], c=data['Failure'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Sensor_1')
plt.ylabel('Sensor_2')
plt.title('Sensor Data and Failures')
plt.colorbar(label='Failure (0 = No, 1 = Yes)')
plt.show()

# Splitting Data
X = data[['Sensor_1', 'Sensor_2', 'Sensor_3']]
y = data['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling - Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Random Forest Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Making Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
print("\nCross-Validation Results (Accuracy per fold):")
print(cross_val_results)
print("Average Cross-Validation Accuracy: {:.2f}".format(np.mean(cross_val_results)))

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)

# Feature Importance Visualization
importances = model.feature_importances_
plt.bar(X.columns, importances, color='skyblue')
plt.title('Feature Importance')
plt.show()

# Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Save the trained model for future use
joblib.dump(model, 'random_forest_model.pkl')

# Load the model (optional)
# loaded_model = joblib.load('random_forest_model.pkl')
