import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Sample Data: Monthly Sales Data
data = {
    "Month": [
        "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", 
        "2023-06", "2023-07", "2023-08", "2023-09", "2023-10"
    ],
    "Sales": [200, 210, 250, 270, 300, 310, 330, 350, 400, 420]
}
df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

print("Original Time Series Data:\n", df)

# Plot the Time Series
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Sales'], marker='o', label='Original Sales')
plt.title("Monthly Sales Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Train ARIMA Model
model = ARIMA(df['Sales'], order=(1, 1, 1))  # (p, d, q)
fitted_model = model.fit()
print("\nARIMA Model Summary:\n", fitted_model.summary())

# Forecast Future Sales
forecast_steps = 6  # Forecast for the next 6 months
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(df.index[-1] + pd.Timedelta(days=30), periods=forecast_steps, freq='MS')
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot Forecast vs Actual
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Sales'], marker='o', label='Original Sales')
plt.plot(forecast_series.index, forecast_series, marker='o', label='Forecast', linestyle='--')
plt.title("Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Calculate Error Metrics
y_true = df['Sales'][-forecast_steps:] if len(df) > forecast_steps else df['Sales']
y_pred = forecast[:len(y_true)]
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Save Results
forecast_series.to_csv("sales_forecast.csv", header=["Sales Forecast"])
print("\nForecast saved to 'sales_forecast.csv'.")
