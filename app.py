import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Helper function to calculate accuracy metrics
def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape

# Load the dataset
file_path = 'preprocessed_wpi_data1.csv'  # Update the path as needed
df = pd.read_csv(file_path)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar - Select COMM_NAME and input
st.sidebar.title('Commodity WPI Prediction')
comm_name = st.sidebar.selectbox('Select the commodity you want to analyze:', df['COMM_NAME'].unique())
forecast_steps = st.sidebar.slider('Select number of months to forecast:', 12, 60, 12)

# Filter and prepare data for the specific commodity
df_commodity = df[df['COMM_NAME'] == comm_name].sort_values('Date')
if df_commodity.empty:
    st.error(f"No data found for commodity {comm_name}")
    st.stop()

# Set 'Date' as index for time series
df_commodity.set_index('Date', inplace=True)
ts = df_commodity['WPI']

### General Dataset Statistics ###
st.header("General Dataset Statistics")
st.write(df.describe())
st.write(df.dtypes)

### EDA ###

# Temporal Analysis
st.header("Temporal Analysis")
st.write("WPI Over Time")
plt.figure(figsize=(10, 6))
plt.plot(df_commodity.index, df_commodity['WPI'], label='WPI')
plt.xlabel('Date')
plt.ylabel('WPI')
plt.title(f'WPI Over Time for {comm_name}')
plt.legend()
st.pyplot(plt.gcf())

# Seasonal Decomposition
st.write("Seasonal Decomposition")
decomposition = seasonal_decompose(ts, model='additive', period=12)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
decomposition.observed.plot(ax=ax1, legend=False)
ax1.set_ylabel('Observed')
decomposition.trend.plot(ax=ax2, legend=False)
ax2.set_ylabel('Trend')
decomposition.seasonal.plot(ax=ax3, legend=False)
ax3.set_ylabel('Seasonal')
decomposition.resid.plot(ax=ax4, legend=False)
ax4.set_ylabel('Residual')
plt.tight_layout()
st.pyplot(fig)

# Categorical Analysis
st.header("Categorical Analysis")
st.write("Top 10 Commodities by Count")
top_commodities = df['COMM_NAME'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_commodities.index, y=top_commodities.values)
plt.xticks(rotation=90)
plt.title('Top 10 Commodities by Count')
st.pyplot(plt.gcf())

st.write("Distribution of WPI for Top 10 Commodities")
top_commodities_data = df[df['COMM_NAME'].isin(top_commodities.index)]
plt.figure(figsize=(10, 6))
sns.boxplot(data=top_commodities_data, x='COMM_NAME', y='WPI')
plt.xticks(rotation=90)
plt.title('Distribution of WPI for Top 10 Commodities')
st.pyplot(plt.gcf())

# Numerical Analysis
st.header("Numerical Analysis")
st.write("Distribution of WPI")
plt.figure(figsize=(10, 6))
sns.histplot(df['WPI'], bins=30, kde=True)
plt.title('Distribution of WPI')
st.pyplot(plt.gcf())

st.write("Distribution of COMM_WT")
plt.figure(figsize=(10, 6))
sns.histplot(df['COMM_WT'], bins=30, kde=True)
plt.title('Distribution of COMM_WT')
st.pyplot(plt.gcf())

st.write("Correlation Matrix")
correlation = df[['WPI', 'COMM_WT']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix')
st.pyplot(plt.gcf())

### Forecasting ###

# ARIMA Model
st.header("ARIMA Model")

# Fit the ARIMA model (before preprocessing)
model = auto_arima(ts, start_p=0, start_q=0, max_p=5, max_q=5, m=12,
                   start_P=0, seasonal=True, d=1, D=1, trace=False,
                   error_action='ignore', suppress_warnings=True, stepwise=True)

# Train the ARIMA model
arima_model = ARIMA(ts, order=model.order)
results = arima_model.fit()

# Predict and calculate performance before preprocessing
forecast = results.forecast(steps=forecast_steps)
arima_rmse, arima_mae, arima_mape = calculate_metrics(ts[-forecast_steps:], forecast[:forecast_steps])

# Display ARIMA metrics
st.write(f"ARIMA Before Preprocessing: RMSE={arima_rmse}, MAE={arima_mae}, MAPE={arima_mape}")

# Plot ARIMA predictions
st.write("ARIMA Predictions")
plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts, label='Actual')
plt.plot(pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='M')[1:], forecast, label='ARIMA Forecast', color='red')
plt.title(f'ARIMA Forecast for Commodity (COMM_NAME: {comm_name}) WPI')
plt.legend()
st.pyplot(plt.gcf())

# Random Forest Model
st.header("Random Forest Model")

# Create lag features for the Random Forest model (e.g., 1-month, 3-month lag)
df_commodity['WPI_Lag_1'] = df_commodity['WPI'].shift(1)
df_commodity['WPI_Lag_3'] = df_commodity['WPI'].shift(3)

# Drop rows with NaN values created by lagging
df_commodity.dropna(inplace=True)

# Prepare features and target variable
X = df_commodity[['WPI_Lag_1', 'WPI_Lag_3']]
y = df_commodity['WPI']

# Split data into training and testing sets (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions using Random Forest
rf_predictions = rf_model.predict(X_test)

# Calculate RMSE, MAE, and MAPE for Random Forest
rf_rmse, rf_mae, rf_mape = calculate_metrics(y_test, rf_predictions)

# Display Random Forest metrics
st.write(f"Random Forest Before Preprocessing: RMSE={rf_rmse}, MAE={rf_mae}, MAPE={rf_mape}")

# Plot Random Forest predictions
st.write("Random Forest Predictions")
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, rf_predictions, label='Random Forest Predictions', color='red')
plt.title(f'Random Forest Predictions for Commodity (COMM_NAME: {comm_name}) WPI')
plt.legend()
st.pyplot(plt.gcf())

# LSTM Model
st.header("LSTM Model")

# Normalize WPI column for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
df_commodity['WPI_scaled'] = scaler.fit_transform(df_commodity[['WPI']])

# Prepare the data for LSTM (using a sliding window approach)
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 10  # Number of previous time steps to consider
X_lstm, y_lstm = create_dataset(df_commodity['WPI_scaled'].values, time_step)
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)  # LSTM expects input as [samples, time steps, features]

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32)

# Make predictions using LSTM
lstm_predictions = lstm_model.predict(X_lstm)

# Inverse scale the predictions and actual data
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_lstm_actual = scaler.inverse_transform(y_lstm.reshape(-1, 1))

# Calculate RMSE, MAE, and MAPE for LSTM
lstm_rmse, lstm_mae, lstm_mape = calculate_metrics(y_lstm_actual, lstm_predictions)

# Display LSTM metrics
st.write(f"LSTM: RMSE={lstm_rmse}, MAE={lstm_mae}, MAPE={lstm_mape}")

# Plot LSTM predictions
st.write("LSTM Predictions")
plt.figure(figsize=(10, 6))
plt.plot(df_commodity.index[time_step + 1:], y_lstm_actual, label='Actual')
plt.plot(df_commodity.index[time_step + 1:], lstm_predictions, label='LSTM Predictions', color='red')
plt.title(f'LSTM Predictions for Commodity (COMM_NAME: {comm_name}) WPI')
plt.legend()
st.pyplot(plt.gcf())

# Display Model Comparison
st.header("Model Performance Comparison")

st.write(f"ARIMA: RMSE={arima_rmse}, MAE={arima_mae}, MAPE={arima_mape}")
st.write(f"Random Forest: RMSE={rf_rmse}, MAE={rf_mae}, MAPE={rf_mape}")
st.write(f"LSTM: RMSE={lstm_rmse}, MAE={lstm_mae}, MAPE={lstm_mape}")