# Stock-Prediction-using-LSTM

# 📈 Apple Stock Price Prediction using LSTM

This project builds a deep learning model using an LSTM (Long Short-Term Memory) neural network to predict the closing prices of Apple Inc. (AAPL) stock. The model is trained on historical stock data and learns temporal patterns to make future price predictions.



## 📌 Overview

Stock market prices are influenced by historical trends and exhibit time series behavior. Recurrent Neural Networks (RNNs), especially LSTMs, are effective for capturing such temporal dependencies. This project applies an LSTM model to forecast Apple’s stock closing prices using historical data from Yahoo Finance.

---

## 📊 Data Source

- **Ticker**: AAPL (Apple Inc.)
- **Provider**: [Yahoo Finance](https://finance.yahoo.com/)
- **Time Range**: January 1, 2010 – January 1, 2023
- **Access Method**: Python `yfinance` API

---

## 🔧 Preprocessing & Cleaning

- Selected the `Close` price column for modeling.
- Computed **100-day** and **200-day** moving averages for trend visualization.
- Removed non-numeric columns and ensured no missing values for model input.
- Reshaped data into sequences of **60 time steps** to feed into the LSTM.

---

## ✂️ Train-Test Split

- Split the dataset into:
  - **Training set**: 70% of the data
  - **Testing set**: 30% of the data
- The split is done chronologically to preserve time series integrity.

---

## 🔢 MinMax Scaling

- Applied `MinMaxScaler` to normalize closing price values between **0 and 1**.
- This helps stabilize and accelerate LSTM training.

---

## 🧠 Model Architecture

Built using **Keras** with the following architecture:

```python
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=1))
'''

Two LSTM layers: Stacked to capture complex temporal dependencies.
	•	📤 One Dense output layer: Outputs a single predicted closing price.
	•	⚙️ Optimizer: Adam — efficient and widely used for time series tasks.
	•	📉 Loss Function: Mean Squared Error (MSE) — suitable for regression.


The model was trained on normalized 60-day historical price windows.
	•	📈 Input: Scaled sequences of past 60 days’ closing prices.
	•	🔁 Epochs: 50 (can be tuned for better performance).
	•	✅ Loss Monitoring: Training loss decreased steadily, indicating learning.

### 📊 Evaluation Metrics

| Dataset  | RMSE   |
|----------|--------|
| Training | 154.41 |
| Testing  | 213.42 |


🔍 Interpretation:
	•	The RMSE on the training data is lower than on the test data, which is expected.
	•	The gap between training and testing RMSE suggests some degree of overfitting.
	•	Despite this, the model still captures the overall trend and price trajectory of Apple stock.



