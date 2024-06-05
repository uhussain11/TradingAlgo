import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Fetch the historical data
ticker = 'MES=F'
start_date = datetime.datetime(2020, 1, 1)
df = yf.download(ticker, start=start_date)

# Normalize column names to lowercase
df.columns = df.columns.str.lower()

# Define candlestick pattern recognition functions
def is_hammer(open, high, low, close):
    body = abs(close - open)
    candle_range = high - low
    lower_shadow = min(open, close) - low
    upper_shadow = high - max(open, close)
    return lower_shadow > 2.5 * body and upper_shadow < 0.5 * body and candle_range > 4 * body

def is_inverted_hammer(open, high, low, close):
    body = abs(close - open)
    candle_range = high - low
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return upper_shadow > 2.5 * body and lower_shadow < 0.5 * body and candle_range > 4 * body

def is_shooting_star(open, high, low, close):
    body = abs(close - open)
    candle_range = high - low
    upper_shadow = high - max(open, close)
    lower_shadow = min(open, close) - low
    return upper_shadow > 2.5 * body and lower_shadow < 0.5 * body and body < candle_range / 4

def is_bullish_engulfing(previous_open, previous_close, open, close):
    engulfing_body = abs(close - open)
    previous_body = abs(previous_close - previous_open)
    return previous_close < previous_open and close > open and engulfing_body > 1.5 * previous_body

def is_bearish_engulfing(previous_open, previous_close, open, close):
    engulfing_body = abs(close - open)
    previous_body = abs(previous_close - previous_open)
    return previous_close > previous_open and close < open and engulfing_body > 1.5 * previous_body

def is_doji(open, close, high, low):
    body = abs(close - open)
    candle_range = high - low
    return body / candle_range < 0.05 and candle_range > 0.01

# Calculate EMA and VWAP
df.dropna(inplace=True)  # Drop any rows with NaN values to avoid errors in EMA or VWAP calculation
window_length = 14
df['ema'] = EMAIndicator(close=df['close'], window=window_length).ema_indicator()
df['vwap'] = df.groupby(df.index.date).apply(lambda x: VolumeWeightedAveragePrice(high=x['high'], low=x['low'], close=x['close'], volume=x['volume']).volume_weighted_average_price()).values.flatten()

# Detect significant price levels
order = 20
df['resistance'] = df.iloc[argrelextrema(df['close'].values, np.greater_equal, order=order)[0]]['close']
df['support'] = df.iloc[argrelextrema(df['close'].values, np.less_equal, order=order)[0]]['close']

# Generate trading signals
df['hammer'] = df.apply(lambda row: is_hammer(row['open'], row['high'], row['low'], row['close']), axis=1)
df['inverted_hammer'] = df.apply(lambda row: is_inverted_hammer(row['open'], row['high'], row['low'], row['close']), axis=1)
df['shooting_star'] = df.apply(lambda row: is_shooting_star(row['open'], row['high'], row['low'], row['close']), axis=1)
df['bullish_engulfing'] = df.apply(lambda row: is_bullish_engulfing(row['open'], row['close'], row['open'], row['close']), axis=1)
df['bearish_engulfing'] = df.apply(lambda row: is_bearish_engulfing(row['open'], row['close'], row['open'], row['close']), axis=1)
df['doji'] = df.apply(lambda row: is_doji(row['open'], row['close'], row['high'], row['low']), axis=1)

# Define the target variable
df['future_close_higher'] = df['close'].shift(-1) > df['close']
df['future_close_higher'] = df['future_close_higher'].astype(int)
X = df[['ema', 'vwap', 'hammer', 'inverted_hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 'doji']]
y = df['future_close_higher']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an imputer object that fills with the median
imputer = SimpleImputer(strategy='median')

# Fit on the training data and transform it
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train the model and evaluate it using cross-validation
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
scores = cross_val_score(model, X_train, y_train, cv=5)

# Print model accuracy and cross-validation scores
print(f"Model Accuracy: {accuracy}")
print(f"Cross-validation Scores: {scores.mean()}")

# Simulation of the trading based on model predictions
initial_capital = 100000
positions = pd.DataFrame(index=df.index).fillna(0.0)
positions['MES'] = predictions  # Using model predictions for trading signals
portfolio = pd.DataFrame(index=df.index).fillna(0.0)
portfolio['holdings'] = positions.multiply(df['close'], axis=0)
portfolio['cash'] = initial_capital - (positions.diff().multiply(df['close'], axis=0)).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# Visualization of the equity curve
plt.figure(figsize=(10, 5))
plt.plot(portfolio['total'], label='Portfolio Value')
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Total Portfolio Value')
plt.legend()
plt.show()

# Print total return, annualized return, maximum drawdown, and Sharpe ratio
total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
annualized_return = ((1 + total_return) ** (252 / len(portfolio))) - 1
rolling_max = portfolio['total'].cummax()
daily_drawdown = (portfolio['total'] / rolling_max)

# Calculate and print the maximum drawdown
daily_drawdown = (portfolio['total'] / rolling_max) - 1
max_drawdown = daily_drawdown.min()
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

# Calculate and print the Sharpe ratio
risk_free_rate = 0.01  # Assuming a hypothetical risk-free rate of 1%
sharpe_ratio = ((portfolio['returns'].mean() - risk_free_rate / 252) / portfolio['returns'].std()) * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Additional Visualization: Drawdowns
plt.figure(figsize=(10, 5))
plt.fill_between(daily_drawdown.index, daily_drawdown.values, color='red', step='post', alpha=0.4)
plt.title('Drawdown over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.show()

# Risk Management and Execution Logic
def execute_trade_with_risk_management(entry_point, trade_direction, account_balance, risk_per_trade=0.01, risk_to_reward_ratio=2):
    trade_size = account_balance * risk_per_trade
    stop_loss = 5  # Risking 5 points per trade
    take_profit = stop_loss * risk_to_reward_ratio
    
    stop_loss_price = entry_point - stop_loss if trade_direction == 'long' else entry_point + stop_loss
    take_profit_price = entry_point + take_profit if trade_direction == 'long' else entry_point - take_profit
    
    print(f"Executing {'buy' if trade_direction == 'long' else 'sell'} order at {entry_point}, Stop Loss: {stop_loss_price}, Take Profit: {take_profit_price}, Trade Size: {trade_size}")
    # This function is a placeholder. Integrate with your brokerage's API to execute the order with specified stop loss and take profit.
