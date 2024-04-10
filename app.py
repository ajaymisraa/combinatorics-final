import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.special import jv
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')

def download_data(tickers, start_date, end_date, file_path='stock_data.csv'):
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info("Stock data loaded from file")
    except FileNotFoundError:
        logging.info(f"Downloading stock data for tickers: {', '.join(tickers)}")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data.to_csv(file_path)
        logging.info("Stock data downloaded and saved to file")
    return data

def calculate_returns(data):
    returns = data.pct_change()
    logging.info("Returns calculated")
    return returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        logging.warning("Standard deviation of excess returns is zero. Sharpe ratio cannot be calculated.")
        return np.nan
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    logging.info(f"Sharpe ratio calculated: {sharpe_ratio:.2f}")
    return sharpe_ratio

def generate_gbm_prices(data, num_simulations=1000, time_steps=252, mu=0.1, sigma=0.2):
    logging.info("Generating simulated stock prices using GBM")
    log_returns = np.log(1 + data.pct_change())
    s0 = data.iloc[-1].values.reshape(1, -1)  # Reshape s0 to match the shape of simulated_prices
    dt = 1 / time_steps
    dW = np.random.normal(scale=np.sqrt(dt), size=(time_steps, num_simulations, len(data.columns)))
    simulated_prices = np.zeros((num_simulations, time_steps, len(data.columns)))
    simulated_prices[:, 0, :] = s0
    for t in range(1, time_steps):
        simulated_prices[:, t, :] = simulated_prices[:, t - 1, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW[t])
    logging.info(f"Simulated stock prices generated. Shape: {simulated_prices.shape}")
    return pd.DataFrame(simulated_prices.reshape(num_simulations * time_steps, len(data.columns)), columns=data.columns)

def generate_signals(data, short_window=50, long_window=100):
    logging.info(f"Generating trading signals with short window: {short_window} and long window: {long_window}")
    signals = pd.DataFrame(index=data.index, columns=data.columns)
    signals.iloc[:, :] = 0
    for col in data.columns:
        signals[col] = jv(0, data[col].rolling(window=short_window, min_periods=1).mean()) > jv(0, data[col].rolling(window=long_window, min_periods=1).mean())
    signals = signals.astype(int)
    logging.info(f"Trading signals generated. Shape: {signals.shape}")
    return signals

def optimize_parameters(data, risk_free_rate):
    logging.info("Optimizing parameters")
    def objective(params):
        short_window, long_window = params
        short_window = int(round(short_window))  # Round to the nearest integer
        long_window = int(round(long_window))  # Round to the nearest integer
        signals = generate_signals(data, short_window, long_window)
        portfolio_returns = (signals.shift(1) * data.pct_change()).sum(axis=1)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        return -sharpe_ratio  # Minimize the negative Sharpe ratio

    bounds = ((20, 100), (50, 200))  # Bounds for short_window and long_window
    initial_guess = (50, 100)
    result = minimize(objective, initial_guess, bounds=bounds, method='SLSQP')
    short_window = int(round(result.x[0]))  # Round the optimized short_window to the nearest integer
    long_window = int(round(result.x[1]))  # Round the optimized long_window to the nearest integer
    logging.info(f"Optimized parameters: Short window = {short_window}, Long window = {long_window}")
    return short_window, long_window

def backtest_strategy(data, signals, initial_capital=100000, transaction_cost=0.001):
    logging.info("Backtesting the trading strategy")
    portfolio_returns = (signals.shift(1) * data.pct_change()).sum(axis=1)
    portfolio_values = np.nan_to_num(initial_capital * (1 + portfolio_returns).cumprod(), posinf=1e10, neginf=-1e10)
    portfolio_values *= (1 - transaction_cost) ** (signals.diff().abs().sum(axis=1))
    logging.info("Backtesting completed")
    return portfolio_values

def train_model(data, signals):
    logging.info("Training the machine learning model")
    X = data.values.reshape(-1, data.shape[-1])  # Flatten the first two dimensions
    y = signals.values.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    logging.info(f"Model trained. Train score: {model.score(X_train, y_train):.2f}, Test score: {model.score(X_test, y_test):.2f}")
    return model

def generate_ml_signals(model, data):
    logging.info("Generating trading signals using the machine learning model")
    signals = model.predict(data.values)
    signals = pd.DataFrame(signals.reshape(-1, len(data.columns)), columns=data.columns)
    signals = signals.astype(int)
    logging.info(f"Trading signals generated using ML. Shape: {signals.shape}")
    return signals

def main():
    # Define the stock tickers and time period -- Removed FB & ABT due to data issues. 
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
               'PG', 'UNH', 'MA', 'DIS', 'ADBE', 'PYPL', 'NFLX', 'CMCSA', 'INTC', 'VZ',
               'CRM', 'PFE', 'T', 'KO', 'PEP', 'TMO', 'COST', 'MRK', 'CSCO',
               'WMT', 'AVGO', 'ACN', 'TXN', 'MDT', 'NKE', 'ABBV', 'LIN', 'NEE', 'LLY',
               'ORCL', 'DHR', 'LOW', 'UNP', 'HON', 'UPS', 'C', 'BA', 'CAT', 'GS']
    start_date = '2015-01-01'
    end_date = '2021-12-31'
    risk_free_rate = 0.02

    logging.info("Starting the trading bot")

    # Download historical stock data
    data = download_data(tickers, start_date, end_date)

    # Generate GBM simulated prices
    simulated_prices = generate_gbm_prices(data)

    # Optimize parameters
    short_window, long_window = optimize_parameters(simulated_prices, risk_free_rate)

    signals = generate_signals(simulated_prices, short_window, long_window)
    portfolio_values = backtest_strategy(simulated_prices, signals)

    # Generate trading signals with optimized parameters
    ml_model = train_model(simulated_prices, signals)

    # Generate trading signals using the machine learning model
    ml_signals = generate_ml_signals(ml_model, simulated_prices)

    # Backtest the trading strategy with ML signals
    ml_portfolio_values = backtest_strategy(simulated_prices, ml_signals)

    # Calculate Sharpe ratio for ML strategy
    ml_portfolio_returns = ml_portfolio_values.pct_change()
    ml_sharpe_ratio = calculate_sharpe_ratio(ml_portfolio_returns, risk_free_rate)

    logging.info(f"ML Strategy - Sharpe Ratio: {ml_sharpe_ratio:.2f}")
    
    if len(ml_portfolio_values) > 0:
        logging.info(f"ML Strategy - Cumulative Returns: {ml_portfolio_values.tail(1).values[0] / ml_portfolio_values.head(1).values[0] - 1:.2%}")
    else:
        logging.warning("ML Portfolio values series is empty. Cumulative returns cannot be calculated.")

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_values, label='Original Strategy')
    plt.plot(ml_portfolio_values, label='ML Strategy')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.title('Backtest Results')
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("Trading bot completed")


if __name__ == '__main__':
    main()