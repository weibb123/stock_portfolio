import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

st.title("Stock Portfolio Analysis")

# get current time and start time
def check_stock(stock: str, begin_time: str):
    today = date.today()
    try:
        # Download stock data
        stock_data = yf.download(stock, start=begin_time, end=today)
        return stock_data
    except:
        return "error download data, make sure you type correctly"

def plot_stock(data, symbol):
    # plot using regression
    try:
        # Calculate volume profile
        price_bins = np.linspace(data['Low'].min(), data['High'].max(), 100)
        volume_profile = []

        for i in range(len(price_bins)-1):
            bin_mask = (data['Close'] > price_bins[i]) & (data['Close'] <= price_bins[i+1])
            volume_profile.append(data['Volume'][bin_mask].sum())

        # Estimating support and resistance
        current_price = data['Close'].iloc[-1]
        support_idx = np.argmax(volume_profile[:np.digitize(current_price, price_bins)])
        resistance_idx = np.argmax(volume_profile[np.digitize(current_price, price_bins):]) + np.digitize(current_price, price_bins)

        support_price = price_bins[support_idx]
        resistance_price = price_bins[resistance_idx]

        # Plotting
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 1]})
        ax1.plot(data['Close'], label="Close Price")
        ax1.axhline(y=support_price, color='g', linestyle='--', label='Support')
        ax1.axhline(y=resistance_price, color='r', linestyle='--', label='Resistance')
        ax1.legend()
        ax1.set_title(f'{symbol} Price Data')
        ax2.barh(price_bins[:-1], volume_profile, height=(price_bins[1] - price_bins[0]), color='blue', edgecolor='none')
        ax2.set_title('Volume Profile')

        st.pyplot(fig)

        st.write(f"Estimated Support Price: {support_price:.2f}")
        st.write(f"Estimated Resistance Price: {resistance_price:.2f}")
    except:
        print("Error plotting")

def assess_risk(stock, initial_investment, forecast_days, desired_return):
    num_simulations = 1000
    confidence_level = 0.95
    forecast = int(forecast_days)
    invest = int(initial_investment)

    df = yf.Ticker(stock)
    data = df.history(period='3mo')
    df_data = pd.DataFrame(data)

    # Calculate daily returns
    daily_returns = df_data["Close"].pct_change().dropna()

    # Calculate the average daily return and volatility
    average_daily_return = daily_returns.mean()
    volatility = daily_returns.std()

    # Simulating future returns
    simulated_end_returns = np.zeros(num_simulations)
    for i in range(num_simulations):
        random_returns = np.random.normal(average_daily_return, volatility, forecast)
        cumulative_return = np.prod(1 + random_returns)
        simulated_end_returns[i] = int(initial_investment) * cumulative_return

    # Calculate the final investment values
    final_investment_values = simulated_end_returns
    sorted_returns = np.sort(final_investment_values)
    index_at_var = int((1-confidence_level) * num_simulations)
    var = int(initial_investment) - sorted_returns[index_at_var]
    conditional_var = int(initial_investment) - sorted_returns[:index_at_var].mean()

    num_success = np.sum(final_investment_values >= invest * (1 + int(desired_return)))
    probability_of_success = num_success / num_simulations

    st.write(f"Probability of achieving at least a {int(desired_return)}% return: {probability_of_success*100:.2f}%")
    st.write(f"we are 95% that the worst you can do is to gain/lose: ${var:,.2f}")
    st.write(f"On average, in those worst moments, Expected Tail Loss (Conditional VaR): ${conditional_var:,.2f}")

start = st.date_input("When you want to start?", value=None)
st.write("Your start date:", start)

stock = st.text_input("Stock you want to look up")

st.write("Stock picked:", stock)

# Risk management
initial_money = st.text_input("pick initial investment")
forecast = st.text_input("forecast into future?")
desire_return = st.text_input("desire percent return")

button = st.button("get info", type="primary")
if button:
    
    data = check_stock(stock, start)
    plot_stock(data, stock)
    assess_risk(stock, initial_money, forecast, desire_return)






