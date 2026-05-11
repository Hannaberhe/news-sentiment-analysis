# %% [markdown]
# # Task 2: Stock Price Analysis & Technical Indicators
# ## KAIM Week 1 - Quantitative Analysis
# **Student:** Hanna Berhe

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# %%
# Load stock data
df = pd.read_csv('data/raw/stock_prices.csv')
df.columns = [col.strip().title() for col in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Trading days: {len(df)}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Handle missing values
df = df.fillna(method='ffill').fillna(method='bfill')
print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")

# %%
# Compute technical indicators
df['SMA_20'] = calculate_sma(df['Close'], 20)
df['SMA_50'] = calculate_sma(df['Close'], 50)
df['EMA_20'] = calculate_ema(df['Close'], 20)
df['RSI'] = calculate_rsi(df['Close'])
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])

print("Technical indicators computed successfully!")

# %% [markdown]
# ## 1. Price with Moving Averages

# %%
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Close'], label='Close Price', color='black', linewidth=1.5)
ax.plot(df['Date'], df['SMA_20'], label='SMA 20', color='blue', alpha=0.7)
ax.plot(df['Date'], df['SMA_50'], label='SMA 50', color='orange', alpha=0.7)
ax.plot(df['Date'], df['EMA_20'], label='EMA 20', color='red', linestyle='--', alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.set_title('Stock Price with Moving Averages')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/moving_averages.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 2. RSI Analysis

# %%
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df['Date'], df['RSI'], color='purple', linewidth=1.5)
ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
ax.fill_between(df['Date'], 70, df['RSI'], where=(df['RSI'] > 70), color='red', alpha=0.1)
ax.fill_between(df['Date'], 30, df['RSI'], where=(df['RSI'] < 30), color='green', alpha=0.1)
ax.set_xlabel('Date')
ax.set_ylabel('RSI')
ax.set_title('Relative Strength Index (RSI)')
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/rsi_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. MACD Analysis

# %%
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df['Date'], df['MACD'], label='MACD', color='blue', linewidth=1.5)
ax.plot(df['Date'], df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1.5)
colors = ['green' if x >= 0 else 'red' for x in df['MACD_Hist']]
ax.bar(df['Date'], df['MACD_Hist'], color=colors, alpha=0.5, label='Histogram')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('MACD')
ax.set_title('MACD (Moving Average Convergence Divergence)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/macd_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Bollinger Bands

# %%
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Date'], df['Close'], label='Close Price', color='black', linewidth=1.5)
ax.plot(df['Date'], df['BB_Upper'], label='Upper Band', color='red', linestyle='--', alpha=0.7)
ax.plot(df['Date'], df['BB_Middle'], label='Middle Band (SMA)', color='blue', linestyle='--', alpha=0.7)
ax.plot(df['Date'], df['BB_Lower'], label='Lower Band', color='green', linestyle='--', alpha=0.7)
ax.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.set_title('Bollinger Bands (20-day, 2 std)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/bollinger_bands.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Summary
# 
# 1. **Moving Averages**: SMA 20/50 and EMA 20 computed and visualized
# 2. **RSI**: Overbought/oversold levels identified at 70/30
# 3. **MACD**: Signal line crossovers indicate momentum changes
# 4. **Bollinger Bands**: Volatility bands show price containment
# 5. **Next**: Combine with sentiment analysis for prediction model
