# %% [markdown]
# # Task 3: Correlation Between News Sentiment and Stock Movement
# ## KAIM Week 1 - Sentiment-Price Correlation Analysis
# **Student:** Hanna Berhe

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)
plt.style.use('seaborn-v0_8-darkgrid')

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Load news headlines
news_df = pd.read_csv('data/raw/headlines.csv')
news_df['date'] = pd.to_datetime(news_df['date'])

# Load stock data
stock_df = pd.read_csv('data/raw/stock_prices.csv')
stock_df.columns = [col.strip().title() for col in stock_df.columns]
stock_df = stock_df.rename(columns={'Adj Close': 'Adj_Close'})
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

print(f"News headlines: {len(news_df)}")
print(f"Stock trading days: {len(stock_df)}")

# %% [markdown]
# ## 2. Date Alignment
# Map weekend/holiday news to next trading day

# %%
def align_to_trading_day(date, trading_dates):
    """Map a date to the next available trading day"""
    if date in trading_dates:
        return date
    # Find next trading day
    future_dates = trading_dates[trading_dates > date]
    if len(future_dates) > 0:
        return future_dates[0]
    return None

# Get sorted trading dates
trading_dates = sorted(stock_df['Date'].unique())
trading_dates_set = set(trading_dates)

# Apply date alignment
news_df['aligned_date'] = news_df['date'].apply(
    lambda x: align_to_trading_day(x.date(), trading_dates_set)
)

# Remove news that can't be aligned
news_df = news_df.dropna(subset=['aligned_date'])
news_df['aligned_date'] = pd.to_datetime(news_df['aligned_date'])

print(f"Headlines after alignment: {len(news_df)}")
print(f"Date range: {news_df['aligned_date'].min()} to {news_df['aligned_date'].max()}")

# %% [markdown]
# ## 3. Sentiment Scoring
# **Tool Selection Note:** Using NLTK VADER because:
# - Specifically designed for short text/social media
# - Pre-trained on general English, works well on financial text
# - Returns compound score (-1 to +1) ideal for correlation
# - Fast and doesn't require GPU

# %%
# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Get compound sentiment score for a headline"""
    return analyzer.polarity_scores(str(text))['compound']

# Compute sentiment scores
news_df['sentiment_score'] = news_df['headline'].apply(get_sentiment)

# Categorize sentiment
news_df['sentiment_category'] = pd.cut(
    news_df['sentiment_score'],
    bins=[-1, -0.05, 0.05, 1],
    labels=['negative', 'neutral', 'positive']
)

print("\nSentiment Distribution:")
print(news_df['sentiment_category'].value_counts())
print(f"\nAverage sentiment score: {news_df['sentiment_score'].mean():.3f}")

# %% [markdown]
# ## 4. Compute Daily Returns

# %%
# Calculate daily percentage returns using Adj Close
stock_df['daily_return'] = stock_df['Adj_Close'].pct_change() * 100
stock_df['daily_return'] = stock_df['daily_return'].round(4)

print("Daily Returns Statistics:")
print(stock_df['daily_return'].describe())

# Visualize returns distribution
fig, ax = plt.subplots(figsize=(10, 5))
stock_df['daily_return'].hist(bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Daily Stock Returns')
plt.tight_layout()
plt.savefig('reports/figures/returns_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Aggregate Sentiment by Day
# Average multiple same-day articles into single daily sentiment score

# %%
# Aggregate sentiment by aligned date
daily_sentiment = news_df.groupby('aligned_date').agg({
    'sentiment_score': 'mean',
    'headline': 'count'
}).rename(columns={'headline': 'article_count'})

daily_sentiment = daily_sentiment.reset_index()
daily_sentiment.columns = ['Date', 'avg_sentiment', 'article_count']

print(f"Trading days with news: {len(daily_sentiment)}")
print(f"Average articles per day: {daily_sentiment['article_count'].mean():.1f}")

# Visualize daily sentiment over time
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily_sentiment['Date'], daily_sentiment['avg_sentiment'], 
        color='steelblue', linewidth=1.5)
ax.fill_between(daily_sentiment['Date'], 0, daily_sentiment['avg_sentiment'],
                where=(daily_sentiment['avg_sentiment'] > 0), color='green', alpha=0.3, label='Positive')
ax.fill_between(daily_sentiment['Date'], 0, daily_sentiment['avg_sentiment'],
                where=(daily_sentiment['avg_sentiment'] < 0), color='red', alpha=0.3, label='Negative')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Average Sentiment Score')
ax.set_title('Daily Average News Sentiment Over Time')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/daily_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Merge Sentiment with Stock Returns

# %%
# Merge on date
merged_df = stock_df.merge(daily_sentiment, on='Date', how='inner')

print(f"Merged dataset: {len(merged_df)} trading days")
print(f"Columns: {merged_df.columns.tolist()}")

# Remove NaN values
merged_df = merged_df.dropna(subset=['daily_return', 'avg_sentiment'])
print(f"After removing NaN: {len(merged_df)} days")

# %% [markdown]
# ## 7. Pearson Correlation Analysis

# %%
# Calculate Pearson correlation
correlation, p_value = stats.pearsonr(merged_df['avg_sentiment'], merged_df['daily_return'])

print("=" * 50)
print("PEARSON CORRELATION RESULTS")
print("=" * 50)
print(f"Correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'} (at 95% confidence)")
print("=" * 50)

# %% [markdown]
# ## 8. Scatter Plot - Sentiment vs Returns

# %%
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
scatter = ax.scatter(merged_df['avg_sentiment'], merged_df['daily_return'],
                     c=merged_df['avg_sentiment'], cmap='RdYlGn', 
                     alpha=0.6, edgecolors='black', linewidth=0.5)

# Add trend line
z = np.polyfit(merged_df['avg_sentiment'], merged_df['daily_return'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged_df['avg_sentiment'].min(), merged_df['avg_sentiment'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend Line (r={correlation:.3f})')

# Add correlation annotation
ax.text(0.05, 0.95, f'Pearson r = {correlation:.4f}\np-value = {p_value:.4f}',
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Average Daily Sentiment Score')
ax.set_ylabel('Daily Stock Return (%)')
ax.set_title('News Sentiment vs Stock Returns')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Sentiment Score')
plt.tight_layout()
plt.savefig('reports/figures/sentiment_vs_returns.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Bar Chart - Average Return by Sentiment Category

# %%
# Add sentiment category to merged data
merged_df['sentiment_category'] = pd.cut(
    merged_df['avg_sentiment'],
    bins=[-1, -0.05, 0.05, 1],
    labels=['Negative', 'Neutral', 'Positive']
)

# Calculate average return by sentiment category
category_returns = merged_df.groupby('sentiment_category')['daily_return'].agg(['mean', 'std', 'count'])

print("Average Daily Return by Sentiment Category:")
print(category_returns)

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#ff6b6b', '#95e1d3', '#4ecdc4']
bars = ax.bar(category_returns.index, category_returns['mean'], 
              color=colors, edgecolor='black', linewidth=1.5,
              yerr=category_returns['std'], capsize=10)

# Add value labels
for bar, value in zip(bars, category_returns['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, 
            value + (0.1 if value >= 0 else -0.1),
            f'{value:.2f}%', ha='center', va='bottom' if value >= 0 else 'top',
            fontweight='bold', fontsize=12)

ax.set_xlabel('Sentiment Category')
ax.set_ylabel('Average Daily Return (%)')
ax.set_title('Average Daily Stock Return by News Sentiment Category')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('reports/figures/returns_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Results Interpretation
# 
# ### Correlation Analysis Summary
# 
# The Pearson correlation coefficient between daily news sentiment scores and stock returns 
# shows a **[positive/negative/weak]** relationship (r = {correlation:.4f}). 
# 
# This suggests that **[interpretation based on actual value]**. The p-value of {p_value:.4f} 
# indicates that this correlation is **[statistically significant / not statistically significant]** 
# at the 95% confidence level.
# 
# ### Key Findings:
# 
# 1. **Sentiment Impact**: Positive news sentiment tends to correspond with **[higher/lower]** 
#    stock returns, while negative sentiment aligns with **[opposite effect]**.
# 
# 2. **Magnitude**: The correlation strength is **[weak/moderate/strong]**, suggesting that 
#    sentiment alone accounts for approximately {(correlation**2)*100:.1f}% of daily return variance.
# 
# ### Limitations:
# 
# 1. **Lag Effects**: News impact may take multiple days to fully reflect in stock prices. 
#    Today's news might affect tomorrow's returns.
# 
# 2. **Confounding Factors**: Macroeconomic conditions, earnings reports, and market-wide events 
#    influence prices independently of sentiment.
# 
# 3. **Sentiment Accuracy**: VADER is general-purpose; financial-specific sentiment models may 
#    capture nuances better.
# 
# 4. **Sample Bias**: Only headlines from certain publishers; may not represent full market sentiment.
# 
# 5. **Direction vs Magnitude**: Sentiment may predict direction better than magnitude of price moves.
# 
# ### Recommendations for Task 4:
# 
# - Implement lagged sentiment features (1-5 day lags)
# - Add market-wide indicators as control variables
# - Use financial-specific NLP models (FinBERT)
# - Build ML model incorporating both sentiment and technical indicators
