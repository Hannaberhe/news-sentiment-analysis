# %% [markdown]
# # Task 1: Exploratory Data Analysis & NLP Analysis
# ## KAIM Week 1 - Predicting Stock Price Moves with News Sentiment
# **Student:** Hanna Berhe

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
# Load data
df = pd.read_csv('data/raw/headlines.csv')
print(f"Total headlines: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
df.head()

# %% [markdown]
# ## 1. Descriptive Statistics

# %%
# Headline length statistics
df['headline_length'] = df['headline'].str.len()
df['word_count'] = df['headline'].str.split().str.len()

print("=== HEADLINE STATISTICS ===")
print(f"Average length: {df['headline_length'].mean():.1f} characters")
print(f"Average words: {df['word_count'].mean():.1f} words")
print(f"Max length: {df['headline_length'].max()} characters")
print(f"Min length: {df['headline_length'].min()} characters")

# %% [markdown]
# ## 2. Publisher Analysis

# %%
# Extract domain from email-format publishers
df['publisher_domain'] = df['publisher'].apply(
    lambda x: re.search(r'@([\w.]+)', str(x)).group(1) 
    if re.search(r'@([\w.]+)', str(x)) else str(x)
)

# Top publishers
top_publishers = df['publisher_domain'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 6))
top_publishers.plot(kind='barh', color='skyblue', edgecolor='black')
ax.set_xlabel('Number of Articles')
ax.set_title('Top 10 Publishers by Article Count')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('reports/figures/publisher_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 10 Publishers:")
print(top_publishers)

# %% [markdown]
# ## 3. Publication Frequency Over Time

# %%
# Convert date and analyze publication patterns
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()

# Daily publication frequency
daily_volume = df.groupby(df['date'].dt.date).size()

fig, ax = plt.subplots(figsize=(12, 5))
daily_volume.plot(ax=ax, color='steelblue', linewidth=1.5)
ax.fill_between(daily_volume.index, daily_volume.values, alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Headlines')
ax.set_title('Daily News Publication Frequency')

# Highlight spikes
mean_volume = daily_volume.mean()
std_volume = daily_volume.std()
spikes = daily_volume[daily_volume > mean_volume + 2*std_volume]
ax.scatter(spikes.index, spikes.values, color='red', s=50, zorder=5, label='Spikes (>2 std)')
ax.legend()

plt.tight_layout()
plt.savefig('reports/figures/publication_frequency.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Average daily publications: {mean_volume:.1f}")
print(f"Number of spike days: {len(spikes)}")

# %% [markdown]
# ## 4. NLP Analysis - Keyword/Topic Extraction

# %%
# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_headline'] = df['headline'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['clean_headline'])

# Get top keywords
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])

# Visualize top keywords
fig, ax = plt.subplots(figsize=(10, 6))
words, counts = zip(*top_words.items())
ax.barh(range(len(words)), counts, color='lightcoral', edgecolor='black')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Keywords in Financial Headlines')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('reports/figures/keyword_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Top 10 Keywords:")
for i, (word, count) in enumerate(list(top_words.items())[:10], 1):
    print(f"{i}. {word}: {count}")

# %% [markdown]
# ## 5. Summary of Findings
# 
# 1. **Publisher Analysis**: Identified top financial news sources
# 2. **Temporal Patterns**: Clear publication patterns with identifiable spikes
# 3. **Keywords**: Financial terminology dominates, providing basis for sentiment analysis
# 4. **Next Steps**: Use these insights for sentiment scoring and correlation with stock prices
