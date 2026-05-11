"""Sentiment analysis for financial headlines"""
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(df, text_column='headline'):
    """Analyze sentiment of headlines using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    
    df = df.copy()
    sentiments = df[text_column].apply(lambda x: analyzer.polarity_scores(str(x)))
    
    df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
    df['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
    df['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])
    df['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
    
    # Categorize sentiment
    df['sentiment_label'] = pd.cut(
        df['sentiment_compound'],
        bins=[-1, -0.05, 0.05, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    return df
