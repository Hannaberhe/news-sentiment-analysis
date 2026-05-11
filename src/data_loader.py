"""Data loading module for news and stock data"""
import pandas as pd
import numpy as np
import re

def load_headlines(filepath):
    """Load and parse news headlines"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Extract publisher domain from email format
    if 'publisher' in df.columns:
        df['publisher_domain'] = df['publisher'].apply(
            lambda x: re.search(r'@([\w.]+)', str(x)).group(1) 
            if re.search(r'@([\w.]+)', str(x)) else str(x)
        )
    
    df['headline_length'] = df['headline'].str.len()
    df['word_count'] = df['headline'].str.split().str.len()
    
    return df

def load_stock_data(filepath):
    """Load and validate stock price data"""
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df.columns = [col.strip().title() for col in df.columns]
    df = df.rename(columns={'Adj Close': 'Adj_Close'})
    
    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def handle_missing_values(df):
    """Handle missing values in stock data"""
    df = df.copy()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df
