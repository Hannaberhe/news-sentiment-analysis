"""Tests for technical indicators"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators import calculate_sma, calculate_ema, calculate_rsi

@pytest.fixture
def sample_prices():
    return pd.Series([10, 12, 11, 13, 14, 15, 14, 13, 12, 11])

def test_sma(sample_prices):
    sma = calculate_sma(sample_prices, window=3)
    assert len(sma) == len(sample_prices)
    assert not sma.isnull().all()

def test_ema(sample_prices):
    ema = calculate_ema(sample_prices, window=3)
    assert len(ema) == len(sample_prices)
    assert not ema.isnull().all()

def test_rsi_range(sample_prices):
    rsi = calculate_rsi(sample_prices)
    assert (rsi.dropna() >= 0).all()
    assert (rsi.dropna() <= 100).all()
