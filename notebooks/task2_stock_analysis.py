
# %% [markdown]
# ## 5. PyNance Metrics - Additional Financial Indicators

# %%
from src.pynance_metrics import compute_atr, compute_mfi, compute_obv, compute_stochastic

# Compute PyNance metrics
df['ATR'] = compute_atr(df)
df['MFI'] = compute_mfi(df)
df['OBV'] = compute_obv(df)
df['Stoch_K'], df['Stoch_D'] = compute_stochastic(df)

print("PyNance metrics computed successfully!")

# %%
# Visualize PyNance metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('PyNance Financial Metrics', fontsize=16, fontweight='bold')

# ATR - Volatility
axes[0, 0].plot(df['Date'], df['ATR'], color='darkred', linewidth=1.5)
axes[0, 0].set_title('Average True Range (ATR) - Volatility')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('ATR')
axes[0, 0].grid(True, alpha=0.3)

# MFI - Money Flow Index
axes[0, 1].plot(df['Date'], df['MFI'], color='teal', linewidth=1.5)
axes[0, 1].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Overbought')
axes[0, 1].axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Oversold')
axes[0, 1].set_title('Money Flow Index (MFI)')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('MFI')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# OBV - On-Balance Volume
axes[1, 0].plot(df['Date'], df['OBV'], color='purple', linewidth=1.5)
axes[1, 0].set_title('On-Balance Volume (OBV)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('OBV')
axes[1, 0].grid(True, alpha=0.3)

# Stochastic Oscillator
axes[1, 1].plot(df['Date'], df['Stoch_K'], label='%K', color='blue', linewidth=1.5)
axes[1, 1].plot(df['Date'], df['Stoch_D'], label='%D', color='orange', linewidth=1.5)
axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=20, color='green', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Stochastic Oscillator')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Value')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/pynance_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
