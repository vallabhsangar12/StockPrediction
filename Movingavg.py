import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


df = pd.read_csv('NIFTY_6month.csv')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')

df['SMA_20'] = df['Close'].rolling(window=20).mean()
# Calculate Exponential Moving Average (EMA) for a 20-day window
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# Step 1: Plotting with Matplotlib
plt.figure(figsize=(12, 6))

# Plot the Close Price, SMA and EMA
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue', linestyle='-', marker='o')
plt.plot(df['Date'], df['SMA_20'], label='20-Day SMA', color='green', linestyle='--')
plt.plot(df['Date'], df['EMA_20'], label='20-Day EMA', color='red', linestyle='-.')

# Add title and labels
plt.title('NSE Nifty 50 - Price with Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (₹)', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid for better visibility
plt.grid(True)

# Add legend
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the static plot
plt.show()

# Step 2: Plotting with Plotly for Interactivity

# Create an interactive plot
fig = go.Figure()

# Add Close Price, SMA, and EMA lines with hover functionality
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines+markers', 
                         name='Close Price', line=dict(color='blue'),
                         hovertemplate='Date: %{x}<br>Close Price: ₹%{y}<extra></extra>'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', 
                         name='20-Day SMA', line=dict(color='green', dash='dash'),
                         hovertemplate='Date: %{x}<br>20-Day SMA: ₹%{y}<extra></extra>'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', 
                         name='20-Day EMA', line=dict(color='red', dash='dot'),
                         hovertemplate='Date: %{x}<br>20-Day EMA: ₹%{y}<extra></extra>'))

# Customize layout for Plotly
fig.update_layout(
    title='NSE Nifty 50 - Price with Moving Averages (Interactive)',
    xaxis_title='Date',
    yaxis_title='Price (₹)',
    template='plotly_dark',
    xaxis_tickangle=45,
    hovermode='closest'
)

# Show the interactive plot
fig.show()
