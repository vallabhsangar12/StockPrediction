import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
df=pd.read_csv('NIFTY_month.csv')
print(df)
print(df.head())
df.columns = df.columns.str.strip()

print(df.info())
#df.plot(title="NSE Data",xlabel="Date",ylabel="Close")
#plt.show()


# Plotting the 'Close' price over time



plt.figure(figsize=(12, 6))

# Plot with markers and different line styles
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue', linestyle='-', marker='o', markersize=5)
plt.plot(df['Date'], df['Open'], label='Open Price', color='green', linestyle='-', marker='s', markersize=5)
plt.plot(df['Date'], df['High'], label='High Price', color='red', linestyle='-', marker='^', markersize=5)
plt.plot(df['Date'], df['Low'], label='Low Price', color='orange', linestyle='-', marker='x', markersize=5)

# Add title and labels
plt.title('NSE Nifty 50 - Price Series with Markers', fontsize=14)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Price (₹)', fontsize=8)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add a grid
plt.grid(True)

# Add legend
plt.legend()


# Adjust layout and show the plot
plt.tight_layout()
plt.show()

