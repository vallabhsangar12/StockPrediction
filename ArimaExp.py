import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the data from the CSV file
df = pd.read_csv('Raw_sales.csv')

# Rename columns for better readability
df.columns = ['Month', 'Sales']

# Drop rows with index 105 and 106 (if necessary)
df.drop(106, axis=0, inplace=True)
df.drop(105, axis=0, inplace=True)

# Check for missing values and drop if any
if df['Sales'].isnull().any():
    df.dropna(subset=['Sales'], inplace=True)

# Convert 'Month' to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Set 'Month' as the index
df.set_index('Month', inplace=True)

# Print the first few rows to check data
print("Data Head:")
print(df.head())

# Visualizing the data using matplotlib
df.plot()
plt.title('Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Perform the Augmented Dickey-Fuller (ADF) test to check stationarity
try:
    result = adfuller(df['Sales'])
    # Print the results of the ADF test
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

    # Interpretation of the test
    if result[1] < 0.05:
        print("The data is stationary (p-value < 0.05).")
    else:
        print("The data is not stationary (p-value >= 0.05).")
except Exception as e:
    print("Error while performing ADF test:", e)

# Making the data stationary (Differencing)
try:
    # Apply differencing to make data stationary
    df['Sales_diff'] = df['Sales']-df['Sales'].shift(12)
    print(df)

    # Drop the NaN value created by differencing
    df.dropna(subset=['Sales_diff'], inplace=True)

    # Perform the ADF test on the differenced data to check stationarity
    result_diff = adfuller(df['Sales_diff'])
    
    # Print the results of the ADF test
    print('ADF Statistic (Differenced Data):', result_diff[0])
    print('p-value (Differenced Data):', result_diff[1])
    print('Critical Values (Differenced Data):', result_diff[4])

    # Interpretation of the test
    if result_diff[1] < 0.05:
        print("The differenced data is stationary (p-value < 0.05).")
    else:
        print("The differenced data is not stationary (p-value >= 0.05).")
except Exception as e:
    print("Error while performing ADF test on differenced data:", e)
df['Sales_diff'].plot()
plt.title('Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF
plot_acf(df['Sales_diff'].iloc[13:], lags=40, ax=ax1)
ax1.set_title("Autocorrelation Function (ACF)")

# Plot PACF
plot_pacf(df['Sales_diff'].iloc[13:], lags=40, ax=ax2)
ax2.set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()


#Arima Implementation
#p=1,d=1,0 or 1
from statsmodels.tsa.arima.model import ARIMA  # Correct import
import pandas as pd

# Assuming `df` is your DataFrame and 'Sales' is the column
# Replace `order` with the desired ARIMA order (p, d, q)
model = ARIMA(df['Sales'], order=(1, 1, 1))
model_fit = model.fit()

# Display summary
print(model_fit.summary())
df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()