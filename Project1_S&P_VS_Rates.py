import os
from fredapi import Fred
import yfinance as yf
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# --- Milestone 1: Pulling Data from FRED and Yahoo Finance ---
'''
Pull Federal Funds Rate data from FRED using fredapi library. The data is downloaded using the API key stored in 'fred_key.txt'.
The head, tail, and shape of the DataFrame are printed to the console.

'''

fred = Fred(api_key_file='fred_key.txt')
fedfunds = fred.get_series('FEDFUNDS')

sp500 = yf.download('^GSPC', start='1954-01-01')

print(fedfunds.head())
print(fedfunds.tail())
print(fedfunds.shape)

'''
Pull S&P 500 data from Yahoo Finance using yfinance library. The data is downloaded from January 1, 1954 to December 31, 2024. 
Thehead, tail, and shape of the DataFrame are printed to the console.
'''

print(sp500.head())
print(sp500.tail())
print(sp500.shape)

# Save raw data so we dont havev to repull it every time we run the script
fedfunds.to_csv('fedfunds_raw.csv')
sp500.to_csv('sp500_raw.csv')


 # --- Milestone 2: Resampling S&P 500 Data to Monthly Frequency ---

'''

S&P 500 data is daily, but the Fed Funds Rate is monthly — so we need to bring them to the same frequency before merging

'''

sp500_monthly = sp500['Close'].resample('MS').last().iloc[:, 0]  # Resample to monthly frequency, taking the last closing price of each month

# MS means month start,  it groups the daily data by month and takes the last closing price in each month, 
# giving you one value per month, timestamped the same way your FRED data is.

print(sp500_monthly.head())
print(sp500_monthly.tail())
print(sp500_monthly.shape)

'''

Merge into a single DataFrame for analysis. The resulting DataFrame will have two columns: 'FedFunds' and 'SP500', 
with the index being the date. Any rows with missing values are dropped to ensure clean data for analysis.

'''
df = pd.DataFrame({
    'FedFunds': fedfunds, 
    'SP500': sp500_monthly
    })

df = df.dropna()  # Drop any rows with missing values

print(df.head())
print(df.tail())
print(df.shape)


'''
lag/correlation analysis 

'''
df['FedFunds_change'] = df['FedFunds'].diff()  # Calculate the change in Fed Funds Rate. .diff() computes the difference between the current and previous row, giving you the change in the Fed Funds Rate from one month to the next.
df['SP500_change'] = df['SP500'].pct_change() * 100  # Calculate the percentage change in S&P 500. pct_change() computes the percentage change between the current and previous row, and multiplying by 100 converts it to a percentage.

print(df.head())
print(df.tail())

df.to_csv('fedfunds_sp500_merged.csv')  # Save the merged DataFrame to a CSV file for future analysis


# ---- Milestone 3: Correlation Analysis ----

df = pd.read_csv('fedfunds_sp500_merged.csv', index_col=0, parse_dates=True)  # Read the merged DataFrame from the CSV file. This keeps your script faster and avoids hitting the FRED API repeatedly while you're iterating on analysis.

print(df.head())
print(df.shape)


'''
Creating lagged versions of the Fed Funds Rate change to analyze its correlation with the S&P 500 change.
The lagged columns are created by shifting the 'FedFunds_change' column by 1 to 6 months. This allows us to see how changes in the Fed Funds Rate might affect the S&P 500 in subsequent months.
'''

for lag in [1, 3, 6, 12]:
    df['FedFunds' + f'_lag{lag}'] = df['FedFunds_change'].shift(lag)

print(df.head(15))


# Calculate correlation at each lag and store the results in a dictionary. The correlation is calculated between the lagged Fed Funds Rate change and the S&P 500 change. The results are printed to the console.
correlations = {}
for lag in [0, 1, 3, 6, 12]:
    if lag == 0: 
            corr = df['FedFunds_change'].corr(df['SP500_change'])
    else:
        corr = df[f'FedFunds_lag{lag}'].corr(df['SP500_change'])
    correlations[lag] = corr

print(correlations)


# checks statistical significance , not just the raw correlation coeifficent, since a small correlation on 850 data points could still be statistically significant. The p-value is calculated for each lagged correlation using the Pearson correlation test from the scipy.stats library. The results are printed to the console.

for lag in [0,1,3,6,12]:
    col = 'FedFunds_change' if lag == 0 else f'FedFunds_lag{lag}'
    valid = df[[col, 'SP500_change']].dropna()  # Drop rows with NaN values for the current lagged column and S&P 500 change
    corr, p_value = stats.pearsonr(valid[col], valid['SP500_change'])  # Calculate the Pearson correlation coefficient and p-value
    print(f'Lag {lag} months: correlation = {corr:.4f}, p-value = {p_value:.4f}')  # Print the results with formatted output


# --- Milestone 4: Visualizing the Correlation Results ---


'''

Line Graph: Fed Funds Rate vs. S&P 500 Index Level
 
 '''
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(df.index, df['FedFunds'], color='tab:blue', label='Fed Funds Rate')
ax1.set_xlabel('Date')
ax1.set_ylabel('Fed Funds Rate (%)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  # Create a second y-axis for the S&P 500 change
ax2.plot(df.index, df['SP500'], color='tab:red', label='S&P 500')
ax2.set_ylabel('S&P 500 Index Level', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Fed Funds Rate vs. S&P 500 Index Level (1954-2026)')
fig.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('chart1_rate_vs_sp500.png', dpi=150)  # Save the figure as a PNG file with 150 dpi resolution
plt.show()  # Display the plot


'''

Scatter Plot: Fed Funds Rate Change vs. S&P 500 Change

'''

fig, ax = plt.subplots(figsize=(10, 6))

valid = df[['FedFunds_change', 'SP500_change']].dropna()  # Drop rows with NaN values for the current lagged column and S&P 500 change

ax.scatter(valid['FedFunds_change'], valid['SP500_change'], alpha=0.5, color='tab:blue')

# add regression line
z = np.polyfit(valid['FedFunds_change'], valid['SP500_change'], 1)  # Fit a linear regression line to the data
p = np.poly1d(z)  # Create a polynomial object from the fitted coefficients
x_line = np.linspace(valid['FedFunds_change'].min(), valid['FedFunds_change'].max(), 100)  # Generate x values for the regression line
ax.plot(x_line, p(x_line), color='tab:red', linewidth=2, label='Regression Line')

ax.set_xlabel('Fed Funds Rate Change (percentage points)')
ax.set_ylabel('S&P 500 Return (%)')
ax.set_title('Fed Funds Rate Change vs. S&P 500 Return (Same Month)')
plt.legend()  # Add a legend to the plot

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('chart2_scatter_regression.png', dpi=150)  # Save the figure as a PNG file with 150 dpi resolution
plt.show()  # Display the plot


'''

Chart 3: Correlation-by-Lag Bar Chart

'''

fig, ax = plt.subplots(figsize=(10, 6))

lags = [0, 1, 3, 6, 12]
corr_values = [correlations[lag] for lag in lags]  # Extract correlation values for the specified lags

bars = ax.bar([str(lag) for lag in lags], corr_values, color='tab:blue')  # Create a bar chart with lag labels on the x-axis

# highlight the statistically significant bars (lag 0 and 1) in a different color
bars[0].set_color('tab:red')  # Highlight lag 0 bar in red
bars[1].set_color('tab:red')  # Highlight lag 1 bar in red


ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Lag (months)')
ax.set_ylabel('Correlation Coefficient')
ax.set_title('Correlation Between Fed Funds Rate Change and S&P 500 Return by Lag')

# Add a note distinguishing statistically significant correlations
ax.text(0.02, 0.95, 'Red bars indicate statistically significant correlations (p < 0.05)', 
        transform=ax.transAxes, fontsize=9, verticalalignment='top', color='red')


plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('chart3_correlation_by_lag.png', dpi=150)  # Save the figure as a PNG file with 150 dpi resolution
plt.show()  # Display the plot


'''
Rate hike vs. rate cut analysis
'''

df['rate_direction'] = df['FedFunds_change'].apply(
     lambda x: 'Hike' if x > 0 else ('Cut' if x < 0 else 'No Change'))  # Categorize Fed Funds Rate changes as 'Hike', 'Cut', or 'No Change'

avg_returns = df.groupby('rate_direction')['SP500_change'].mean()  # Calculate the average S&P 500 return for each rate change category
print(avg_returns)  # Print the average returns to the console

fig, ax = plt.subplots(figsize=(8, 6))

colors = {'Hike': 'tab:red', 'Cut': 'tab:green', 'No Change': 'tab:grey'}  # Define colors for each category
bar_colors = [colors[direction] for direction in avg_returns.index]  # Map the colors to the average returns index

ax.bar(avg_returns.index, avg_returns.values, color=bar_colors)  # Create a bar chart for average returns by rate change category
ax.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0 for reference
ax.set_ylabel('Average S&P 500 Return (%)')  # Set the y-axis label
ax.set_title('Average S&P 500 Return by Fed Funds Rate Change Direction')  

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('chart4_hike_vs_cut.png', dpi=150)  # Save the figure as a PNG file with 150 dpi resolution
plt.show()  # Display the plot








