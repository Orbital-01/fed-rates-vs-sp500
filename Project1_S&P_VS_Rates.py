import os
from fredapi import Fred

'''
Pull Federal Funds Rate data from FRED using fredapi library. The data is downloaded using the API key stored in 'fred_key.txt'.
The head, tail, and shape of the DataFrame are printed to the console.

'''

fred = Fred(api_key_file='fred_key.txt')
fedfunds = fred.get_series('FEDFUNDS')

print(fedfunds.head())
print(fedfunds.tail())
print(fedfunds.shape)

'''
Pull S&P 500 data from Yahoo Finance using yfinance library. The data is downloaded from January 1, 1954 to December 31, 2024. 
Thehead, tail, and shape of the DataFrame are printed to the console.
'''
import yfinance as yf

sp500 = yf.download('^GSPC', start='1954-01-01', end='2024-12-31')

print(sp500.head())
print(sp500.tail())
print(sp500.shape)

# Save raw data so we dont havev to repull it every time we run the script
fedfunds.to_csv('fedfunds_raw.csv')
sp500.to_csv('sp500_raw.csv')
