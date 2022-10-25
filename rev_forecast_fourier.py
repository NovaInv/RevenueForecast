# NovaInv October 19, 2022
# Use forier analysis to forecast revenue of United Postal Service (UPS).
# Uncomment parts of code to show desired figures.
# References: https://ataspinar.com/2020/12/22/time-series-forecasting-with-stochastic-signal-analysis-techniques/
#             https://stockrow.com/UPS/financials/income/quarterly

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
import talib
import math
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_in_data(filename):
	# pull data from file and rearrange
	data = pd.read_excel(filename) #read from excel file
	colNames = data.Index #save column headers
	data = data.iloc[:,1:].transpose() #tranpose data
	data.columns = colNames #reset column names
	return data

def fourierExtrapolation(x, n_predict, harmonicPercent, useTop=True):
	# fast fourier transform of timeseries
	n = x.size
	n_harm = int(n*harmonicPercent)     # number of harmonics in model
	t = np.arange(0, n)

	x_freqdom = np.fft.fft(x)  # detrended x in frequency domain
	f = np.fft.fftfreq(n)      # frequencies
	indexes = list(range(n))

	if useTop:
		# sort indexes by best (highest) amplitudes, lower -> higher
		indexes.sort(key = lambda i: np.max(np.absolute(x_freqdom[i])))
		#print(indexes)
		t = np.arange(0, n + n_predict)
		restored_sig = np.zeros(t.size)
		for i in indexes[-(1 + n_harm * 2):]:
		    ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		    phase = np.angle(x_freqdom[i])          # phase
		    restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
		return restored_sig 
	else:
		# sort indexes by frequency, lower -> higher
		indexes.sort(key = lambda i: np.absolute(f[i]))
		#print(indexes)
		t = np.arange(0, n + n_predict)
		restored_sig = np.zeros(t.size)
		for i in indexes[:(1 + n_harm * 2)]:
			# loop through and reconstruct signal
		    ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		    phase = np.angle(x_freqdom[i])          # phase
		    restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
		return restored_sig 

periodsToForecast = 8
revenue = read_in_data("UPS_IS.xls")["Revenue"] #grab just the revenue
revenue /= 1e6 #get values in millions
#revenue.reset_index(inplace=True, drop=True)
xvalues = range(len(revenue))
xvalues_ext = range(0,len(revenue)+5)

coeff = np.poly1d( np.polyfit(xvalues,revenue, deg=2)) #fit quardratic regression to remove trend
rev_trend = coeff(xvalues_ext) #generate trend line for graph
rev_detrended = revenue - coeff(xvalues) #subtract trend to obtain seasonal component

###### Detrend Data and Show Seasonal Component ###########
# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(xvalues, revenue, label="Revenue", linewidth=3)
# ax.plot(xvalues_ext, rev_trend, label="Trend")
# ax.plot(xvalues, rev_detrended, label="Seasonal Component")
# ax.legend(loc="upper left")
# ax.set_title('Quarterly Revenue with Trend and Seasonal Component')
# plt.show()

###### Fourier Analysis ########
# fft_y_ = np.fft.fft(np.array(rev_detrended))
# fft_y = np.abs(fft_y_[:len(fft_y_)//2])

# fft_x_ = np.fft.fftfreq(len(rev_detrended))
# fft_x = fft_x_[:len(fft_x_)//2] # only need positive half of frequency spectrum

##### Plot frequency spectrum ########
# fig, ax = plt.subplots(figsize=(8,3))
# ax.plot(fft_x, fft_y)
# ax.set_ylabel('Amplitude', fontsize=14)
# ax.set_xlabel('Frequency [1/Quarter]', fontsize=14)
# plt.show()

####### Loop through harmoinc percentages to find the best reconstruction #########
# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(xvalues, rev_detrended, label="Seasonal", linewidth=3)
# neww = range(0,len(xvalues)+1)
# frac_list = [0.1,0.2,0.5]
# for i in frac_list:
# 	restored = fourierExtrapolation(rev_detrended, 1, i, useTop=True)
# 	label = f'{i*100}% Harmonics'
# 	ax.plot(neww, restored, label=label)

# ax.legend(loc='upper left')
# plt.show()


####### Forier Extrapolaiton ##########
seasonal_forecast = fourierExtrapolation(rev_detrended, periodsToForecast, 0.5, useTop=True) #forecast seasonal compenent using 50% of top harmonics present in frequency spectrum
new_xvalues = range(len(revenue), len(revenue)+periodsToForecast)
forecast = seasonal_forecast[-periodsToForecast:] + coeff(new_xvalues) #add forecasted seasonal compenent to trend

##### Plot revenue and forecast #######
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(xvalues, revenue, label="Revenue", linewidth=3)
ax.plot(xvalues_ext, rev_trend, label="Trend")
ax.plot(new_xvalues, forecast, label="Forecast")
ax.legend(loc="upper left")
ax.set_title(f'Quarterly Revenue with {periodsToForecast} Period Forecast')
plt.show()