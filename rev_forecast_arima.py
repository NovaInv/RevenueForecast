# NovaInv October 19, 2022
# Use ARMA model to forecast revenue of United Postal Service (UPS).
# Uncomment parts of code to show desired figures.
# References: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
#             https://stockrow.com/UPS/financials/income/quarterly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import warnings
import math
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)

def read_in_data(filename):
	# pull data from file and rearrange
	data = pd.read_excel(filename) #read from excel file
	colNames = data.Index #save column headers
	data = data.iloc[:,1:].transpose() #tranpose data
	data.columns = colNames #reset column names
	return data

periodsToForecast = 8
revenue = read_in_data("UPS_IS.xls")["Revenue"] #grab just the revenue
revenue /= 1e6 #get values in millions

xvalues = range(len(revenue))
xvalues_ext = range(0,len(revenue)+periodsToForecast)

coeff = np.poly1d( np.polyfit(xvalues,revenue, deg=2)) #fit quardratic regression to remove trend
rev_trend = coeff(xvalues_ext) #generate trend line for graph
rev_detrended = revenue - coeff(xvalues) #subtract trend to obtain seasonal component

#### Plot acf and pacf plots ########
fig, axs = plt.subplots(2,2, figsize=(10,6))
axs[0,0].plot(revenue)
axs[1,0].plot(rev_detrended)
plot_acf(rev_detrended, lags=15, ax=axs[0,1])
plot_pacf(rev_detrended, lags=15, ax=axs[1,1])
plt.show()

# ARMA Model and forecast
model = ARIMA(rev_detrended,order=(4,0,4),trend='n',enforce_invertibility=False).fit() # parameters obtained from acf and pacf plots
pred = model.forecast(steps=periodsToForecast)

new_xvalues = range(len(revenue), len(revenue)+periodsToForecast)
forecast = pred.values + coeff(new_xvalues) #add forecasted seasonal compenent to trend


# SARIMAX Model autofit and forecast with confidence intervals
smodel = pm.auto_arima(revenue, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=1, seasonal=True,
							d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

# Forecast and confidence intervals
s_pred, confint = smodel.predict(n_periods=periodsToForecast, return_conf_int=True)
lower_conf = pd.Series(confint[:, 0], index=new_xvalues)
upper_conf = pd.Series(confint[:, 1], index=new_xvalues)



##### Plot revenue and forecast #######
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(xvalues, revenue, label="Revenue", linewidth=3)
ax.plot(xvalues_ext, rev_trend, label="Trend")
ax.plot(new_xvalues, forecast, label="ARMA Forecast")
ax.plot(new_xvalues, s_pred, label="SARIMA Forecast")
ax.fill_between(new_xvalues, lower_conf, upper_conf, color='k', alpha=.25, label="SARIMA 95% Conf Intervals")
ax.legend(loc="upper left")
ax.set_title(f'Quarterly Revenue with {periodsToForecast} Period Forecast')
plt.show()



##### Grid Search Optimal ARMA(p,q) parameters ###########
# plist, qlist, aiclist, mselist = [], [], [], []
# for p in range(0,8):
# 	print(p)
# 	for q in range(0,8):
# 		model = ARIMA(rev_detrended,order=(p,0,q),seasonal_order=(0,0,0,0),trend='n',enforce_invertibility=False).fit()
# 		#print(model.summary())
# 		index_start = math.floor(len(rev_detrended)*0.75)
# 		index_end = len(rev_detrended)
# 		in_sample_reconstruction = model.predict(start=index_start, end=index_end-1)
# 		mse = mean_squared_error(np.array( rev_detrended[index_start:index_end]),in_sample_reconstruction)
# 		#print(f"Model ({p},0,{q})  AIC: {model.aic()}  MSE: {mse}")
# 		plist.append(p)
# 		qlist.append(q)
# 		aiclist.append(model.aic)
# 		mselist.append(mse)

# results = pd.DataFrame({'p':plist,'q':qlist, 'aic':aiclist, 'mse':mselist})
# results.sort_values(by=['aic'], inplace=True, ascending=True)
# print(results.head(10))
# results.sort_values(by=['mse'], inplace=True, ascending=True)
# print(results.head(10))
