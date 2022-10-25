# RevenueForecast
This repository stores python scripts used to forecast revenue using various techniques. With adjustment, the scripts could be used to forecast other metrics as well.
The data used in the examples are for United Postal Service (UPS) and is included in the repository.

# ARMA Model & SARIMA Model
Revenue is forecasted n periods ahead using ARMA model. The p and q parameters were estimated using acf and pacf plots. The parameters for the SARIMA model is grid
searched using pmdarima's auto_arima function.
![image](https://user-images.githubusercontent.com/45056473/197661888-575d6e7f-ba6a-4ead-9339-36753276c1fe.png)

# Fourier Analysis
Revenue is forecasted n periods ahead using fourier analysis. The frequency spectrum is calculated using numpy's fast fourier transform, and either the high frequencies
or low amplitude frequencies are filted out. The signal is reconstructed and extrapolated to yield forecast. 
![image](https://user-images.githubusercontent.com/45056473/197662542-dcdcc6c7-0aeb-4e20-a8d1-944fd2b59dee.png)
