# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 22-09-2025



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv("/content/car_price_prediction.csv")

# Use only Price column, limit to first 2000 rows for speed
X = data['Price'].dropna().reset_index(drop=True).head(2000)

plt.rcParams['figure.figsize'] = [12, 6]

# Plot original price series
plt.plot(X)
plt.title('Car Price Data (first 2000 samples)')
plt.show()

# ACF and PACF with fewer lags
plt.subplot(2, 1, 1)
plot_acf(X, lags=50, ax=plt.gca())
plt.title('ACF (Price)')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=50, ax=plt.gca())
plt.title('PACF (Price)')
plt.tight_layout()
plt.show()

# Fit ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params.get('ar.L1', 0)
theta1_arma11 = arma11_model.params.get('ma.L1', 0)

# Simulate smaller ARMA(1,1)
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=500)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process (500 samples)')
plt.show()

# Fit ARMA(2,2)
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params.get('ar.L1', 0)
phi2_arma22 = arma22_model.params.get('ar.L2', 0)
theta1_arma22 = arma22_model.params.get('ma.L1', 0)
theta2_arma22 = arma22_model.params.get('ma.L2', 0)

# Simulate smaller ARMA(2,2)
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=500)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process (500 samples)')
plt.show()

```
### OUTPUT:

ORIGINAL:
<img width="1046" height="499" alt="image" src="https://github.com/user-attachments/assets/056679f9-d6e6-45ba-abb2-5c7944287c83" />

Partial Autocorrelation
<img width="1200" height="291" alt="image" src="https://github.com/user-attachments/assets/8df5ac51-2045-4fc3-82e0-33e0de9bf3e4" />

Autocorrelation
<img width="1183" height="278" alt="image" src="https://github.com/user-attachments/assets/75fdd0cf-fbc1-40de-8b85-0ee1d4d839a0" />

SIMULATED ARMA(1,1) PROCESS:
<img width="989" height="507" alt="image" src="https://github.com/user-attachments/assets/0c7dfa8a-9316-4bbc-b763-0da102478e49" />

Partial Autocorrelation
<img width="1036" height="496" alt="image" src="https://github.com/user-attachments/assets/04692cea-db44-4bba-a588-dc9464b8a946" />


Autocorrelation
<img width="1034" height="501" alt="image" src="https://github.com/user-attachments/assets/aab2da54-e5be-409e-b031-325d176c804b" />



SIMULATED ARMA(2,2) PROCESS:
<img width="1038" height="500" alt="image" src="https://github.com/user-attachments/assets/09468fcc-d3d6-4bf5-9d92-038f541a0a35" />

Partial Autocorrelation
<img width="1044" height="498" alt="image" src="https://github.com/user-attachments/assets/fb4b92c7-bbeb-4bea-82b8-2fc995aaa3cd" />

Autocorrelation
<img width="1031" height="494" alt="image" src="https://github.com/user-attachments/assets/46ab201b-84a8-43df-b672-0349ef3aed19" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
