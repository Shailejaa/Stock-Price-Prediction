import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style

# Importing data from Quandl platform;
data = quandl.get('WIKI/GOOGL',
                  authtoken='36drUqdt4p27MMzeVoZR')

# To display all columns
pd.set_option('display.max_columns', None)

# Getting only relevant columns
data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Creating relationships among features manually
data['HL_percent'] = ((data['Adj. High'] - data['Adj. Low']) / data['Adj. Low']) * 100
data['HC_percent'] = ((data['Adj. High'] - data['Adj. Close']) / data['Adj. Close']) * 100
data['Percent_chg'] = ((data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open']) * 100

# Storing Adj. Close column in forecast_col separately so in future if we want to go with any other label other than
# Adj. Close then we just have to replace feature name here only.
forecast_col = 'Adj. Close'

# print(data.isnull().sum())
# print(data.isnull().any().any())  # Way2 #Checking if at all there is nan in the entire dataset.

# Creating and storing the number of forecast, I just want to predict for 45 days so ill set to 45.
forecast_out = math.ceil(0.0131 * len(data))
# print(len(data), forecast_out) # 3424 45

# Creating Label column and assigning forecast col to it by shifting the forecast out number from tail of the data
data['Label'] = data[forecast_col].shift(-forecast_out)
# "data['Adj. Close'].shift(-45)" it will shift up the Adj. Close col up bu 45 days so at bottom there will be 45 NAN
# Label values for 45 feature rows and The bottom 45 values in the ‘Adj. Close’ column will be removed from the
# DataFrame

# Getting 30 days data exported to use for evaluation manually 
X_check = data[-forecast_out - 30:-forecast_out]  # will slice like "data[-75:-45]" from the end of the dataframe.

# Exporting
X_check.to_excel('Check_Data.xlsx')

# Now storing the same 30 days data same as X_check into X_future to train along with y and predicts the values and
# compare it with X_check label values manually or to check with Evaluation metrics once the model gets ready
X_future = data.drop(['Label'], axis=1)[-forecast_out - 30:-forecast_out]  # ""data[-75:-45]"" same slice as X_check
y_true = data['Label'][-forecast_out - 30:-forecast_out]

# Storing actual X and y along with X_lately for main model training and testing
X = np.array(data.drop(['Label'], axis=1))
X_lately = X[-forecast_out:]  # stored predictive features and to train it separately thta is 45 days
X = X[:-forecast_out - 30]  # removed predictive features rows and X_check rows that is 75 from X data

data.dropna(inplace=True)  # dropping nan 45 rows before assigning data to y

y = np.array(data['Label'])
y = y[:-30]  # dropping last 30 rows data because we are going to use it for y_true.

# print(len(data),len(X), len(y), len(X_future),len(X_check), len(y_true), len(X_lately))  #3379 3349 3349 30 30 30 45

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

scaler = preprocessing.StandardScaler()  # Initializing

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_lately = scaler.transform(X_lately)
X_future = scaler.transform(X_future)

reg = LinearRegression()  # Initializing
# reg = svm.SVR() # Initializing SVM ALGO

reg_fit = reg.fit(X_train, y_train)
reg_score = reg.score(X_test, y_test)

# print(reg_score)   # 0.9703311620446131 more than older script "0.8829838573145374" - Linear

forecast_set = reg.predict(X_future)

data['Forecast'] = np.nan
last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Calculate the number of days needed to cover all predictions
num_days = len(forecast_set)

# Generate future dates excluding weekends
next_date = datetime.datetime.fromtimestamp(next_unix)
end_date = next_date + datetime.timedelta(days=num_days * 2)  # Roughly double the num_days to account for weekends
future_dates = pd.bdate_range(start=next_date, end=end_date)

# If future_dates is shorter than forecast_set, extend future_dates
while len(future_dates) < len(forecast_set):
    end_date += datetime.timedelta(days=7)  # Extend by one week
    future_dates = pd.bdate_range(start=next_date, end=end_date)

# Now, your loop
for i, forecast in enumerate(forecast_set):
    data.loc[future_dates[i]] = [np.nan for _ in range(len(data.columns) - 1)] + [forecast]

style.use('ggplot')

data['Adj. Close'].plot()
data['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Calculate error metrics
y_pred = reg.predict(X_future)  # Predicting to compare with set aside data.
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted Values'])
y_pred_df.to_excel('Y_Pred.xlsx', index=False)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = sqrt(mse)

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
# MAE: 37.00421375881677, MSE: 3292.068602668291, RMSE: 57.37655098268186
