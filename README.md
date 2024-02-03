# Stock Price Prediction using Linear Regression

This project aims to predict the future closing price of a given stock across a given period of time in the future. For this project, we will use a Linear Regression model from the Scikit-Learn library in Python.

## Dataset

The dataset used for this project originates from the `quandl` library, specifically the `WIKI/GOOGL` stock prices. The dataset includes features such as the Adjusted Close Price, Adjusted Open Price, Adjusted High Price, Adjusted Low Price, and Adjusted Volume of Trade.

## Features

We calculate three additional features:

- `HL_percent`: The percentage change between the high and low prices of the day.
- `HC_percent`: The percentage change between the high price of the day and the closing price.
- `Percent_chg`: The percentage change between the opening price of the day and the closing price.

## Labels

The 'Label' column is created by shifting the 'Adj. Close' the column up by a certain number of days (`forecast_out`). This means that for any given row in the dataset, the 'Label' column will contain the 'Adj. Close' price from `forecast_out` days in the future.

## Model

We use a Linear Regression model from the Scikit-Learn library to predict the 'Adj. Close' price. The model is trained on the historical stock price data and the calculated features. The trained model is then used to predict the 'Adj. Close' price for the next `forecast_out` days.

## Evaluation

The Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) are used to evaluate the performance of the model.

## Future Work

The model's performance can potentially be improved by tuning its parameters or using different features.

# NOTE:
This project is inspired by and follows Sentdex’s ‘Machine Learning with Python’ playlist (videos 2 to 12) of Simple Linear Regression. While the project is largely based on concepts and code from these videos, I have also made some modifications to better suit my specific needs and to enhance my understanding of the subject matter. 

This is one of my initial projects in Machine Learning, and I am open to constructive feedback and suggestions for improvement. If you notice any mistakes or areas where I could do better, please feel free to share your insights. Your input is greatly appreciated!
