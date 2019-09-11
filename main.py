# REFERENCE = https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7

import pandas as pd
import datetime, math
import numpy as np
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.pipeline import make_pipeline
from datetime import timedelta

# Adjusting the style of matplotlib
style.use('ggplot')

def download_data(stocks, service, start, end):
    df = web.DataReader(stocks, service, start = start, end = end)
    df.to_pickle('data.pkl')
    return df

def plot_moving_average(close, stock, window):
    mavg = close.rolling(window = window).mean()
    close.plot(label = stock)
    mavg.plot(label = 'MAVG')
    plt.legend()
    plt.show()

def plot_returns(close):
    rets = close / close.shift(1) - 1
    rets.plot(label='return')
    plt.show()

def plot_correlation(df, stock1, stock2):
    retscomp = df['Adj Close'].pct_change()
    corr = retscomp.corr()
    plt.scatter(retscomp[stock1], retscomp[stock2])
    plt.xlabel("Returns " + stock1)
    plt.ylabel("Returns " + stock2)
    plt.show()

def plot_kde(df):
    retscomp = df['Adj Close'].pct_change()
    pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))
    plt.show()

def plot_heatmaps(df):
    retscomp = df['Adj Close'].pct_change()
    corr = retscomp.corr()
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns)
    plt.show()

def plot_risk_return(df):
    retscomp = df['Adj Close'].pct_change()
    plt.scatter(retscomp.mean(), retscomp.std())
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')
    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )
    plt.show()

def feature_engineering(df, stock):
    """Create features to feed ML model"""
    features = pd.DataFrame()
    features['Adj Close'] = df['Adj Close'][stock]
    features['Volume'] = df['Volume'][stock]
    features['HL_PCT'] = (df['High'][stock] - df['Low'][stock]) / df['Close'][stock] * 100.0
    features['PCT_change'] = (df['Close'][stock] - df['Open'][stock]) / df['Open'][stock] * 100.0
    return features

def preprocess_data(features):
    """Clean the data"""
    # Drop missing value
    features.fillna(value = -99999, inplace = True)
    
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(features)))
    
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    features['label'] = features[forecast_col].shift(-forecast_out)
    X = np.array(features.drop(['label'], 1))
    
    # Scale the X so that everyone can have the same distribution for linear regression
    X = scale(X)
    
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_forecast = X[-forecast_out : ]
    X = X[ : -forecast_out]

    # Separate label and identify it as y
    y = np.array(features['label'])
    y_forecast = y[-forecast_out : ]
    y = y[ : -forecast_out]

    return (X, y, X_forecast, y_forecast)

def split_train_test(X, y, test_ratio):
    """Splits the data into the train and test sets"""
    X_train = X[ : -int(len(X) * test_ratio)]
    X_test = X[len(X_train) : ]
    y_train = y[ : -int(len(y) * test_ratio)]
    y_test = y[len(y_train) : ]

    return (X_train, y_train, X_test, y_test)

def train_models(X, y):
    """Runs several regression tests"""
    X_train, y_train, X_test, y_test = split_train_test(X, y, 0.1)
    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)
    
    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors = 2)
    clfknn.fit(X_train, y_train)

    confidencereg = clfreg.score(X_test, y_test)
    confidencepoly2 = clfpoly2.score(X_test,y_test)
    confidencepoly3 = clfpoly3.score(X_test,y_test)
    confidenceknn = clfknn.score(X_test, y_test)

    print("Accuracy of Linear regression | ", confidencereg, "%")
    print("Accuracy of Quadratic Regression 2 | ", confidencepoly2, "%")
    print("Accuracy of Quadratic Regression 3 | ", confidencepoly3, "%")
    print("Accuracy of KNN Regression | ", confidenceknn, "%")
    
    return (clfreg, clfpoly2, clfpoly3, clfknn)

def predict_future(X_forecast, y_forecast, clfreg, clfpoly2, clfpoly3, clfknn):
    """Predicts the future stock price depending on the candlestick information"""
    predictions = pd.DataFrame()
    predictions['Linear regression'] = clfreg.predict(X_forecast)
    predictions['Quadratic Regression 2'] = clfpoly2.predict(X_forecast)
    predictions['Quadratic Regression 3'] = clfpoly3.predict(X_forecast)
    predictions['KNN Regression'] = clfknn.predict(X_forecast)
    
    print(predictions)
    return predictions

def plot_predictions(features, predictions):
    """Plots the predictions with the Adj Close prices"""
    plot_data = pd.DataFrame()
    plot_data['Adj Close'] = features['Adj Close']

    last_date = plot_data.index[-1]
    plot_data['Linear regression'] = np.nan
    plot_data['Quadratic Regression 2'] = np.nan
    plot_data['Quadratic Regression 3'] = np.nan
    plot_data['KNN Regression'] = np.nan

    for idx in predictions.index:
        next_date = last_date + timedelta(days=1)
        last_date = next_date
        next_data = predictions.loc[idx]
        plot_data.at[next_date, 'Linear regression'] = next_data['Linear regression']
        plot_data.at[next_date, 'Quadratic Regression 2'] = next_data['Quadratic Regression 2']
        plot_data.at[next_date, 'Quadratic Regression 3'] = next_data['Quadratic Regression 3']
        plot_data.at[next_date, 'KNN Regression'] = next_data['KNN Regression']

    plot_data.plot()
    plt.show()
        
        
        
        




        

    print(plot_data)








# start = datetime.datetime(2010, 1, 1)
# end = datetime.datetime(2017, 1, 11)
# df = download_data(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo', start, end)

df = pd.read_pickle('data.pkl')

# plot_moving_average(df['Close']['AAPL'], 'AAPL', 100)
# plot_returns(df['Close']['AAPL'])
# plot_correlation(df, 'AAPL', 'GE')
# plot_kde(df)
# plot_heatmaps(df)
# plot_risk_return(df)

##
features = feature_engineering(df, 'AAPL')
X, y, X_forecast, y_forecast = preprocess_data(features)
clfreg, clfpoly2, clfpoly3, clfknn = train_models(X, y)
predictions = predict_future(X_forecast, y_forecast, clfreg, clfpoly2, clfpoly3, clfknn)
plot_predictions(features, predictions)

##


