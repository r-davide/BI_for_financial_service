#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: 807160 Raccuglia Davide

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# function that calculates the simple (gross) return
def gross_return(data):
    # gross_return = pd.Series(dtype=float)
    # for i in range(1, len(data)):
    #     ret = (data[i] - data[i-1]) / data[i-1]
    #     ret_ts = pd.Series(ret, index=[data.index[i]])
    #     gross_return = gross_return.append(ret_ts)
    # return gross_return
        
    return data.pct_change().dropna()


# function that calculates the continuously compounded return
def cc_return(data):
    return np.log(data / data.shift(1)).dropna()


# function that performs a single plot of all the columns in the df
def plot_data(data, title, labels, colors):
    plt.figure(figsize=(21,12))
    if (isinstance(data, pd.Series)):
        series = plt.plot(data)
        plt.setp(series, color=colors[0], linewidth=1.5)
    else:
        i = 0
        for i in range(len(data.columns)):
            series = plt.plot(data.iloc[:,i])
            if (data.columns[i] == '^GSPC'):
                plt.setp(series, 
                         color=colors[i], 
                         linewidth=1.5, 
                         linestyle='--')
            else:
                plt.setp((series), color=colors[i], linewidth=2)
        plt.legend(data.columns, loc='best', fontsize=20)
    plt.title(title, fontsize=28)
    plt.xlabel(labels[0], fontsize=20)
    plt.ylabel(labels[1], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    plt.show()
    

# function that performs four-panel plots, with histograms,
# of the CC Return of each stock in comparison with the reference index
def density_plot(data, colors):
    plt.figure(figsize=(21, 12))
    plt.subplots_adjust(hspace=.35, wspace=.15)
    plt.rc('font', size=13)
    
    plt.subplot(221)
    stock = plt.plot(data.iloc[:,0])
    plt.title(data.columns[0] + ' - CC Return', fontsize=21)
    plt.xlabel('Date (year)', fontsize=18)
    plt.ylabel('CC Return', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.setp(stock, color=colors[0], linewidth=1.5)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(223)
    index = plt.plot(data.iloc[:,1])
    plt.title(data.columns[1] + ' - CC Return', fontsize=21)
    plt.xlabel('Date (year)', fontsize=18)
    plt.ylabel('CC Return', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.setp(index, color=colors[1], linewidth=1.5)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(222)
    plt.hist(data.iloc[:,0], density=True, bins=15, color='#DEDEDE')
    plt.title(data.columns[0] + ' - Distribution of CC Return', fontsize=21)
    data.iloc[:,0].plot.density(color=colors[0])
    plt.xlabel('CC Return', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(224)
    plt.hist(data.iloc[:,1], density=True, bins=15, color='#DEDEDE')
    plt.title(data.columns[1] + ' - Distribution of CC Return', fontsize=21)
    data.iloc[:,1].plot.density(color=colors[1])
    plt.ylabel('Density', fontsize=18)
    plt.xlabel('CC Return', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.show()


from statsmodels import api as sm           # qqplot

# functions that performs diagnostic plots, for the input series,
# four panels: histogram, boxplot, density and quantile-quantile
def diagnostic_plots(data, color):
    plt.figure(figsize=(21, 12))
    plt.subplots_adjust(hspace=.35, wspace=.20)
    plt.rc('font', size=13)
    
    # histogram of Monthly CC Return 
    plt.subplot(221)
    plt.hist(data, bins=15, color=color, alpha=0.3)
    plt.axvline(data.quantile(0), color='#000000', linewidth=1.5)
    plt.axvline(data.quantile(.25), color='#005fed', linewidth=1.5)
    plt.axvline(data.quantile(.5), color='#eb152a', linewidth=1.5)
    plt.axvline(data.quantile(.75), color='#005fed', linewidth=1.5)
    plt.axvline(data.quantile(1), color='#000000', linewidth=1.5)
    plt.title(data.name + ' - Monthly CC Return', fontsize=21)
    plt.xlabel('CC Return', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    # Boxplot of Monthly CC Return
    plt.subplot(222)
    plt.boxplot(data)
    data.to_frame().boxplot()
    plt.title('CC Return Boxplot', fontsize=21)
    plt.ylabel('CC Return', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    
    # Smoothed density of Monthly CC Return
    plt.subplot(223)
    data.plot.density(color=color)
    plt.title('Smoothed Density', fontsize=21)
    plt.xlabel('CC Return', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    # QQplot of Monthly CC Return
    p = plt.subplot(224)
    sm.qqplot(data, line='s', ax=p, color='b', markerfacecolor='w')
    plt.title('Normal Q-Q Plot', fontsize=21)
    plt.xlabel('Theoretical Quantiles', fontsize=18)
    plt.ylabel('Sample Quantiles', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    
    plt.show()

# calculate descriptive statistics
def descriptive_stats(data):
    # describe() generates descriptives (mean, std and quantiles)
    out = data.describe()
    other = pd.Series([data.var(), data.skew(), data.kurtosis()],
                      index=['var', 'skewness', 'kurtosis'])
    out = out.append(other)
    
    return out


# scatter plot for a given dataframe with two columns    
def scatter_plot(data, scale, color):
    plt.figure(figsize=(21, 12))
    plt.scatter(data[data.columns[0]], 
                data[data.columns[1]],
                c=color,
                edgecolor='None',
                s=scale,
                alpha=.3)
    plt.title('CC Return scatter plot about ' 
              + data.columns[0]
              + ' and '
              + data.columns[1], 
              fontsize=21)
    plt.xlabel(data.columns[0] + ' - CC Return', fontsize=18)
    plt.ylabel(data.columns[1] + ' - CC Return', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.grid(lw=.25)
    plt.show()
    
    
# calculate beta coefficient
def beta(stock, market_index):
    return np.cov(stock, market_index)[0][1] / np.var(market_index)


# function that plots beta
def beta_plot(data, colors):
    plt.figure(figsize=(21, 12))
    plt.subplots_adjust(hspace=.40)
    plt.rc('font', size=13)
    
    # market index plot
    plt.subplot(311)
    a = plt.plot(data[data.columns[0]])
    plt.setp(a, color=colors[0], linewidth=1.5)
    plt.title(data.columns[0], fontsize=21)
    plt.ylabel('USD ($)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    # stock adjusted close price plot
    plt.subplot(312)
    b = plt.plot(data[data.columns[1]])
    plt.setp(b, color=colors[1], linewidth=1.5)
    plt.title(data.columns[1], fontsize=21)
    plt.ylabel('USD ($)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    # beta plot
    plt.subplot(313)
    c = plt.plot(data[data.columns[2]])
    plt.setp(c, color=colors[2], linewidth=1.5)
    plt.title(data.columns[2], fontsize=21)
    plt.xlabel('Date (Year)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.show()
    

def plot_seasonal_decompose(stock_seasonal_dec, stock_name):
    plt.figure(figsize=(21, 12))
    plt.subplots_adjust(hspace=.5)
    
    plt.subplot(411)
    stock_seasonal_dec.observed.plot()
    plt.title(stock_name + ' Adjusted close price', fontsize=21)
    plt.xlabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(412)
    stock_seasonal_dec.trend.plot()
    plt.title('Trend', fontsize=21)
    plt.xlabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(413)
    stock_seasonal_dec.seasonal.plot()
    plt.title('Seasonal', fontsize=21)
    plt.xlabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.subplot(414)
    stock_seasonal_dec.resid.plot()
    plt.title('Resid', fontsize=21)
    plt.xlabel('Date', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.show()


# Forecast 
from sklearn.svm import SVR
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def forecast_svr(stock_name, start, end, kernel, gamma):
    # GET THE DATA FROM YAHOO FINANCE
    df = pdr.get_data_yahoo(stock_name, start, end)
    # GET THE ADJUSTED CLOSE PRICE
    df = df['Adj Close'].groupby(pd.Grouper(freq='M')).mean().to_frame()
    
    # VAR FOR PREDICTING 'l' MONTHS OUT INTO THE FUTURE
    forecast_out = 12
    # CREATE ANOTHER COLUMN FOR THE 'TARGET', SHIFTED UP BY 'N' MONTHS
    df['Target'] = df['Adj Close'].shift(-forecast_out)
    print('\n---------------- ' + stock_name + ' ----------------')
    print('Calculating...')    
    # CREATE THE INDIPENDET DATA SET X
    # convert the df into np array and remove the last 'l' rows
    X = np.array(df.drop(['Target'], 1))[:-forecast_out]
    
    # CREATE THE DEPENDENT DATA SET Y
    # convert the df into np array and remove the last 'l' rows
    y = np.array(df['Target'])[:-forecast_out]
    
    # SPLIT THE DATA INTO 96 MONTHS TRAINING AND 10 MONTHS TESTING
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=.09, 
                                                        random_state=42)

    # CREATE AND TRAIN THE SUPPORT VECTOR REGRESSOR
    svr_rbf = SVR(kernel=kernel, C=1e3, gamma=gamma) #radio basis function kernel
    svr_rbf.fit(X_train, y_train)
    
    # SHOW THE MODEL PREDICTION
    # get the last 'l' rows of the feature data set
    X_forecast = df.drop(['Target'], 1)[:-forecast_out]
    X_forecast = X_forecast.tail(forecast_out)
    X_forecast = np.array(X_forecast)
    
    model_pred = svr_rbf.predict(X_forecast)
    
    # VISUALIZE THE DATA
    valid = df[X.shape[0]:].copy(deep=False)
    valid['Target'] = model_pred
    valid.columns = ['Adj Close', 'Prediction']
    # valid.iloc[0,1] = None
    
    plt.figure(figsize=(21, 12))
    plt.title(stock_name + 
              ' Adjusted Close Price - SVR-' + kernel + ' Predictions', 
              fontsize=21)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj Close Price (USD)', fontsize=18)
    plt.plot(df['Adj Close'], color='k')
    # plt.plot(valid['Adj Close'], color='limegreen')
    plt.plot(valid['Prediction'], color='red')
    plt.axvspan(valid['Adj Close'].index[0], 
                valid['Adj Close'].index[-1], 
                facecolor='#98DCE8', 
                alpha=.2)
    plt.legend(['Observed', 'Forecast'], fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(lw=.25)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    
    plt.show()

    # Testing model confidence: Score -> coefficient R^2, best is 1.0
    print('Result:')
    print(valid)
    print()
    svr_confidence = svr_rbf.score(X_test, y_test)
    print ('R^2 Coefficient: ', round(svr_confidence, 4) , ' (confidence)')
    # Calculate the mean absolute error
    MAE = mean_absolute_error(valid['Adj Close'], valid['Prediction'])
    print('Mean Absolute Error (MAE): ', round(MAE, 4))
    # Calculate the mean squared error 
    MSE = mean_squared_error(valid['Adj Close'], valid['Prediction'])
    print('Mean Squared Error (MSE): ', round(MSE, 4))
    # Calculate Root mean square error
    RMSE = np.sqrt(MSE)
    print('Root Mean Squared Error (RMSE): ', round(RMSE, 4))

    return MAE, MSE, RMSE, svr_confidence
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
