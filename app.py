###############################################################################
############### BUSINESS INTELLIGENCE PER I SERVIZI FINANZIARI ################
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Progetto d'esame - anno accademico 2019/2020
# @author: 807160 Raccuglia Davide
# Stocks: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), Google (GOOG)
# Index: S&P500 (^GSPC)

###############################################################################
###############################################################################

# Import dependencies
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
import statsmodels.api as sm
import functions as f
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Global
colors = ['slategray', 'dodgerblue', 'darkorange', 'red', 'green']

###############################################################################
###################### STREAM: FROM JAN 2010 TO OCT 2019 ######################

start = '2010-01-01'
end = '2019-10-31'

###############################################################################
######################### GET DATA FROM YAHOO! FINANCE ########################

AAPL_df  = pdr.get_data_yahoo('AAPL', start, end)
MSFT_df  = pdr.get_data_yahoo('MSFT', start, end)
AMZN_df  = pdr.get_data_yahoo('AMZN', start, end)
GOOG_df  = pdr.get_data_yahoo('GOOG', start, end)
SP500_df = pdr.get_data_yahoo('^GSPC', start, end)

###############################################################################
######################## CHANGE TO MONTHLY GRANULARITY ########################

AAPL_df_M  = AAPL_df.groupby(pd.Grouper(freq='M')).mean()
MSFT_df_M  = MSFT_df.groupby(pd.Grouper(freq='M')).mean()
AMZN_df_M  = AMZN_df.groupby(pd.Grouper(freq='M')).mean()
GOOG_df_M  = GOOG_df.groupby(pd.Grouper(freq='M')).mean()
SP500_df_M = SP500_df.groupby(pd.Grouper(freq='M')).mean()

###############################################################################
######################## GET ONLY THE ADJ CLOSE COLUMN ########################

AAPL_adj_close  = AAPL_df_M['Adj Close']
MSFT_adj_close  = MSFT_df_M['Adj Close']
AMZN_adj_close  = AMZN_df_M['Adj Close']
GOOG_adj_close  = GOOG_df_M['Adj Close']
SP500_adj_close = SP500_df_M['Adj Close']

###############################################################################
#################### PLOTTING THE STOCK TAKEN INTO ANALYSIS ###################

f.plot_data(AAPL_adj_close, 
            'Adjusted close price of AAPL', 
            ['Date (year)', 'Adj close price (USD)'], 
            [colors[0]])

f.plot_data(MSFT_adj_close, 
            'Adjusted close price of MSFT', 
            ['Date (year)', 'Adj close price (USD)'], 
            [colors[1]])

f.plot_data(AMZN_adj_close, 
            'Adjusted close price of AMZN', 
            ['Date (year)', 'Adj close price (USD)'], 
            [colors[2]])

f.plot_data(GOOG_adj_close, 
            'Adjusted close price of GOOG',  
            ['Date (year)', 'Adj close price (USD)'], 
            [colors[3]])

###############################################################################
############ MERGE ALL STOCKS DATA INTO A DATA OBJECT AND PLOT IT #############

STOCKS_adj_close = pd.concat([AAPL_adj_close, 
                              MSFT_adj_close, 
                              AMZN_adj_close, 
                              GOOG_adj_close], axis=1)

STOCKS_adj_close.columns = ['AAPL', 'MSFT', 'AMZN', 'GOOG']

f.plot_data(STOCKS_adj_close, 
            'Adj Close Price of AAPL, MSFT, AMZN, GOOG', 
            ['Date (years)', 'Adjusted Close Price (USD)'], 
            colors[:4])

###############################################################################
############## MERGE S&P500 WITH EACH STOCK INTO DIFFERENT OBJECT #############

AAPL_SP500_adjclose = pd.concat([AAPL_adj_close, SP500_adj_close], axis=1)
AAPL_SP500_adjclose.columns = ['AAPL', '^GSPC']

MSFT_SP500_adjclose = pd.concat([MSFT_adj_close, SP500_adj_close], axis=1)
MSFT_SP500_adjclose.columns = ['MSFT', '^GSPC']

AMZN_SP500_adjclose = pd.concat([AMZN_adj_close, SP500_adj_close], axis=1)
AMZN_SP500_adjclose.columns = ['AMZN', '^GSPC']

GOOG_SP500_adjclose = pd.concat([GOOG_adj_close, SP500_adj_close], axis=1)
GOOG_SP500_adjclose.columns = ['GOOG', '^GSPC']

###############################################################################
############ COMPUTE AND PLOT SIMPLE MONTHLY RETURNS OF EACH ASSET ############

AAPL_SP500_rtn = f.gross_return(AAPL_SP500_adjclose)
f.plot_data(AAPL_SP500_rtn,
            'Gross Return of AAPL and ^GSPC',
            ['Date', 'Gross Return'],
            [colors[0], colors[4]])

MSFT_SP500_rtn = f.gross_return(MSFT_SP500_adjclose)
f.plot_data(MSFT_SP500_rtn,
            'Gross Return of MSFT and ^GSPC',
            ['Date', 'Gross Return'],
            [colors[1], colors[4]])

AMZN_SP500_rtn = f.gross_return(AMZN_SP500_adjclose)
f.plot_data(AMZN_SP500_rtn,
            'Gross Return of AMZN and ^GSPC',
            ['Date', 'Gross Return'],
            [colors[2], colors[4]])

GOOG_SP500_rtn = f.gross_return(GOOG_SP500_adjclose)
f.plot_data(GOOG_SP500_rtn,
            'Gross Return of GOOG and ^GSPC',
            ['Date', 'Gross Return'],
            [colors[3], colors[4]])

###############################################################################
############# COMPUTE AND PLOT CC MONTHLY RETURNS OF EACH ASSET ###############

AAPL_SP500_ccrtn = f.cc_return(AAPL_SP500_adjclose)
f.plot_data(AAPL_SP500_rtn,
            'CC Return of AAPL and ^GSPC',
            ['Date', 'CC Return'],
            [colors[0], colors[4]])

MSFT_SP500_ccrtn = f.cc_return(MSFT_SP500_adjclose)
f.plot_data(MSFT_SP500_rtn,
            'CC Return of MSFT and ^GSPC',
            ['Date', 'CC Return'],
            [colors[1], colors[4]])

AMZN_SP500_ccrtn = f.cc_return(AMZN_SP500_adjclose)
f.plot_data(AMZN_SP500_rtn,
            'CC Return of AMZN and ^GSPC',
            ['Date', 'CC Return'],
            [colors[2], colors[4]])

GOOG_SP500_ccrtn = f.cc_return(GOOG_SP500_adjclose)
f.plot_data(GOOG_SP500_ccrtn,
            'CC Return of GOOG and ^GSPC',
            ['Date', 'CC Return'],
            [colors[3], colors[4]])

###############################################################################
################ COMPARING CC MONTHLY RETURNS OF ALL ASSETS ###################

f.density_plot(AAPL_SP500_ccrtn, [colors[0], colors[4]])
f.density_plot(MSFT_SP500_ccrtn, [colors[1], colors[4]])
f.density_plot(AMZN_SP500_ccrtn, [colors[2], colors[4]])
f.density_plot(GOOG_SP500_ccrtn, [colors[3], colors[4]])

###############################################################################
############### DIAGNOSTIC PLOTS FOR EACH SERIES CC RETURN ####################

f.diagnostic_plots(AAPL_SP500_ccrtn['AAPL'], colors[0])
f.diagnostic_plots(MSFT_SP500_ccrtn['MSFT'], colors[1])
f.diagnostic_plots(AMZN_SP500_ccrtn['AMZN'], colors[2])
f.diagnostic_plots(GOOG_SP500_ccrtn['GOOG'], colors[3])

# compare histograms
plt.figure(figsize=(21, 12))
plt.subplots_adjust(hspace=.35, wspace=.15)
plt.rc('font', size=13)

plt.subplot(221)
plt.hist(AAPL_SP500_ccrtn['AAPL'], 
         density=True, bins=15, 
         color='#DEDEDE')
plt.title(AAPL_SP500_ccrtn['AAPL'].name + 
          ' - Distribution of CC Return', fontsize=21)
AAPL_SP500_ccrtn['AAPL'].plot.density(color=colors[0])
plt.xlabel('CC Return', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.subplot(222)
plt.hist(MSFT_SP500_ccrtn['MSFT'], 
         density=True, bins=15, 
         color='#DEDEDE')
plt.title(MSFT_SP500_ccrtn['MSFT'].name + 
          ' - Distribution of CC Return', fontsize=21)
MSFT_SP500_ccrtn['MSFT'].plot.density(color=colors[1])
plt.ylabel('Density', fontsize=18)
plt.xlabel('CC Return', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.subplot(223)
plt.hist(AMZN_SP500_ccrtn['AMZN'], 
         density=True, bins=15, 
         color='#DEDEDE')
plt.title(AMZN_SP500_ccrtn['AMZN'].name + 
          ' - Distribution of CC Return', fontsize=21)
AMZN_SP500_ccrtn['AMZN'].plot.density(color=colors[2])
plt.xlabel('CC Return', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.subplot(224)
plt.hist(GOOG_SP500_ccrtn['GOOG'], 
         density=True, bins=15, 
         color='#DEDEDE')
plt.title(GOOG_SP500_ccrtn['GOOG'].name + 
          ' - Distribution of CC Return', fontsize=21)
GOOG_SP500_ccrtn['GOOG'].plot.density(color=colors[3])
plt.ylabel('Density', fontsize=18)
plt.xlabel('CC Return', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.show()

# compare boxplots
plt.figure(figsize=(21, 12))
plt.subplots_adjust(hspace=.35, wspace=.15)
plt.rc('font', size=13)

plt.subplot(221)
plt.boxplot(AAPL_SP500_ccrtn['AAPL'])
AAPL_SP500_ccrtn['AAPL'].to_frame().boxplot()
plt.title('CC Return Boxplot', fontsize=21)
plt.ylabel('CC Return', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

plt.subplot(222)
plt.boxplot(MSFT_SP500_ccrtn['MSFT'])
MSFT_SP500_ccrtn['MSFT'].to_frame().boxplot()
plt.title('CC Return Boxplot', fontsize=21)
plt.ylabel('CC Return', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

plt.subplot(223)
plt.boxplot(AMZN_SP500_ccrtn['AMZN'])
AMZN_SP500_ccrtn['AMZN'].to_frame().boxplot()
plt.title('CC Return Boxplot', fontsize=21)
plt.ylabel('CC Return', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

plt.subplot(224)
plt.boxplot(GOOG_SP500_ccrtn['GOOG'])
GOOG_SP500_ccrtn['GOOG'].to_frame().boxplot()
plt.title('CC Return Boxplot', fontsize=21)
plt.ylabel('CC Return', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

# compare Q-Q plots
plt.figure(figsize=(21, 12))
plt.subplots_adjust(hspace=.35, wspace=.15)
plt.rc('font', size=13)

a = plt.subplot(221)
sm.qqplot(AAPL_SP500_ccrtn['AAPL'],
           line='s', ax=a,
           color='b', markerfacecolor='w')
plt.title('AAPL Normal Q-Q Plot', fontsize=21)
plt.xlabel('Theoretical Quantiles', fontsize=18)
plt.ylabel('Sample Quantiles', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

m = plt.subplot(222)
sm.qqplot(MSFT_SP500_ccrtn['MSFT'],
           line='s', ax=m,
           color='b', markerfacecolor='w')
plt.title('MSFT Normal Q-Q Plot', fontsize=21)
plt.xlabel('Theoretical Quantiles', fontsize=18)
plt.ylabel('Sample Quantiles', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

z = plt.subplot(223)
sm.qqplot(AMZN_SP500_ccrtn['AMZN'],
           line='s', ax=z,
           color='b', markerfacecolor='w')
plt.title('AMZN Normal Q-Q Plot', fontsize=21)
plt.xlabel('Theoretical Quantiles', fontsize=18)
plt.ylabel('Sample Quantiles', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

g = plt.subplot(224)
sm.qqplot(GOOG_SP500_ccrtn['GOOG'],
           line='s', ax=g,
           color='b', markerfacecolor='w')
plt.title('GOOG Normal Q-Q Plot', fontsize=21)
plt.xlabel('Theoretical Quantiles', fontsize=18)
plt.ylabel('Sample Quantiles', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(lw=.25)

###############################################################################
#################### UNIVARIATE DESCRIPTIVE STATISTICS ########################

# compute arithmetic mean, variance, standard deviation, skewness and kurtosis
# of Stock CC retun data colletion

# particularly, skewness measures the symmetry (or asymmetry) of a distribution
# around its mean and we have that if it is:
# = 0, then the distribution is symmetric, so it's a normal distribution
# < 0, then the distribution has longer left tail than the normal one
# > 0, then the distribution has longer right tail than the normal one

# kurtosis, instead, tell us something about the tail of distribution,
# i.e. if it is:
# = 0, then the tail is like that of a normal distribution
# < 0, then the tail is thinner than that of a normal distribution
# > 0, then the tail is thicker than that of a normal distribution

AAPL_stats = f.descriptive_stats(AAPL_SP500_ccrtn['AAPL'])
MSFT_stats = f.descriptive_stats(MSFT_SP500_ccrtn['MSFT'])
AMZN_stats = f.descriptive_stats(AMZN_SP500_ccrtn['AMZN'])
GOOG_stats = f.descriptive_stats(GOOG_SP500_ccrtn['GOOG'])

###############################################################################
################# BIVARIATE DISTRIBUTION ANALYSIS (MONTHLY) ###################

# merge each return series in one object
ALL_ccrtn = pd.concat([AAPL_SP500_ccrtn['AAPL'],
                       MSFT_SP500_ccrtn['MSFT'],
                       AMZN_SP500_ccrtn['AMZN'],
                       GOOG_SP500_ccrtn['GOOG'],
                       GOOG_SP500_ccrtn['^GSPC']], axis=1)

# Compute the covariance and correlation matrices of the returns of all assets
cov_monthly  = ALL_ccrtn.cov()
corr_monthly = ALL_ccrtn.corr(method='pearson')

# scatter plot of pair (stock, index) cc return
f.scatter_plot(AAPL_SP500_ccrtn, 250, colors[0])
f.scatter_plot(MSFT_SP500_ccrtn, 250, colors[1])
f.scatter_plot(AMZN_SP500_ccrtn, 250, colors[2])
f.scatter_plot(GOOG_SP500_ccrtn, 250, colors[3])


# multiple scatter plot
scat_mat = pd.plotting.scatter_matrix(ALL_ccrtn, 
                                      alpha=.4,
                                      range_padding=.2,
                                      figsize=(21,12),
                                      diagonal='kde',
                                      marker='o', 
                                      edgecolor='None')
for ax in scat_mat.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)

scat_mat[0,0].set_yticklabels(['-0.1', '0.0', '0.1'], fontsize=14)
scat_mat[1,0].set_yticklabels(['', '-0.1', '0.0', '0.1'], fontsize=14)
scat_mat[2,0].set_yticklabels(['', '-0.1', '0.0', '0.1', '0.2'], fontsize=14)
scat_mat[3,0].set_yticklabels(['', '-0.1', '0.0', '0.1'], fontsize=14)
scat_mat[4,0].set_yticklabels(['', '-0.1', '-0.05', 
                                   '0.0', '0.05'], fontsize=14)
scat_mat[4,0].set_xticklabels(['', '-0.1', '0.0', '0.1'], fontsize=14)
scat_mat[4,1].set_xticklabels(['', '-0.1', '-0.05', 
                                   '0.0', '0.05', 
                                   '0.1', '0.15'], fontsize=14)
scat_mat[4,2].set_xticklabels(['', '-0.1', '0.0', '0.1', '0.2'], fontsize=14)
scat_mat[4,3].set_xticklabels(['', '-0.1', '-0.05', 
                                   '0.0', '0.05', 
                                   '0.1', '0.15'], fontsize=14)
scat_mat[4,4].set_xticklabels(['', '-0.1', '-0.05', 
                                   '0.0', '0.05'], fontsize=14)
# plt.suptitle('Correlation of CC Returns (Monthly)', fontsize=(34))

###############################################################################
################# BIVARIATE DISTRIBUTION ANALYSIS (DAILY) #####################

# change to daily granularity
AAPL_df_D  = AAPL_df.groupby(pd.Grouper(freq='D')).mean().dropna()
MSFT_df_D  = MSFT_df.groupby(pd.Grouper(freq='D')).mean().dropna()
AMZN_df_D  = AMZN_df.groupby(pd.Grouper(freq='D')).mean().dropna()
GOOG_df_D  = GOOG_df.groupby(pd.Grouper(freq='D')).mean().dropna()
SP500_df_D = SP500_df.groupby(pd.Grouper(freq='D')).mean().dropna()

# get the adj close price
AAPL_adj_close  = AAPL_df_D['Adj Close']
MSFT_adj_close  = MSFT_df_D['Adj Close']
AMZN_adj_close  = AMZN_df_D['Adj Close']
GOOG_adj_close  = GOOG_df_D['Adj Close']
SP500_adj_close = SP500_df_D['Adj Close']

# merge each asset with the index
AAPL_SP500_adjclose = pd.concat([AAPL_adj_close, SP500_adj_close], axis=1)
AAPL_SP500_adjclose.columns = ['AAPL', '^GSPC']

MSFT_SP500_adjclose = pd.concat([MSFT_adj_close, SP500_adj_close], axis=1)
MSFT_SP500_adjclose.columns = ['MSFT', '^GSPC']

AMZN_SP500_adjclose = pd.concat([AMZN_adj_close, SP500_adj_close], axis=1)
AMZN_SP500_adjclose.columns = ['AMZN', '^GSPC']

GOOG_SP500_adjclose = pd.concat([GOOG_adj_close, SP500_adj_close], axis=1)
GOOG_SP500_adjclose.columns = ['GOOG', '^GSPC']

# calculate cc returns
AAPL_SP500_ccrtn = f.cc_return(AAPL_SP500_adjclose)
MSFT_SP500_ccrtn = f.cc_return(MSFT_SP500_adjclose)
AMZN_SP500_ccrtn = f.cc_return(AMZN_SP500_adjclose)
GOOG_SP500_ccrtn = f.cc_return(GOOG_SP500_adjclose)

# merge each return series in one object
ALL_ccrtn = pd.concat([AAPL_SP500_ccrtn['AAPL'],
                       MSFT_SP500_ccrtn['MSFT'],
                       AMZN_SP500_ccrtn['AMZN'],
                       GOOG_SP500_ccrtn['GOOG'],
                       GOOG_SP500_ccrtn['^GSPC']], axis=1)

# Compute the covariance and correlation matrices of the returns of all assets
cov_daily  = ALL_ccrtn.cov()
corr_daily = ALL_ccrtn.corr(method='pearson')

# scatter plot of pair (stock, index) cc return
f.scatter_plot(AAPL_SP500_ccrtn, 250, colors[0])
f.scatter_plot(MSFT_SP500_ccrtn, 250, colors[1])
f.scatter_plot(AMZN_SP500_ccrtn, 250, colors[2])
f.scatter_plot(GOOG_SP500_ccrtn, 250, colors[3])

# multiple scatter plot
scat_mat = pd.plotting.scatter_matrix(ALL_ccrtn, 
                                      alpha=.2,
                                      range_padding=.1,
                                      figsize=(21,12),
                                      diagonal='kde',
                                      marker='o', 
                                      edgecolor='None')
for ax in scat_mat.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)

scat_mat[0,0].set_yticklabels(['-0.1', '-0.05', '0.0', '0.05'], fontsize=14)
scat_mat[1,0].set_yticklabels(['', '-0.1', '-0,05', 
                                    '0.0', '0.05', '0.1'], fontsize=14)
scat_mat[2,0].set_yticklabels(['', '-0.1', '0.0', '0.1'], fontsize=14)
scat_mat[3,0].set_yticklabels(['', '0.0', '0.1'], fontsize=14)
scat_mat[4,0].set_yticklabels(['', '-0.05', '-0.025', 
                                    '0.0', '0.025', '0.05'], fontsize=14)
scat_mat[4,0].set_xticklabels(['', '-0.1', '-0.05', 
                                    '0.0', '0.05'], fontsize=14)
scat_mat[4,1].set_xticklabels(['', '-0.1', '-0.05', 
                                    '0.0', '0.05', '0.1'], fontsize=14)
scat_mat[4,2].set_xticklabels(['', '-0.1', '-0.05', '0.0', 
                                    '0.05', '0.1', '0.15'], fontsize=14)
scat_mat[4,3].set_xticklabels(['', '-0.05', '0.0', 
                                    '0.05', '0.1', '0.15'], fontsize=14)
scat_mat[4,4].set_xticklabels(['', '-0.06', '-0.04', '-0.02', 
                                    '0.0', '0.02', '0.04'], fontsize=14)
# plt.suptitle('Correlation of CC Returns (Daily)', fontsize=(34))

###############################################################################
############################## BETA COMPUTATION ###############################

print('################ BETA COMPUTATION ################')

# get monthly adjusted close price of each stock (and index)
AAPL_adj_close  = AAPL_df_M['Adj Close'].dropna()
MSFT_adj_close  = MSFT_df_M['Adj Close'].dropna()
AMZN_adj_close  = AMZN_df_M['Adj Close'].dropna()
GOOG_adj_close  = GOOG_df_M['Adj Close'].dropna()
SP500_adj_close = SP500_df_M['Adj Close'].dropna()

# calculate the monthly cc returns
AAPL_ccrtn = f.cc_return(AAPL_adj_close)
AAPL_ccrtn.name = 'CC Return'
MSFT_ccrtn = f.cc_return(MSFT_adj_close)
MSFT_ccrtn.name = 'CC Return'
AMZN_ccrtn = f.cc_return(AMZN_adj_close)
AMZN_ccrtn.name = 'CC Return'
GOOG_ccrtn = f.cc_return(GOOG_adj_close)
GOOG_ccrtn.name = 'CC Return'
SP500_ccrtn = f.cc_return(SP500_adj_close)
SP500_ccrtn.name = 'CC Return'

# calculate beta coefficient of each stock using a time window of 10 years
beta_AAPL = f.beta(AAPL_ccrtn, SP500_ccrtn)
beta_MSFT = f.beta(MSFT_ccrtn, SP500_ccrtn)
beta_AMZN = f.beta(AMZN_ccrtn, SP500_ccrtn)
beta_GOOG = f.beta(GOOG_ccrtn, SP500_ccrtn)

# create pandas series to store beta values
AAPL_beta_ts = pd.Series(dtype='float64')
MSFT_beta_ts = pd.Series(dtype='float64')
AMZN_beta_ts = pd.Series(dtype='float64')
GOOG_beta_ts = pd.Series(dtype='float64')

# beta time window (months)
time_window = 36

period_time = len(SP500_ccrtn)
# the start time is the 21th month, which is the value at 
# 20th index of the CC return series
start_time = time_window 

for i in range(start_time, period_time):
    beta_aapl = f.beta(AAPL_ccrtn[i-time_window : i-1], 
                        SP500_ccrtn[i-time_window : i-1])
    beta_msft = f.beta(MSFT_ccrtn[i-time_window : i-1], 
                        SP500_ccrtn[i-time_window : i-1])
    beta_amzn = f.beta(AMZN_ccrtn[i-time_window : i-1], 
                        SP500_ccrtn[i-time_window : i-1])
    beta_goog = f.beta(GOOG_ccrtn[i-time_window : i-1], 
                        SP500_ccrtn[i-time_window : i-1])
    
    AAPL_beta_ts = AAPL_beta_ts.append(
        pd.Series(beta_aapl, index=[AAPL_ccrtn.index[i]])
        )
    MSFT_beta_ts = MSFT_beta_ts.append(
        pd.Series(beta_msft, index=[MSFT_ccrtn.index[i]])
        )
    AMZN_beta_ts = AMZN_beta_ts.append(
        pd.Series(beta_amzn, index=[AMZN_ccrtn.index[i]])
        )
    GOOG_beta_ts = GOOG_beta_ts.append(
        pd.Series(beta_goog, index=[GOOG_ccrtn.index[i]])
        )
        
    # Print on the console the beta values computed
    print('######### Beta Time Window #########')
    print("Start time: " + 
          str(SP500_ccrtn.index[i-time_window]).split(' ')[0])
    print("End time: " + str(SP500_ccrtn.index[i-1]).split(' ')[0])
    print('########### Date for beta ##########')
    print("Date: " + str(SP500_ccrtn.index[i]).split(' ')[0])
    print("AAPL beta:" + str(beta_aapl))
    print("MSFT beta:" + str(beta_msft))
    print("AMZN beta:" + str(beta_amzn))
    print("GOOG beta:" + str(beta_goog))

# rename the series column
AAPL_beta_ts.name = 'AAPL_beta'
MSFT_beta_ts.name = 'MSFT_beta'
AMZN_beta_ts.name = 'AMZN_beta'
GOOG_beta_ts.name = 'GOOG_beta'

# add NANs to each beta series to reset them lengths
AAPL_beta_ts = AAPL_beta_ts.append(
    pd.Series(np.repeat(None, (time_window)), 
                 index=SP500_ccrtn.index[0:time_window])
    ).sort_index()
MSFT_beta_ts = MSFT_beta_ts.append(
    pd.Series(np.repeat(None, (time_window)), 
                 index=SP500_ccrtn.index[0:time_window])
    ).sort_index()
AMZN_beta_ts = AMZN_beta_ts.append(
    pd.Series(np.repeat(None, (time_window)), 
                 index=SP500_ccrtn.index[0:time_window])
    ).sort_index()
GOOG_beta_ts = GOOG_beta_ts.append(
    pd.Series(np.repeat(None, (time_window)), 
                 index=SP500_ccrtn.index[0:time_window])
    ).sort_index()

# plot beta
AAPL = pd.concat([SP500_adj_close, AAPL_adj_close, AAPL_beta_ts], axis=1)
AAPL.columns = ['SP500 Adjusted Close Price', 
                'AAPL Adjusted Close Price', 
                'AAPL beta']
f.beta_plot(AAPL, ['green', 'slategrey', '#2ee6be'])

MSFT = pd.concat([SP500_adj_close, MSFT_adj_close, MSFT_beta_ts], axis=1)
MSFT.columns = ['SP500 Adjusted Close Price', 
                'MSFT Adjusted Close Price', 
                'MSFT beta']
f.beta_plot(MSFT, ['green', 'dodgerblue', '#2ee6be'])

AMZN = pd.concat([SP500_adj_close, AMZN_adj_close, AMZN_beta_ts], axis=1)
AMZN.columns = ['SP500 Adjusted Close Price', 
                'AMZN Adjusted Close Price', 
                'AMZN beta']
f.beta_plot(AMZN, ['green', 'darkorange', '#2ee6be'])

GOOG = pd.concat([SP500_adj_close, GOOG_adj_close, GOOG_beta_ts], axis=1)
GOOG.columns = ['SP500 Adjusted Close Price', 
                'GOOG Adjusted Close Price', 
                'GOOG beta']
f.beta_plot(GOOG, ['green', 'red', '#2ee6be'])

###############################################################################
################################# FORECASTING #################################

# Time series decomposition of each asset adjusted close price
AAPL_sd = sm.tsa.seasonal_decompose(AAPL_df_M['Adj Close'],
                                    period=12,
                                    model='additive')
MSFT_sd = sm.tsa.seasonal_decompose(MSFT_df_M['Adj Close'],
                                    period=12,
                                    model='additive')
AMZN_sd = sm.tsa.seasonal_decompose(AMZN_df_M['Adj Close'],
                                    period=12,
                                    model='additive')
GOOG_sd = sm.tsa.seasonal_decompose(GOOG_df_M['Adj Close'],
                                    period=12,
                                    model='additive')

# plot the seasonality
f.plot_seasonal_decompose(AAPL_sd, 'AAPL')
f.plot_seasonal_decompose(MSFT_sd, 'MSFT')
f.plot_seasonal_decompose(AMZN_sd, 'AMZN')
f.plot_seasonal_decompose(GOOG_sd, 'GOOG')

# Infering forecasting model using SVR to predict returns for every financial
# instruments considered (AAPL, MSFT, AMZM, GOOG)
print()
print('################ FORECASTING - SVR RBF Kernel ################')

AAPL_fc_MAE, AAPL_fc_MSE, AAPL_fc_RMSE, AAPL_fc_confidence = f.forecast_svr(
    'AAPL', start, end, 'rbf', 'scale')

MSFT_fc_MAE, MSFT_fc_MSE, MSFT_fc_RMSE, MSFT_fc_confidence = f.forecast_svr(
    'MSFT', start, end, 'rbf', 'scale')

AMZN_fc_MAE, AMZN_fc_MSE, AMZN_fc_RMSE, AMZN_fc_confidence = f.forecast_svr(
    'AMZN', start, end, 'rbf', 'scale')

GOOG_fc_MAE, GOOG_fc_MSE, GOOG_fc_RMSE, GOOG_fc_confidence = f.forecast_svr(
    'GOOG', start, end, 'rbf', .01)

###############################################################################
############################ PORTFOLIO MANAGEMENT #############################

# get data to calculate the beta value
start = '2010-01-01'
end = '2019-10-31'

df = pd.DataFrame()
assets = ['AAPL', 'MSFT', 'AMZN', 'GOOG', '^GSPC']

for stock in assets:
    df[stock] = pdr.get_data_yahoo(stock, start, end)['Adj Close']
    
df = df.groupby(pd.Grouper(freq='M')).mean() 
df.head()

df_returns = np.log(df / df.shift(1)).dropna() # get CC return of each stock

# get data for the bought_assets
pf_start = '2019-01-01'
pf = pd.DataFrame() # using for portfolio management

for stock in assets:
    pf[stock] = pdr.get_data_yahoo(stock, pf_start, end)['Adj Close']

pf.head()

pf_returns = np.log(pf / pf.shift(1)).dropna()

# calculate beta coefficient
beta = dict()
for stock in assets[:4]:
    beta[stock] = np.cov(df_returns[stock], 
                         df_returns['^GSPC'])[0][1] / np.var(df_returns['^GSPC'])
    
# sum all the beta values
beta_sum = 0
for asset in beta:
    beta_sum = beta_sum + beta[asset]
    
# beta_inv_sum = 0
# for stock in beta:
#     beta_inv_sum = beta_inv_sum + ( beta_sum / beta[stock] )

# set weight according to the beta value (higher the beta, higher the weight)
weights = dict()
for stock in beta:
    # weights[stock] = ( beta_sum / beta[stock] ) / beta_inv_sum
    weights[stock] = beta[stock] / beta_sum

def portfolio_management(weights, portfolio_df, budget, transac_cost): 
    # distribute the budget according to the weights
    distribution = dict() # contains the amount of money invested on each asset
    for stock in weights: 
        distribution[stock] = weights[stock] * budget
    
    # calculate the number of share per asset to buy 
    share_volumes = dict()
    for stock in distribution:
        share_volumes[stock] = np.floor(distribution[stock] / pf[stock][0])
        # share_volumes[stock] = int(distribution[stock] / pf[stock][0])
    
    # calculate each asset value in bought_assets
    bought_assets = dict()
    total_invested = 0 # total money spent to buy all assets
    sold_assets = dict() # sold the assets of bought_assets after 10 months
    for stock in share_volumes:
        bought_assets[stock] = share_volumes[stock] * pf[stock][0]
        total_invested = total_invested + bought_assets[stock]
        sold_assets[stock] = share_volumes[stock] * pf[stock][-1] * transaction_cost
    
    # calculate the bought_assets return
    portfolio_returns = dict()
    for stock in bought_assets:
        if (bought_assets[stock] > 0):
            portfolio_returns[stock] = (( sold_assets[stock] - 
                                          bought_assets[stock] ) / 
                                         bought_assets[stock])
    
    # new var for the weights just for printing on console
    w = dict()
    for stock in weights:
        w[stock] = round(weights[stock], 4)
        
    print('Assets weights distribution:\n' +  str( w ) + '\n')
    print('Distribution ($):\n' +  str( distribution ) + '\n')
    
    for stock in bought_assets:
        if (bought_assets[stock] > 0):
            print( stock + ': Opening price on ' 
                   + pf.index[0].strftime('%Y-%m-%d') + ' $' + 
                   str(round(bought_assets[stock] / share_volumes[stock], 2) ))
            
            print( stock + ': Closing price on ' 
                   + pf.index[-1].strftime('%Y-%m-%d') + ' $' + 
                   str(round(sold_assets[stock] / share_volumes[stock], 2) ))
            
            print( stock + ': Owned shares', share_volumes[stock] )
            
            print( stock + ': Return', round(portfolio_returns[stock], 4) )
            
            print( stock + ': P&L $' + str(round(sold_assets[stock] - 
                                           bought_assets[stock], 2) ))
            print()

budget = 50000 
transaction_cost = 0.95 # that is the 5% of the amount 

print()
print('##### PORTFOLIO MANAGEMENT #####')
portfolio_management(weights, pf, budget, transaction_cost)

# removing useless variables
del (asset, 
     beta_aapl, 
     beta_msft, 
     beta_amzn, 
     beta_goog, 
     stock,
     ax,
     a,
     g, 
     i,
     m, 
     z)






















