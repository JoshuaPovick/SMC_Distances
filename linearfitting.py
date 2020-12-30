###############
### Fitting ###
###############

'''
Useful geometry functions for fitting linear models 
'''

import numpy as np

def mad(data):
    '''
    Calculate the Median Absolute Deviation (MAD) of data
    
    Input:
    -----
        data: values to calculate MAD of
    '''
    return np.median(np.absolute(data - np.median(data)))

####################################################################################

def add_noise(data, data_err, distribution = 'normal'):
    '''
    Add noise to data and return new values
    
    Input:
    -----
        data: data to add noise to
        data_err: 1d array-like object of errors for quant
        distribution: which distribution to use 'normal', 'poisson', 'uniform'    
    '''
    
    if distribution == 'normal':
        return np.random.normal(data,data_err)
    if distribution == 'poisson':
        return quant + np.random.poisson(data_err)
    if distribution == 'uniform':
        return np.random.uniform(-data_err+data,data+data_err)

####################################################################################

import statsmodels.api as sm

def ols_fit(x, y):
    '''
    Calculate OLS fit of a line making use of statsmodels.api
    
    Parameters:
    ----------
        x: x coordinates of line
        y: y coordinates of line
    
    Returns:
    -------
        m: slope of OLS line
        b: intercept of OLS line
    '''
    
    # fit model
    model = np.asarray(x).T #np.array([x]).T
    model = sm.add_constant(model)
    model_fit = sm.OLS(y,model).fit()
    mb = np.asarray([model_fit.params[1], model_fit.params[0]])
    err = np.asarray(model_fit.bse[::-1])    
    return mb, err

####################################################################################

from scipy.optimize import minimize
from scipy.stats import binned_statistic

def lin_modl(m,b,x):
    
    '''
    Slope-intercept form of a line
    
    Parameters:
    ----------
        m: slope of line
        x: x coordinate for line
        b: intercept of line
        
    Returns:
    -------
        y: coordinate y of line
    '''
    
    y = m*x + b
    return y

def lin_lnL(theta,x,y,x_err,y_err):
    
    '''
    Log likelihood for linmodl
    
    Parameters:
    ----------
        theta: parameters to plug into linmodl (m,b)
        x: x coordinate
        y: y coordinate 
        x_err: errors for x
        y_err: errors for y
    
    Returns:
    -------
        lnl: log likelihood of particluar linear model
    '''
    
    m, b = theta
    modl = linmodl(m,b,x)
    inv_sig2 = np.reciprocal(np.square(x_err)+np.square(y_err))
    lnl = -0.5 * np.sum(np.multiply(np.square((y - modl)),inv_sig2) - np.log(inv_sig2/(2*np.pi)))
    return lnl

def lin_mle_fit(x, y, y_err, x_err, x_bins = None):
    
    '''
    Do a MLE fit of a linear model (y = mx + b). If bins != None, then fit to medians of the x_bins
    
    Input:
    -----
        x: x coordinate
        y: y coordinate
        x_err: errors for x with len(x)
        y_err: errors for y with len(x)
        x_bins: bins for median fit ex. x_bins = np.arange(np.floor(np.min(x)),np.ceil(np.max(x)),1.0) 
        
    Output:
    ------
        lin_m: slope of line fit
        lin_b: intercept of line fit
    '''
    
    if bins == None:
        # Initialize ML calculation
        m_guess = (y[1] - y[0])/(x[1] - x[0])
        b_guess = y[0]
        
        #minimize ML
        nll = lambda *args: -lin_lnL(*args)
        guess = np.array([m_guess, b_guess])
        soln = minimize(nll, med_guess, args = (x, y, x_err, y_err))
        lin_m, lin_b = soln.x
        
        return lin_m, lin_b
        
    else:    
        #Calculate median value for each bin
        bin_stats, _, bin_ind = binned_statistic(x, y, statistic = 'median', bins = x_bins)
        
        uniq = np.unique(bin_ind)
        
        med_x_err = []
        med_y_err = []
        for i in range(len(uniq)):
            single_bin = np.where(bin_ind == uniq[i])
            med_x_err.append(mad(np.asarray(x)[single_bin]))
            med_y_err.append(mad(np.asarray(y)[single_bin]))
    
#         #Calculate spread (MAD) in values in each bin
#         bin_stats_err, _, _ = binned_statistic(x, y, statistic = lambda s: np.median(np.absolute(s - np.median(s))),
#                                                bins = bins)
    
        #Initialize ML calculation
        med_x = bin_stats[0]
        med_y = bin_stats[1]
        med_x_err = bin_stats_err[0]
        med_y_err = bin_stats_err[1]
    
        med_m_guess = (med_y[1] - med_y[0])/(med_x[1] - med_x[0])
        med_b_guess = med_y[0]
    
        # minimize ML and find slopes and intercepts
        nll = lambda *args: -lin_lnL(*args)
        med_guess = np.array([med_m_guess, med_b_guess])
        med_soln = minimize(nll, med_guess, args = (med_x, med_y, med_x_err, med_y_err))
        med_lin_m, med_lin_b = med_soln.x
    
        return med_lin_m, med_lin_b

####################################################################################