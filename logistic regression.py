# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:14:44 2021

@author: Joey
"""
import numpy as np
import pandas as pd


## attempt at a from-scratch implementation logistic regression

## import data 
diabetes_data = pd.read_csv('diabetes.csv')
# scale the data to [0,1] to help with faster convergence
scaled_diabetes_data = (diabetes_data - diabetes_data.min())/(diabetes_data.max()
                                                              - diabetes_data.min())
# helper function to calculate logit function
def logit(x):
    return 1/(1+ np.exp(x))

# prediction for a set of predictors and model parameters, x and t
def prediction(x,t):
    #print(x,t)
    return logit(-1*np.dot(np.concatenate(([1], x)),t))

# helper to get values for a given row, returns pair of (response, predictors)
def get_yx(row):
    return (row[-1], row[0:-1])

# log likelihood function for a given model t over data d
# n predictors, m data points
def log_likelihood(t, d, n, m):
    #find the contribution of each data point, put these into a list
    # then sum over the list for the total log-likelihood
    contributions=[]
    for i in range(m):
        yx = get_yx(d,i)
        contributions.append(np.dot(t,[1] + yx[1])*(yx[0] - 1) 
                             - np.log(1+np.exp(-1*np.dot(t,[1] + yx[1]))))
    return(np.sum(contributions))


## find model coefficients with stochastic gradient ascent
## to maximize log-likelihood

def stochastic_gradient(t,yx,n):
    sg = [yx[0] - prediction(yx[1],t)]
    for i in range(0,n):
        sg.append(yx[1][i]*(yx[0]-prediction(yx[1],t)))
    return np.array(sg)

## run through the data one by one, making some step at each point
## d data, n predictors, m datapoints
## loop through entire set steps times
## scale step in gradient direction by alpha(learning rate)

def stochastic_gradient_ascent(d,n,m,steps, alpha):
    t = np.zeros(n+1)
    data_array = d.to_numpy()
    for i in range(steps):
        ## go through data in random sequence each step
        order = np.random.permutation(m)
        for j in range(0,m):  
            
            g = stochastic_gradient(t, get_yx(data_array[order[j]]),n)
            t = t + g * alpha
    return t

## see what kind of accuracy the model has on the training set of all data
model = stochastic_gradient_ascent(scaled_diabetes_data, 8,768, 100, .3)
s = 0
for i in range(0,768):
    array = scaled_diabetes_data.to_numpy()
    a = get_yx(array[i])
    b = prediction(a[1], model)
    if b >0.5:
        s =s + 1-a[0]
    else:
        s= s+ a[0]
print('fitting on all data')
print('%i errors on training set  ' % s)
rate = s/768
print('%f error rate' % rate)

## validation functions
def test_errors(test,model):
    array = test.to_numpy()
    errors = 0
    for i in range(0,len(test)):
        a = get_yx(array[i])
        b = prediction(a[1], model)
        if b >0.5:
            errors = errors + 1-a[0]
        else:
            errors= errors+ a[0]
    return errors/len(test)

# simple test/train split error
def data_split(data, fr):
    test_set = data.sample(frac = fr)
    train_set = data.drop(test_set.index)
    return (test_set, train_set)

def test_set_validation(d,n,frac,steps,alpha):
    split = data_split(d, frac)
    
    model = stochastic_gradient_ascent(split[1],n,len(split[1]),steps,alpha)
    return test_errors(split[0], model)
print('split 20% of data to use as test set')
print('error rate of :')
print(test_set_validation(scaled_diabetes_data, 8, .2, 150, .3))

# k-folds cross validation
def k_folds_split(data, k):
    splits =[]
    remain = data
    for i in range(0,k):
        current_split = remain.sample(frac = 1/(k-i))
        remain = remain.drop(current_split.index)
        splits.append(current_split)
    return splits
def k_fold_cross_validation(d, n, steps, alpha, k):
    folds = k_folds_split(d, k)
    error_rates = []
    for i in range(0,k):
        test_set = folds[i]
        train_set = d.drop(folds[i].index)
        model = stochastic_gradient_ascent(train_set,n,len(train_set),steps,alpha)
        error_rates.append(test_errors(test_set, model))
    return error_rates
print('just splitting once gives some variance to error due to how the data is split')
print('doing 5-fold cross validation gives better results')
k_folds_errors = k_fold_cross_validation(scaled_diabetes_data, 8, 150, .3, 5)
print('cross validation errors')
print(k_folds_errors)
print('Average error %f' % np.mean(k_folds_errors))

