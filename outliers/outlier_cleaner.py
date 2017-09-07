#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    from sklearn import linear_model

    reg = linear_model.LinearRegression()

    error = np.ravel(np.abs(predictions - net_worths))
    n_remove = np.ceil(len(predictions)/10)
    index = np.argsort(error)[:-n_remove]
    print(index)
    for i in index:
        print(i)
        cleaned_data.append((ages[i, 0], net_worths[i, 0], error[i]))
    print(cleaned_data)
    return cleaned_data

