# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import os

mainDir = r'C:\Users\gianl\Documents\GitHub\ATMAN'

def get_folder_struct(source, verbose = True):
    sourceDir = os.path.join(mainDir, source)
    
    if os.path.isdir(sourceDir):
        
        outDict = {'main': sourceDir}
        for f in os.listdir(sourceDir):
            outDict.update({f: os.path.join(sourceDir, f)})
            
        return outDict
        
    else:
        if verbose: print(f'No such directory: {sourceDir}')


####### video read in



##### 


from functools import reduce
def compose(*funcs):    
    def binary_compose(f, g):
        return lambda x : f(g(x))            
    return reduce(binary_compose, funcs, lambda x: x)

import pandas as pd
def prepend_index_level(df, idx_name, idx_value):
    return pd.concat({idx_value: df}, names = [idx_name])

def find_last_and_next_list_member(number_of_interest, 
                                   available_values):
    """
    This function takes a number and a list of values and returns 
    the greatest value smaller than the number in the list of values, and
    the smaller value greater than the number in the list of values.
    Very useful for interpolation
    
    Parameters
    ----------
    number_of_interest : integer
        The number to find the last occurence before and the first occurence after of.
    available_values : list of integers
        List of values that are available (for example, not missing in a dictionary)

    Returns
    -------
    ls_member, nx_member -> two integers, see above

    """

    if number_of_interest in available_values:
        return number_of_interest, number_of_interest
    else:
        try:
            ls_member = max([f for f in available_values if f < number_of_interest])
        except ValueError: ### occurs if max([]) --> no f smaller than frame.
            ls_member = None
        
        try:
            nx_member = min([f for f in available_values if f > number_of_interest])
        except:
            nx_member = None

        return ls_member, nx_member

import numpy as np
def shift_array(arr, num, fill_value=np.nan):
    arr = np.roll(arr,num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr

