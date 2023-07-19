# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import os
import time
import datetime
from typing import Union

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


def make_timestamp():
    ts = time.time()
    ts_format = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    
    return ts_format


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

def split_into_contiguous(inList):
    ### take a strictly monotonously increasing list of integers and 
    ### split it into lists with contiguous integers
    
    x1, xs = inList[0], inList[1:]
    
    outList = []
    currList = [x1]
    
    for x2 in xs:
        if x2 == x1 + 1:
            currList.append(x2)
        else:
            outList.append(currList)
            currList = [x2]
            
        x1 = x2
    
    outList.append(currList)
    
    return outList
    
def widen_list(inList, by = 1):
    by = int(by)
    assert by >= 0
    if by == 0:
        return inList
    
    myList = [min(inList) - 1, *inList, max(inList) + 1]
    if by == 1:
        return myList
    else:
        return widen_list(myList, by = by - 1)
    
    
def split_array_in_two(myArr, left_prop = 1, overall_prop = 2, axis = 0):
    ### left prop: parts of overall array that go to the left split, as proportion of overall
    ### i.e: if left_prop = 2 and overall_prop = 3, the array is split 2/3, 1/3 along axis
    assert left_prop < overall_prop
    
    parts = np.array_split(myArr, overall_prop, axis = axis)
    left_parts, right_parts = parts[:left_prop], parts[left_prop:]
    return np.concatenate(left_parts, axis = axis), np.concatenate(right_parts, axis = axis)

### signature of functions

import inspect
def get_positional_arguments(someFunc):
    spec = inspect.getfullargspec(someFunc)
    if spec.defaults is not None:
        num_kwargs = len(spec.defaults)
        return spec.args[:-num_kwargs]
    else:
        return spec.args

def get_return_names(someFunc):
    src, num_lines = inspect.getsourcelines(someFunc)
    src = [ln.strip(' ').rstrip('\n') for ln in src]
    ret = [ln for ln in src if ln.startswith('return ')]
    ret = [ln[7:] for ln in ret] ### remove 'return '
    ret = [ln.split(',') for ln in ret] ### split return values by comma
    ret = [[val.strip(' ') for val in ln] for ln in ret] ### drop whitespace
    
    nums_ret_vals = [len(ln) for ln in ret]
    assert len(set(nums_ret_vals)) == 1, f'Ambiguous number of return values in {someFunc}: {ret}'
    num_ret_vals = nums_ret_vals[0]
    
    ret_vals = [vals for vals in ret if vals != ['None'] * num_ret_vals]
    
    ### we now have to check for uniqueness of the non-'None' return values
    # since we have asserted all return values to have the same length
    # we can compare them elementwise
    ret_vals_elementwise = [[vals[i] for vals in ret_vals] for i in range(num_ret_vals)]
    ret_vals_uniques = [len(set(vals)) for vals in ret_vals_elementwise]
    assert ret_vals_uniques == [1] * num_ret_vals, f'non-uniquely names return values in {someFunc}: {ret_vals}'
    
    return ret_vals[0] ### due to uniqueness, take first.

#### Pandas Functions

def fill_with_hard_limit(
        df_or_series: Union[pd.DataFrame, pd.Series], limit: int,
        fill_method='interpolate',
        **fill_method_kwargs) -> Union[pd.DataFrame, pd.Series]:
    """The fill methods from Pandas such as ``interpolate`` or ``bfill``
    will fill ``limit`` number of NaNs, even if the total number of
    consecutive NaNs is larger than ``limit``. This function instead
    does not fill any data when the number of consecutive NaNs
    is > ``limit``.

    Adapted from: https://stackoverflow.com/a/30538371/11052174

    :param df_or_series: DataFrame or Series to perform interpolation
        on.
    :param limit: Maximum number of consecutive NaNs to allow. Any
        occurrences of more consecutive NaNs than ``limit`` will have no
        filling performed.
    :param fill_method: Filling method to use, e.g. 'interpolate',
        'bfill', etc.
    :param fill_method_kwargs: Keyword arguments to pass to the
        fill_method, in addition to the given limit.

    :returns: A filled version of the given df_or_series according
        to the given inputs.
    """

    # Keep things simple, ensure we have a DataFrame.
    try:
        df = df_or_series.to_frame()
    except AttributeError:
        df = df_or_series

    # Initialize our mask.
    mask = pd.DataFrame(True, index=df.index, columns=df.columns)

    # Get cumulative sums of consecutive NaNs.
    grp = (df.notnull() != df.shift().notnull()).cumsum()

    # Add columns of ones.
    grp['ones'] = 1

    # Loop through columns and update the mask.
    for col in df.columns:

        mask.loc[:, col] = (
                (grp.groupby(col)['ones'].transform('count') <= limit)
                | df[col].notnull()
        )

    # Now, interpolate and use the mask to create NaNs for the larger
    # gaps.
    method = getattr(df, fill_method)
    out = method(limit=limit, **fill_method_kwargs)[mask]

    # Be nice to the caller and return a Series if that's what they
    # provided.
    if isinstance(df_or_series, pd.Series):
        # Return a Series.
        return out.loc[:, out.columns[0]]

    return out

def split(df):
    """
    Split a dataframe into a list of dataframes with consecutive indices
    """
    nonconsecIdx = df.index
    consecidx = np.arange(len(df))
    return [d for _, d in df.groupby(nonconsecIdx - consecidx)]
    
    
