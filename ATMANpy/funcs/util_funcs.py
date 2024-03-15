# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import cv2 as cv
import os
import time
import datetime
from typing import Union

mainDir = r'C:\Users\gianl\Documents\GitHub\ATMAN'

def get_folder_struct(source, verbose = True):
    if source is None:
        sourceDir = mainDir
    else:
        sourceDir = os.path.join(mainDir, source)
    
    if os.path.isdir(sourceDir):
        outDict = {'main': sourceDir}
        for f in os.listdir(sourceDir):
            outDict.update({f: os.path.join(sourceDir, f)})
            
        return outDict
    else:
        if verbose: print(f'No such directory: {sourceDir}')

def maybe_make_folder(dir_that_exists, folder_to_maybe_create):
    fullPath = os.path.join(dir_that_exists, folder_to_maybe_create)
    try:
        os.mkdir(fullPath)
    except FileExistsError:
        pass
    return fullPath

def make_timestamp():
    ts = time.time()
    ts_format = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    
    return ts_format


####### video read in


def text_to_dict(textPath):
    with open(textPath) as f:
        lines = f.readlines()

    def parse(textStr):
        name, val = textStr.replace(' ', '').replace('\n', '').split('=')
        
        try:
            if '.' in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass
        
        return name, val
    
    outDict = {name.lower(): val for name, val in [parse(l) for l in lines]}
    return outDict

def read_config(match_name, config_name = '_config'):
    matchDir = get_folder_struct('scripts')['matches']
    configPath = os.path.join(matchDir, match_name, f'{config_name}.txt')
    
    configDict = text_to_dict(configPath)

    ### add the video capture to the config
    cap = get_video(configDict['video_id'])   
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    configDict['cap'] = cap
    configDict['natural_scale'] = width / 1920
    configDict['scale_factor'] = configDict['natural_scale'] * configDict['add_scale']
    
    print('Reading config...')
    for key, val in configDict.items():
        print(f'    {key}: {val}')
    
    return configDict

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
def shift_array(arr, num, axis = 0, fill_value = np.nan):
    if fill_value is np.nan:
        arr = arr.astype(float)
    
    swapArr = np.swapaxes(arr, 0, axis) ### move axis to front
    rollArr = np.roll(swapArr, num, axis = 0)
    if num < 0:
        rollArr[num:] = fill_value
    elif num > 0:
        rollArr[:num] = fill_value
    outArr = np.swapaxes(rollArr, 0, axis)
    return outArr

def argmin_multidim(arr):
    return np.unravel_index(arr.argmin(), arr.shape)
    ### below is how it's done in sf.viterbi_trajectories for a dist probarr of shape: plane, traj, plane, traj
    # argmins_flat = dist_probArr.reshape((plane * traj, plane, traj)).argmin(axis = 0)
    # np.unravel_index(argmins_flat, (plane, traj))
    
def argmax_multidim(arr):
    return np.unravel_index(arr.argmax(), arr.shape)

def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    import scipy.ndimage.filters as filters
    import scipy.ndimage.morphology as morphology
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)       

def cartesian_product(*xi):
    ### TODO: does not work for more than two lists
    ### two lists of x and y coordinates
    return np.dstack(np.meshgrid(*xi)).reshape(-1, len(xi))

def add_in_nan_rows(arr, is_nan_idxs, axis = 0):
    """
    add nan entries back into an array with removed nan entries along one dimension (the axis argument).
    along the axis dimension, the length of the array == is_nan_idxs.sum().
    wherever is_nan_idxs is False, we add in a np.nan array of matching shape.
    """
    
    swapArr = np.swapaxes(arr, 0, axis) ### bring axis to front
    
    arrList = []
    non_nan_idx = 0
    for is_nan in is_nan_idxs:
        if is_nan:
            fillArr = np.full_like(swapArr[0], fill_value = np.nan)
        else:
            fillArr = swapArr[non_nan_idx]
            non_nan_idx += 1
        arrList.append(fillArr)
    
    outArr = np.stack(arrList, axis = 0)
    outArr = np.swapaxes(outArr, axis, 0)
    return outArr
    

def split_into_contiguous(inList, offset = 1):
    ### take a strictly monotonously increasing list of integers and 
    ### split it into lists with contiguous integers
    
    if len(inList) == 0:
        return []
    
    x1, xs = inList[0], inList[1:]
    
    outList = []
    currList = [x1]
    
    for x2 in xs:
        if x2 == x1 + offset:
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
    

    
