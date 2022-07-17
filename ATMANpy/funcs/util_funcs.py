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
    



