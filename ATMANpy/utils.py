# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import os

mainDir = r'C:\Users\gianl\Desktop\ATMAN'

def get_folder_struct(source, verbose = True):
    sourceDir = os.path.join(mainDir, source)
    
    if os.path.isdir(sourceDir):
        
        outDict = {'main': sourceDir}
        for f in os.listdir(sourceDir):
            outDict.update({f: os.path.join(sourceDir, f)})
            
        return outDict
        
    else:
        if verbose: print(f'No such directory: {sourceDir}')
