# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:56:21 2023

@author: gianl
"""

from dataclasses import dataclass
from typing import List
from ATMANpy.types.Features import StreamFeatures, StreamFeaturesBase

@dataclass(frozen = True)
class SFList:
    sfList: List ### a list of stream features
    
    def __repr__(self):
        return f'SFList({self.length()} StreamFeatures from {self[0].first()[0]} to {self[-1].last()[0]})'
    
    def __iter__(self):
        return self.sfList.__iter__()
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.sfList[item]
        else:
            return SFList(self.sfList[item])
    
    def length(self):
        return len(self.sfList)
    
    def apply(self, func, assignArgs = None, assignVals = None, where = None):
        return SFList([mySF.apply(func, assignArgs = assignArgs, assignVals = assignVals, where = where) for mySF in self])
    
    def apply_whole(self, func):
        return SFList([func(mySF) for mySF in self])
    
    def merge(self, otherSFList):
        assert self.length() == otherSFList.length(), 'Lists to be merged have to be of same length!'
        return SFList([aSF.merge(bSF) for aSF, bSF in zip(self, otherSFList)])
    
    def concatenate(self, axis = 0):
        ### concatenates a list of features; axis similar to pandas
        # axis = 0 ---> along index (i.e. frames)
        # axis = 1 ---> along columns (i.e. features)
        if self.length() == 0:
            return StreamFeatures({})
        else:    
            concatFunc = StreamFeaturesBase.extend if axis == 0 else StreamFeaturesBase.merge
            x, xs = self[0], self[1:]
            return concatFunc(x, xs.concatenate())