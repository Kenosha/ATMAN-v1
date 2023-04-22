# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:59:24 2023

@author: gianl
"""

from dataclasses import dataclass

import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Lines import Line
from ATMANpy.types.Shapes3D import Line3D


@dataclass(frozen = True)
class StreamFeatures:
    featDict: dict
    
    @staticmethod
    def from_store(video_id, store_name, featnameList = None):
        featDict = ff.load_stored_features(video_id, store_name, featnameList = featnameList)
        return StreamFeatures(featDict)
    
    def __iter__(self):
        return iter(self.featDict)
    
    def __getitem__(self, item):
        if isinstance(item, int):
            ### get a single frame feature dictionary!
            try:
                return self.featDict[item]
            except KeyError:
                raise Exception(KeyError, f'No such frame: {item}')
            
        if isinstance(item, str):
            try:
                return {frame: frameDict[item] for frame, frameDict in self.featDict.items()}
            except KeyError:
                raise Exception(KeyError, f'No such feature: {item}')
            
        else:
            raise Exception('Pass in a frame (int) or an feature name (str)')

    def drop(self, item):
        if isinstance(item, int):
            ### drop a single frame feature dictionary!
            dropDict = {frame: features for frame, features in self.featDict.items() if frame != item}        
        if isinstance(item, str):
            dropDict = {frame: {name: feat for name, feat in features.items() if name != item} for frame, features in self.featDict.items()}
        return StreamFeatures(dropDict)
    
    def add(self, key, value):
        
        if isinstance(key, int) and isinstance(value, dict): ### adding a frame!
            if key in self.featDict:
                raise Exception(f'frame {key} exists, can not add entry.')
            else:
                new_featDict = self.featDict.copy()
                new_featDict[key] = value
                return StreamFeatures(new_featDict)
        
        elif isinstance(key, str): ### adding a feature
            new_featDict = {}    
            if isinstance(value, dict): ### dictionary like {frame: feature}, where feature might be different for every frame
                for frame, features in self.featDict.items():
                    if key in features:
                        raise Exception(f'feature {key} exists, can not add entry.')
                    else:
                        new_features = features.copy()
                        try:
                            new_features[key] = value[frame]
                        except KeyError:
                            new_features[key] = None
                        new_featDict[frame] = new_features
                        
            else: ### some constant thing, added everywhere 
                for frame, features in self.featDict.items():
                    if key in features:
                        raise Exception(f'feature {key} exists, can not add entry.')
                    else:
                        new_features = features.copy()
                        new_features[key] = value
                        new_featDict[frame] = new_features
            
            return StreamFeatures(new_featDict)
        
        else:
            raise Exception('Invalid key, value combination to add to stream features.')
        
    def rename_feature(self, old_name, new_name):
        rename_stringFunc = lambda inStr: new_name if inStr == old_name else inStr
        rename_dictFunc = lambda inDict: {rename_stringFunc(k): v for k, v in inDict.items()}
        
        renamedDict = {frame: rename_dictFunc(features) for frame, features in self.featDict.items()}
        return StreamFeatures(renamedDict)

    def split_joint_features(self, min_feature_length = 0):
        featList = ff.split_joint_features(self.featDict, min_feature_length = min_feature_length)
        return [StreamFeatures(featDict) for featDict in featList]
        
    def length(self):
        return len(self.featDict)
    
    def get_frames(self):
        return list(self.featDict.keys())
    
    def get_event_names(self):
        return list(self.featDict[0].keys())
    
    def spline_2D(self, pointFeature = 'ball', frameList = None):
        outDict = {}
        
        if frameList is None:
            frameList = self.get_frames()

        f1, fs = frameList[0], frameList[1:]
        for f2 in fs:
            ballPt1 = self[f1][pointFeature]
            ballPt2 = self[f2][pointFeature]
            spLine = Line(ballPt1, ballPt2, name = 'Spline')
            outDict[f2] = spLine
            f1 = f2
        return outDict
    
    def spline_3D(self, pointFeature = 'ball', frameList = None):
        outDict = {}
        
        if frameList is None:
            frameList = self.get_frames()

        f1, fs = frameList[0], frameList[1:]
        for f2 in fs:
            ballPt1 = self[f1][pointFeature]
            ballPt2 = self[f2][pointFeature]
            spLine = Line3D(ballPt1, ballPt2, name = 'Spline')
            outDict[f2] = spLine
            f1 = f2
        return outDict
    
        
#%% Testing


video_id = 'pqxAmUWIoKc'
store_name = '10to100_ok2'
