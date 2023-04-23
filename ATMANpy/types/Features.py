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
    streamDict: dict
    
    @staticmethod
    def from_store(video_id, store_name, featnameList = None):
        featDict = ff.load_stored_features(video_id, store_name, featnameList = featnameList)
        return StreamFeatures(featDict)
    
    def __repr__(self):
        features = self.get_features()
        frames = self.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        return f'StreamFeatures(frames: {min_frame} to {max_frame}, features: {features})'
    
    def __iter__(self):
        return iter(self.streamDict)
    
    def __getitem__(self, item):
        if isinstance(item, int):
            ### get a single frame feature dictionary!
            try:
                return self.streamDict[item]
            except KeyError:
                raise Exception(KeyError, f'No such frame: {item}')
            
        if isinstance(item, str):
            try:
                return {frame: frameDict[item] for frame, frameDict in self.streamDict.items()}
            except KeyError:
                raise Exception(KeyError, f'No such feature: {item}')
            
        else:
            raise Exception('Pass in a frame (int) or an feature name (str)')

    def drop(self, item):
        if isinstance(item, int):
            ### drop a single frame feature dictionary!
            dropDict = {frame: featDict for frame, featDict in self.streamDict.items() if frame != item}        
        if isinstance(item, str):
            dropDict = {frame: {name: feat for name, feat in featDict.items() if name != item} for frame, featDict in self.streamDict.items()}
        return StreamFeatures(dropDict)
    
    def keep(self, keepList):
        if all([isinstance(item, int) for item in keepList]):
            ### keeping frames!
            streamDict = {frame: featDict for frame, featDict in self.streamDict.items() if frame in keepList}    
        if all([isinstance(item, str) for item in keepList]):
            ### keeping features!
            streamDict = {frame: {feat: val for feat, val in featDict.items() if feat in keepList} for frame, featDict in self.streamDict.items()}
        return StreamFeatures(streamDict)            
        
    def add(self, key, value):
        
        if isinstance(key, int) and isinstance(value, dict): ### adding a frame!
            if key in self.streamDict:
                raise Exception(f'frame {key} exists, can not add entry.')
            else:
                new_streamDict = self.streamDict.copy()
                new_streamDict[key] = value
                return StreamFeatures(new_streamDict)
        
        elif isinstance(key, str): ### adding a feature
            new_streamDict = {}    
            if isinstance(value, dict): ### dictionary like {frame: feature}, where feature might be different for every frame
                for frame, featDict in self.streamDict.items():
                    if key in featDict:
                        raise Exception(f'feature {key} exists, can not add entry.')
                    else:
                        new_featDict = featDict.copy()
                        try:
                            new_featDict[key] = value[frame]
                        except KeyError:
                            new_featDict[key] = None
                        new_streamDict[frame] = new_featDict
                        
            else: ### some constant thing, added everywhere 
                for frame, featDict in self.streamDict.items():
                    if key in featDict:
                        raise Exception(f'feature {key} exists, can not add entry.')
                    else:
                        new_featDict = featDict.copy()
                        new_featDict[key] = value
                        new_streamDict[frame] = new_featDict
            
            return StreamFeatures(new_streamDict)
        
        else:
            raise Exception('Invalid key, value combination to add to stream features.')
        
    def rename_feature(self, old_name, new_name):
        rename_stringFunc = lambda inStr: new_name if inStr == old_name else inStr
        rename_dictFunc = lambda inDict: {rename_stringFunc(k): v for k, v in inDict.items()}
        
        renamedDict = {frame: rename_dictFunc(featDict) for frame, featDict in self.streamDict.items()}
        return StreamFeatures(renamedDict)

    def split_joint_features(self, min_feature_length = 0):
        featList = ff.split_joint_features(self.streamDict, min_feature_length = min_feature_length)
        return [StreamFeatures(featDict) for featDict in featList]
        
    def length(self):
        return len(self.streamDict)
    
    def get_frames(self):
        return list(self.streamDict.keys())
    
    def get_features(self):
        first_frame = self.get_frames()[0]
        return list(self.streamDict[first_frame].keys())
    
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
