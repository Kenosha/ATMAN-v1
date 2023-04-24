# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:59:24 2023

@author: gianl
"""

from dataclasses import dataclass

import pandas as pd

import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Lines import Line
from ATMANpy.types.Shapes3D import Line3D

### TODO: Safety -> ensure stream features to be monotonously increasing! (don't have to be contiguous)

@dataclass(frozen = True)
class StreamFeaturesBase:
    ### The Stream Features Base Class
    # Everything defined here holds for: 
        # - StreamFeature (like pd.Series) -> frames in index, objects (single feature) in values!
        # - StreamFeatures (like pd.DataFrame) -> frames in index, featDict in values -> multiple features
    streamDict: dict 
    
    def __iter__(self):
        return iter(self.streamDict)

    def __getitem__(self, item):
        ### get a single frame feature dictionary!
        try:
            return self.streamDict[item]
        except KeyError:
            raise Exception(KeyError, f'No such frame: {item}')
    
    def keys(self):
        return self.streamDict.keys()
        
    def items(self):
        return self.streamDict.items()
    
    def values(self):
        return self.streamDict.values()
    
    def to_series(self):
        return pd.Series(self.streamDict, name = self.name)

    def first(self):
        frame = self.get_frames()[0]
        return frame, self.streamDict[frame]
    
    def last(self):
        frame = self.get_frames()[-1]
        return frame, self.streamDict[frame]
    
    def drop(self, item):
        ### drop a single frame feature dictionary!
        streamDict = {frame: featDict for frame, featDict in self.streamDict.items() if frame != item}    
        return self.__class__(streamDict, name = self.name)
    
    def keep(self, keepList):
        streamDict = {frame: featDict for frame, featDict in self.streamDict.items() if frame in keepList}    
        return self.__class__(streamDict, name = self.name)
        
    def add(self, key, value):
        if key in self.streamDict:
            raise Exception(f'frame {key} exists, can not add entry.')
        else:
            streamDict = self.streamDict.copy()
            streamDict[key] = value
            return self.__class__(streamDict, name = self.name)
    
    def extend(self, otherSF):
        ### In the subclasses, check for feature name or feature names (for SF and SFs resp.)
        frames, frames_other = self.get_frames(), otherSF.get_frames()
        frames_overlap = [f for f in frames if f in frames_other]
        if frames_overlap != []:
            raise Exception(f'frames {frames_overlap} exist in both {self.__class__}, can not combine')
        else:
            streamDict = {**self.streamDict, **otherSF.streamDict} ### does not sort!
        return self.__class__(streamDict, name = self.name)
    
    def merge(self, otherSF):
        features, features_other = self.get_features(), otherSF.get_features()
        
        features_overlap = [f for f in features if f in features_other]
        if features_overlap != []:
            raise Exception(f'features {features_overlap} exist in {self} and {otherSF}, can not combine')
            
        streamDict = {}    
        for frame, featObj in self.streamDict.items():
            try:
                other_featObj = otherSF[frame]
            except:
                other_featObj = {f: None for f in features_other}
            finally:
                if isinstance(other_featObj, dict):
                    streamDict[frame] = {self.name: featObj, **other_featObj}
                else:
                    streamDict[frame] = {self.name: featObj, otherSF.name: other_featObj}
        return StreamFeatures(streamDict)
    
    def split_joint_features(self, min_feature_length = 0):
        sfList = ff.split_joint_features(self.streamDict, min_feature_length = min_feature_length)
        return [self.__class__(streamDict, name = self.name) for streamDict in sfList]
        
    def length(self):
        return len(self.streamDict)
    
    def get_frames(self):
        return list(self.streamDict.keys())   
    
@dataclass(frozen = True)
class StreamFeature(StreamFeaturesBase):
    #### (like pd.Series) -> frames in index, objects (single feature) in values!
    name: str = None ### The name of a StreamFeature instance is the name of the stored feature.
    
    def __repr__(self):
        frames = self.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        return f'StreamFeature({self.name}, frames: {min_frame} to {max_frame})'
    
    def get_features(self):
        return [self.name]
    
    def recolor(self, new_bgr):
        recoloredDict = {frame: obj.recolor(new_bgr) for frame, obj in self.streamDict.items()}
        return StreamFeature(recoloredDict, name = self.name)
    
        
@dataclass(frozen = True)
class StreamFeatures(StreamFeaturesBase):
    ####### (like pd.DataFrame) -> frames in index, featDict in values -> multiple features
    name: str = None ### usually, the name of a StreamFeatures instance is ignored.
    
    @staticmethod
    def from_store(video_id, store_name, featnameList = None):
        featDict = ff.load_stored_features(video_id, store_name, featnameList = featnameList)
        return StreamFeatures(featDict)
    
    def __repr__(self):
        features = self.get_features()
        frames = self.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        return f'StreamFeatures({features}, frames: {min_frame} to {max_frame})'
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return super().__getitem__(item)
            
        elif isinstance(item, str):
            try:
                streamDict = {frame: frameDict[item] for frame, frameDict in self.streamDict.items()}
                return StreamFeature(streamDict, name = item)
            except KeyError:
                raise Exception(KeyError, f'No such feature: {item}')
            
        else:
            raise Exception('Pass in a frame (int) or an feature name (str)')

    def drop(self, item):
        if isinstance(item, int):
            ### drop a single frame feature dictionary!
            return super().drop(item)      
        if isinstance(item, str):
            dropDict = {frame: {name: feat for name, feat in featDict.items() if name != item} for frame, featDict in self.streamDict.items()}
        return StreamFeatures(dropDict)
    
    def keep(self, keepList):
        if all([isinstance(item, int) for item in keepList]):
            ### keeping frames!
            return super().keep(keepList)
        if all([isinstance(item, str) for item in keepList]):
            ### keeping features!
            streamDict = {frame: {feat: val for feat, val in featDict.items() if feat in keepList} for frame, featDict in self.streamDict.items()}
        return StreamFeatures(streamDict)            
        
    def add(self, key, value):
        if isinstance(key, int) and isinstance(value, dict): ### adding a frame!
            return super().add(key, value)
        
        elif isinstance(key, str): ### adding a constant feature
            streamDict = self.streamDict.copy()
            for frame, featDict in self.streamDict.items():
                if key in featDict:
                    raise Exception(f'feature {key} exists, can not add entry.')
                else:
                    new_featDict = featDict.copy()
                    new_featDict[key] = value
                    streamDict[frame] = new_featDict
        
            return StreamFeatures(streamDict)
        
        else:
            raise Exception('Invalid key, value combination to add to stream features.')
            
    def merge(self, otherSF):
        features, features_other = self.get_features(), otherSF.get_features()
        
        features_overlap = [f for f in features if f in features_other]
        if features_overlap != []:
            raise Exception(f'features {features_overlap} exist in {self} and {otherSF}, can not combine')
            
        streamDict = {}    
        for frame, featDict in self.streamDict.items():
            try:
                other_featObj = otherSF[frame]
            except:
                other_featObj = {f: None for f in features_other}
            finally:
                if isinstance(other_featObj, dict):
                    streamDict[frame] = {**featDict, **other_featObj}
                else:
                    streamDict[frame] = {**featDict, otherSF.name: other_featObj}
        return StreamFeatures(streamDict)
    
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
    
    #### Should these be here?
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

myDict = {100: {'foo': 2, 'bar': 29}, 101: {'foo': 3, 'bar': 30}}
myDict2 = {102: {'foo': 2, 'bar': 29}, 103: {'foo': 3, 'bar': 30}}

SF = StreamFeatures(myDict)
SF2 = StreamFeatures(myDict2)

SF[100]
SF.drop(100)

SF.add(10, {'fri': 3})

SF.extend(SF2)


SF[101]
SF.extend(SF2)[102]


myDict3 = {100: {'baz': 2, 'goo': 29}, 101: {'baz': 3, 'goo': 30}, 102: {'baz': 3, 'goo': 30}}
myDict3 = {100: {'baz': 2, 'goo': 29}}
SF3 = StreamFeatures(myDict3)

SF.merge(SF3)[100]

myDict = {100: 3, 101: 4, 102: 2}
# myDict2 = {102: 3, 103: 3, 104: 3}
myDict2 = {103: 3, 104: 3}
SF = StreamFeature(myDict, name = 'foo')
SF2 = StreamFeature(myDict2, name = 'foo')

SF.extend(SF2)

SF.first()








