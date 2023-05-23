# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:59:24 2023

@author: gianl
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

import ATMANpy.funcs.feature_funcs as ff
from ATMANpy.funcs.util_funcs import get_positional_arguments, get_return_names

from ATMANpy.types.Lines import Line
from ATMANpy.types.Shapes3D import Line3D

def try_getattr(obj, methodStr):
    try:
        return getattr(obj, methodStr)
    except:
        def noneFunc(*args, **kwargs):
            return None
        return noneFunc

### TODO: Safety -> ensure stream features to be monotonously increasing! (don't have to be contiguous)

@dataclass(frozen = True)
class StreamFeaturesBase:
    ### The Stream Features Base Class
    # Everything defined here holds for: 
        # - StreamFeature (like pd.Series) -> frames in index, objects (single feature) in values!
        # - StreamFeatures (like pd.DataFrame) -> frames in index, featDict in values -> multiple features
    streamDict: dict 
    
    @staticmethod
    def concatenate(featureList, axis = 0):
        ### concatenates a list of features; axis similar to pandas
        # axis = 0 ---> along index (i.e. frames)
        # axis = 1 ---> along columns (i.e. features)
        if len(featureList) == 0:
            return StreamFeatures({})
        else:    
            concatFunc = StreamFeaturesBase.extend if axis == 0 else StreamFeaturesBase.merge
            x, xs = featureList[0], featureList[1:]
            return concatFunc(x, StreamFeaturesBase.concatenate(xs))
        
    def __iter__(self):
        return iter(self.streamDict)

    def keys(self):
        return self.streamDict.keys()
        
    def items(self):
        return self.streamDict.items()
    
    def values(self):
        return self.streamDict.values()
    
    def is_empty(self):
        return self.streamDict == {}

    def to_series(self):
        return pd.Series(self.streamDict, name = self.name)

    def first(self):
        frame = self.get_frames()[0]
        return frame, self.streamDict[frame]
    
    def last(self):
        frame = self.get_frames()[-1]
        return frame, self.streamDict[frame]
    
    def update(self, item, value):
        outDict = self.streamDict.copy()
        outDict[item] = value
        return self.__class__(outDict)
    
    def drop(self, item):
        ### drop a single frame feature dictionary!
        streamDict = {frame: featDict for frame, featDict in self.streamDict.items() if frame != item}    
        return self.__class__(streamDict, name = self.name)
    
    def keep(self, keepList):
        streamDict = {frame: featDict for frame, featDict in self.streamDict.items() if frame in keepList}    
        return self.__class__(streamDict, name = self.name)
    
    def conform(self, conformList):
        streamDict = {}
        for frame in conformList:
            if frame in self.keys():
                streamDict[frame] = self[frame]
            else:
                streamDict[frame] = {feat: None for feat in self.get_features()}
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
    
    def split(self, min_feature_length = 0):
        sfList = ff.split_joint_features(self.streamDict, min_feature_length = min_feature_length)
        return [self.__class__(streamDict, name = self.name) for streamDict in sfList]
        
    def length(self):
        return len(self.streamDict)
    
    def get_frames(self):
        return list(self.streamDict.keys())   
    
    def to_dict(self):
        return self.streamDict
    
    def to_list(self):
        return list(self.streamDict.values())
    
    def to_array(self):
        ### out shape: frames x features x d1 x d2 x ... x dN 
        ### even a StreamFeature instance will have a features dimension, at position 1 (frames always at position 0)
        return np.array([self[featName].to_list() for featName in self.get_features()]).swapaxes(0, 1)
    
@dataclass(frozen = True)
class StreamFeature(StreamFeaturesBase):
    #### (like pd.Series) -> frames in index, objects (single feature) in values!
    name: str = None ### The name of a StreamFeature instance is the name of the stored feature.
    
    @staticmethod
    def from_series(featSer):
        return StreamFeature(featSer.to_dict(), name = featSer.name)
    
    def __repr__(self):
        if self.is_empty():
            return 'StreamFeature()'
        else:
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'StreamFeature({self.name}, frames: {min_frame} to {max_frame})'
        
    def __getitem__(self, item):
        if isinstance(item, list):
            if all([isinstance(el, int) for el in item]):
                return StreamFeature({frame: feature for frame, feature in self.items() if frame in item})
            else:
                return KeyError(f'List passed to StreamFeature must be all integers, instead got {item}')
        
        elif isinstance(item, int): ### get a single frame feature dictionary!
             try:
                 return self.streamDict[item]
             except KeyError:
                 raise KeyError(f'No such frame: {item}')
        
        elif isinstance(item, str):
            if item == self.name:
                return self
            else:
                raise KeyError(f'No such feature: {item}')
        else:
            raise Exception('Pass in a frame (int) or a list of frames')
     
    def __eq__(self, otherSF):
        if self.name == otherSF.name:
            outName = self.name + '_equal'
        else:
            outName = self.name + '_equal_' + otherSF.name
        return BoolFeature({frame: self[frame] == otherSF[frame] for frame in self}, name = outName)
     
    def __ne__(self, otherSF):
        if self.name == otherSF.name:
            outName = self.name + '_unequal'
        else:
            outName = self.name + '_unequal_' + otherSF.name
        return BoolFeature({frame: self[frame] != otherSF[frame] for frame in self}, name = outName)
     
    def maybe_get(self, item):
        try:
            return self[item]
        except KeyError:
            return None
                  
    def get_features(self):
        return [self.name]
    
    def rename(self, new_name):
        return StreamFeature(self.streamDict, name = new_name)
    
    def recolor(self, new_bgr):
        reinstDict = {frame: try_getattr(obj, 'recolor')(new_bgr) for frame, obj in self.streamDict.items()}
        return StreamFeature(reinstDict, name = self.name)
    
    def resize(self, new_thickness):
        reinstDict = {frame: try_getattr(obj, 'recolor')(new_thickness) for frame, obj in self.streamDict.items()}
        return StreamFeature(reinstDict, name = self.name)
    
    def apply(self, func, assignArgs = None): ### TODO: at some point, allow *args, **kwargs
        ### This method allows for function application to single features, element by element
        ### func takes in features (H, courtLines, etc) and returns features
        
        if assignArgs is None:
            funcArgs = get_positional_arguments(func)
        else: ### give the option to directly assign function arguments, regardless of name
            funcArgs = assignArgs
            
        funcVals = get_return_names(func)
    
        assert funcArgs == self.get_features(), f'Expected function arguments ({funcArgs}) have to be same as available features ({self.get_features()})'
    
        ### figure out name
        if len(funcVals) == 1:
            name = funcVals[0]
        else:
            name = func.__name__
            
        applyDict = {}
        for frame in self:
            funcOut = func(self[frame])
            
            if len(funcVals) == 1:
                applyDict[frame] = {name: funcOut}
            else:
                applyDict[frame] = dict(zip(funcVals, funcOut))
    
        return StreamFeatures(applyDict, name = name).squeeze()
        
    def save(self, videoName, storeName):
        ### TODO: integrate further with feature_funcs
        from ATMANpy.funcs.feature_funcs import save_feature_frame
        featSer = self.to_series().rename('x_0')
        save_feature_frame(featSer, self.name, videoName, storeName)

    def unsqueeze(self):
        return StreamFeatures({frame: {self.name: feat} for frame, feat in self.items()})

@dataclass(frozen = True)
class StreamFeatures(StreamFeaturesBase):
    ####### (like pd.DataFrame) -> frames in index, featDict in values -> multiple features
    name: str = None ### usually, the name of a StreamFeatures instance is ignored.
    
    @staticmethod
    def from_store(video_id, store_name, featnameList = None):
        featDict = ff.load_stored_features(video_id, store_name, featnameList = featnameList)
        return StreamFeatures(featDict)
    
    def __repr__(self):
        if self.is_empty():
            return 'StreamFeatures()'
        else:
            features = self.get_features()
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'StreamFeatures({features}, frames: {min_frame} to {max_frame})'
        
    def __getitem__(self, item):
        if isinstance(item, list):
            if all([isinstance(el, int) for el in item]):
                return StreamFeature({frame: featDict for frame, featDict in self.items() if frame in item})
            elif all([isinstance(el, str) for el in item]):
                missing_in_feature = [el for el in item if el not in self.get_features()]
                if len(missing_in_feature) > 0:
                    raise KeyError(f'{missing_in_feature} not contained in features of {self}')
                else:
                    return StreamFeatures({frame: {feat: val for feat, val in featDict.items() if feat in item} for frame, featDict in self.streamDict.items()})
            else:
                raise KeyError(f'List passed to StreamFeature must be all integers or all strings, instead got {item}')
        elif isinstance(item, int): ### get a single frame feature dictionary!
             try:
                 return self.streamDict[item]
             except KeyError:
                 raise KeyError(f'No such frame: {item}')
        elif isinstance(item, str):
            try:
                streamDict = {frame: frameDict[item] for frame, frameDict in self.streamDict.items()}
                return StreamFeature(streamDict, name = item)
            except KeyError:
                raise KeyError(f'No such feature: {item}')
        else:
            raise Exception('Pass in a frame (int) or an feature name (str) or a list of these.')

    def __eq__(self, otherSF):
        shared_features = [f for f in self.get_features() if f in otherSF.get_features()]
        return BoolFeatures({frame: {feat + '_equal': self[frame][feat] == otherSF[frame][feat] for feat in shared_features} for frame in self.get_frames()})
        
    def __ne__(self, otherSF):
        shared_features = [f for f in self.get_features() if f in otherSF.get_features()]
        return BoolFeatures({frame: {feat + '_unequal': self[frame][feat] != otherSF[frame][feat] for feat in shared_features} for frame in self.get_frames()})
        
    def maybe_get(self, item):
        try:
            return self[item]
        except KeyError:
            return {feat: None for feat in self.get_features()}
                  
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

    def length(self):
        return len(self.streamDict)
    
    def get_frames(self):
        return list(self.streamDict.keys())
    
    def get_features(self):
        if self.is_empty():
            return []
        else:
            first_frame = self.get_frames()[0]
            return list(self.streamDict[first_frame].keys())
    
    def apply(self, func, assignArgs = None, assignVals = None, where = None): ###TODO: at some point, allow *args, **kwargs
        ### This method allows for function application to single features, element by element
        ### func takes in features (H, courtLines, etc) and returns features
        ### assignArgs --> List: give the option to directly assign function arguments, regardless of name
        ### assignVals --> List: give the option to rename the result features
        ### where --> BoolFeature: give the option to apply the function not everywhere
    
        if assignArgs is None:
            funcArgs = get_positional_arguments(func)
        else: ### give the option to directly assign function arguments, regardless of name
            funcArgs = assignArgs
        if assignVals is None:
            funcVals = get_return_names(func)
        else:
            funcVals = assignVals
    
        assert set(funcArgs).issubset(set(self.get_features())), f'Expected function arguments ({funcArgs}) have to be contained in available features ({self.get_features()})'
        
        ### figure out name
        if len(funcVals) == 1:
            name = funcVals[0]
        else:
            name = func.__name__
            
        applyDict = {}
        for frame in self:
            
            if where is None or where[frame]:
                funcOut = func(*[self[frame][arg] for arg in funcArgs])
                
                if len(funcVals) == 1:
                    applyDict[frame] = {name: funcOut}
                else:
                    applyDict[frame] = dict(zip(funcVals, funcOut))
                    
            else: ### if where is not None AND where[frame] is False...
                applyDict[frame] = {val: self[frame][val] for val in funcVals} 
                # ... we don't apply the function, instead look for the function values in the SF itself.
                    
    
        return StreamFeatures(applyDict, name = name).squeeze()
    
    def squeeze(self):
        if len(self.get_features()) == 1:
            return self[self.get_features()[0]]
        else:
            return self
    
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
    
@dataclass(frozen = True)
class BoolFeature(StreamFeature):
    def __repr__(self):
        if self.is_empty():
            return 'BoolFeature()'
        else:
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'BoolFeature({self.name}, frames: {min_frame} to {max_frame})'
    
    def neg(self):
        return BoolFeature({frame: not bool_feat for frame, bool_feat in self.items()})
        

@dataclass(frozen = True)
class BoolFeatures(StreamFeatures):
    
    def __repr__(self):
        if self.is_empty():
            return 'BoolFeatures()'
        else:
            features = self.get_features()
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'BoolFeatures({features}, frames: {min_frame} to {max_frame})'
        
    def any(self):
        boolArr = self.to_array().any(axis = 1)
        frames = self.get_frames()
        return BoolFeature(dict(zip(frames, boolArr)), name = 'any')
    
    def all(self):
        boolArr = self.to_array().all(axis = 1)
        frames = self.get_frames()
        return BoolFeature(dict(zip(frames, boolArr)), name = 'all')
    
    def neg(self):
        return BoolFeatures({frame: {feat_name: not bool_feat for feat_name, bool_feat in featDict.items()} for frame, featDict in self.items()})
        
        
#%% Testing


# video_id = 'pqxAmUWIoKc'
# store_name = '10to100_ok2'

# myDict = {100: {'foo': 2, 'bar': 29}, 101: {'foo': 3, 'bar': 30}}
# myDict2 = {102: {'foo': 2, 'bar': 29}, 103: {'foo': 3, 'bar': 30}}

# SF = StreamFeatures(myDict)
# SF2 = StreamFeatures(myDict2)

# SF[100]
# SF.drop(100)

# SF.add(10, {'fri': 3})

# SF.extend(SF2)


# SF[101]
# SF.extend(SF2)[102]


# myDict3 = {100: {'baz': 2, 'goo': 29}, 101: {'baz': 3, 'goo': 30}, 102: {'baz': 3, 'goo': 30}}
# myDict3 = {100: {'baz': 2, 'goo': 29}}
# SF3 = StreamFeatures(myDict3)

# SF.merge(SF3)[100]

# myDict = {100: 3, 101: 4, 102: 2}
# # myDict2 = {102: 3, 103: 3, 104: 3}
# myDict2 = {103: 3, 104: 3}
# SF = StreamFeature(myDict, name = 'foo')
# SF2 = StreamFeature(myDict2, name = 'foo')

# SF.extend(SF2)

# SF.first()








