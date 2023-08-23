# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:59:24 2023

@author: gianl
"""

from dataclasses import dataclass
import os
import time
import cv2 as cv
import numpy as np
import pandas as pd
from typing import List
from functools import partial

from skvideo.io import FFmpegWriter

import ATMANpy.funcs.feature_funcs as ff
from ATMANpy.funcs.util_funcs import get_positional_arguments, get_return_names
from ATMANpy.funcs.util_funcs import prepend_index_level, get_folder_struct, make_timestamp

from ATMANpy.types.Lines import Line
from ATMANpy.types.Shapes3D import Line3D
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, HSVFrame, BoolFrame, MaskFrame
from ATMANpy.types.MultiFrames import GreyFrames, BGRFrames, HSVFrames, BoolFrames, MaskFrames

frameTypeMap = {'bgr': BGRFrame,
                'grey': GreyFrame,
                'hsv': HSVFrame,
                'bool': BoolFrame,
                'mask': MaskFrame}

framesTypeMap = {'bgr': BGRFrames,
                 'grey': GreyFrames,
                 'hsv': HSVFrames,
                 'bool': BoolFrames,
                 'mask': MaskFrames}

def eqFunc(someFeat, otherFeat):
    eq = someFeat == otherFeat
    if isinstance(eq, bool):
        return eq
    elif isinstance(eq, np.ndarray):
        return eq.all()
    else: ### add cases if you need them!
        raise Exception(f'Equality could not be ascertained between {someFeat} and {otherFeat}')

def try_getattr(obj, methodStr):
    try:
        return getattr(obj, methodStr)
    except:
        def noneFunc(*args, **kwargs):
            return None
        return noneFunc

def extract_mp_timer(extractMethod):

    def timed(*args, **kw):

        time_before = time.time()
        result = extractMethod(*args, **kw)
        time_after = time.time()

        timeStr = round(time_after - time_before, 2) 
        selfObj, exFunc, otherArgs = args[0], args[1], args[2:] 
        
        if len(otherArgs) == 1:
            num_workers = otherArgs[0]
            paramStr = f'{num_workers} workers'
        if len(otherArgs) == 2:
            num_workers, chunksize = otherArgs
            paramStr = f'{num_workers} workers and chunks of {chunksize}'
        print(f'{extractMethod.__name__} on {selfObj.__class__.__name__} with {exFunc.__name__} using {paramStr}: took {timeStr} seconds')

        return result

    return timed

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

    def keys(self):
        return self.streamDict.keys()
        
    def items(self):
        return self.streamDict.items()
    
    def values(self):
        return self.streamDict.values()
    
    def is_empty(self):
        return self.streamDict == {}

    def is_contiguous(self):
        frames = self.get_frames()
        return self.length() == frames[-1] - frames[0] + 1

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
        streamDict = {frame: featDict for frame, featDict in self.items() if frame != item}    
        return self.__class__(streamDict, name = self.name)
    
    def keep(self, keepList):
        streamDict = {frame: featDict for frame, featDict in self.items() if frame in keepList}    
        return self.__class__(streamDict, name = self.name)
    
    def filter(self, boolSF):
        streamDict = {frame: featDict for frame, featDict in self.items() if boolSF[frame]}
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
    
    def merge(self, otherSF, how = 'left'):
        ### ensure we work with StreamFeatures
        leftSF, rightSF = self.unsqueeze(), otherSF.unsqueeze()
        
        ### Figure out validity
        left_features, right_features = leftSF.get_features(), rightSF.get_features()
        
        features_overlap = [f for f in left_features if f in right_features]
        if features_overlap != []:
            raise Exception(f'features {features_overlap} exist in {self} and {otherSF}, can not combine')
            
        ### figure out how to merge
        if how == 'left':
            frameList = leftSF.get_frames()
        elif how == 'right':
            frameList = rightSF.get_frames()
        elif how == 'outer':
            left_set, right_set = set(leftSF.get_frames()), set(rightSF.get_frames())
            frameList = list(left_set | right_set)
        elif how == 'inner':
            left_set, right_set = set(leftSF.get_frames()), set(rightSF.get_frames())
            frameList = list(left_set & right_set)
            
        streamDict = {}    
        for frame in frameList:
            ### get left featDict
            try:
                left_featDict = leftSF[frame]
            except KeyError:
                left_featDict = {f: None for f in left_features} 
            
            ### get right featDict
            try:
                right_featDict = rightSF[frame]
            except:
                right_featDict = {f: None for f in right_features}
            
            ### combine
            streamDict[frame] = {**left_featDict, **right_featDict}
                
        return StreamFeatures(streamDict)
    
    def split(self, min_feature_length = 0):
        sfList = ff.split_joint_features(self.streamDict, min_feature_length = min_feature_length)
        return SFList([self.__class__(streamDict, name = self.name) for streamDict in sfList])
        
    def length(self):
        return len(self.streamDict)
    
    def get_frames(self):
        return list(self.streamDict.keys())   
    
    def random_choice(self, how_many = 1000):
        ### from a given stream feature, create a bool feature with how_many true entries, selected randomly
        if how_many > self.length():
            print(f'Can not select {how_many} keys randomly in a feature of size {self.length} without replacement.')
            return None
        else:
            frames = self.get_frames()
            chosen = np.random.choice(frames, size = how_many, replace = False)
            boolDict = {frame: frame in chosen for frame in frames}
            return BoolFeature(boolDict, name = f'random_{self.name}')
    
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
                return self.__class__({frame: feature for frame, feature in self.items() if frame in item})
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
     
    def __eq__(self, otherObj):
        if isinstance(otherObj, StreamFeature):
            if self.name == otherObj.name:
                outName = self.name + '_equal'
            else:
                outName = self.name + '_equal_' + otherObj.name
            return BoolFeature({frame: eqFunc(self[frame], otherObj[frame]) for frame in self}, name = outName)
        else:
            return BoolFeature({frame: eqFunc(self[frame], otherObj) for frame in self}, name = self.name + '_equal')
            
    def __ne__(self, otherObj):
        if isinstance(otherObj, StreamFeature):
            if self.name == otherObj.name:
                outName = self.name + '_unequal'
            else:
                outName = self.name + '_unequal_' + otherObj.name
            return BoolFeature({frame: not eqFunc(self[frame], otherObj[frame]) for frame in self}, name = outName)
        else:
            return BoolFeature({frame: not eqFunc(self[frame], otherObj) for frame in self}, name = self.name + '_unequal')
            
    def maybe_get(self, item):
        try:
            return self[item]
        except KeyError:
            return None
                  
    def get_features(self):
        return [self.name]
    
    def conform(self, conformList):
        if isinstance(conformList, StreamFeaturesBase) or isinstance(conformList, ConstFeaturesBase):
            conformList = conformList.get_frames() 
        
        streamDict = {}
        for frame in conformList:
            if frame in self.get_frames():
                streamDict[frame] = self[frame]
            else:
                streamDict[frame] = None
        return self.__class__(streamDict, name = self.name)  
      
    def fillna(self, fill_value):
        if isinstance(fill_value, StreamFeature):
            return self.__class__({frame: fill_value.maybe_get(frame) if feat is None else feat for frame, feat in self.items()}, name = self.name)
        else:
            return self.__class__({frame: fill_value if feat is None else feat for frame, feat in self.items()}, name = self.name)
      
    def dropna(self):
        return StreamFeature({frame: feat for frame, feat in self.items() if feat is not None}, name = self.name)
    
    def isna(self):
        return BoolFeature({frame: feat is None for frame, feat in self.items()}, name = self.name)
    
    def rename(self, new_name):
        return self.__class__(self.streamDict, name = new_name)
    
    def recolor(self, new_bgr):
        reinstDict = {frame: try_getattr(obj, 'recolor')(new_bgr) for frame, obj in self.streamDict.items()}
        return self.__class__(reinstDict, name = self.name)
    
    def resize(self, new_thickness):
        reinstDict = {frame: try_getattr(obj, 'resize')(new_thickness) for frame, obj in self.streamDict.items()}
        return self.__class__(reinstDict, name = self.name)
    
    def rescale(self, scale_factor):
        reinstDict = {frame: try_getattr(obj, 'rescale')(scale_factor) for frame, obj in self.streamDict.items()}
        return self.__class__(reinstDict, name = self.name)
      
    def apply(self, func, **kwargs): ### TODO: at some point, allow *args, **kwargs
        ### This method allows for function application to single features, element by element
        ### func takes in features (H, courtLines, etc) and returns features
        funcVals = get_return_names(func)
    
        ### figure out name
        if len(funcVals) == 1:
            name = funcVals[0]
        else:
            name = func.__name__
            
        applyDict = {}
        for frame in self:
            funcOut = func(self[frame], **kwargs)
            
            if len(funcVals) == 1:
                applyDict[frame] = {name: funcOut}
            else:
                applyDict[frame] = dict(zip(funcVals, funcOut))
                
        return StreamFeatures(applyDict, name = name).squeeze()
    
    def _save_contour(self, videoName, storeName, compression = 24):
        exFunc = ff.extract_contour_values
        outSF = self.apply(exFunc, compression = compression) 
        featFrm = pd.DataFrame(outSF.streamDict).T
        featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(compression)])
        featFrm.columns = featLabels
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)
    
    def _save_contours(self, videoName, storeName, compression = 24):
        exFunc = ff.extract_contours_values
        outSF = self.apply(exFunc, compression = compression) 
        
        if not outSF.dropna().is_empty(): # TODO: Put this in more places! An all empty SF will give nothing to write and actually cause an error in outSF.to_frame, since an empty SF is always a StreamFeatures (not StreamFeature), which does not have a to_frame method (yet?)
            featFrm = outSF.to_frame()
            if not featFrm.empty:
                featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(compression)])
                featFrm.columns = featLabels
                ff.save_feature_frame(featFrm, self.name, videoName, storeName)
        
    def _save_line(self, videoName, storeName):
        exFunc = ff.extract_line_values
        outSF = self.apply(exFunc)
        featFrm = pd.DataFrame(outSF.streamDict).T
        if not featFrm.empty:
            featLabels = ['x1', 'y1', 'x2', 'y2']
            featFrm.columns = featLabels
            ff.save_feature_frame(featFrm, self.name, videoName, storeName)
            
    def _save_lines(self, videoName, storeName):
        exFunc = ff.extract_lines_values
        outSF = self.apply(exFunc)
        featFrm = outSF.to_frame()
        if not featFrm.empty:
            featLabels = ['x1', 'y1', 'x2', 'y2']
            featFrm.columns = featLabels
            ff.save_feature_frame(featFrm, self.name, videoName, storeName)
        
    def _save_homography(self, videoName, storeName):
        exFunc = ff.extract_homography_values
        outSF = self.apply(exFunc)
        featFrm = pd.DataFrame(outSF.streamDict).T
        featLabels = [f'{dim}_{idx}' for dim in ['x', 'y', 'z'] for idx in range(3)][:8]
        featFrm.columns = featLabels
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)
    
    def _save_projection(self, videoName, storeName):
        exFunc = ff.extract_projection_values
        outSF = self.apply(exFunc)
        featFrm = pd.DataFrame(outSF.streamDict).T
        featLabels = [f'{dim}_{idx}' for dim in ['x', 'y', 'z'] for idx in range(4)]
        featFrm.columns = featLabels
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)
    
    def _save_region(self, videoName, storeName):
        exFunc = ff.extract_region_values
        outSF = self.apply(exFunc)
        featFrm = pd.DataFrame(outSF.streamDict).T
        featLabels = np.ravel([[f'x_{idx}', f'y_{idx}'] for idx in range(4)]) ### region has four corners.
        featFrm.columns = featLabels
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)
    
    def _save_point(self, videoName, storeName):
        exFunc = ff.extract_point_values
        outSF = self.apply(exFunc)
        featFrm = pd.DataFrame(outSF.streamDict).T
        if len(featFrm.columns) == 2:
            featLabels = ['x', 'y']
        else:
            featLabels = ['x', 'y', 'z']
        featFrm.columns = featLabels
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)
    
    def _save_text(self, videoName, storeName):
        featFrm = self.to_series()
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)
    
    def _save_table(self, videoName, storeName):
        featFrm = self.to_frame()
        if not featFrm.empty:
            ff.save_feature_frame(featFrm, self.name, videoName, storeName)
        
    def _save_as_video(self, videoName, storeName,
                       monochrome = True, lossless = True):
        ### the folder name has information about which feature...
        savePath = ff.get_and_make_save_path(self.name, videoName, storeName)
        
        ### ... while the filename has info about the start and stop of the feature
        # NOTE THAT THIS MEANS WE CAN ONLY HAVE CONTIGUOUS VIDEO FEATURES!!!
        assert self.is_contiguous(), 'can only save contiguous video features as video!'
        frames = self.get_frames()
        fs, ls = frames[0], frames[-1]
        
        fileName = f'from_{fs}_to_{ls}.avi'
        
        fullPath = os.path.join(savePath, fileName)
        
        if lossless:
            outputdict = {
              '-vcodec': 'libx264',  #use the h.264 codec
              '-crf': '0',           #set the constant rate factor to 0, which is lossless
              '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                     #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
             }
        else:
            outputdict = {}
            
        writer = FFmpegWriter(fullPath, outputdict = outputdict)
        
        for myFrm in self.values():
            writer.writeFrame(myFrm.array)
        
        writer.close()
        
    def save(self, videoName, storeName, lossless = False, compression = None):
        ### New principle: -> do not save None values! 
        saveSF = self.dropna()
        
        contourNames = ['topPlayer', 'botPlayer']
        homographyNames = ['H']
        regionNames = ['scoreBoard']
        
        if self.name in contourNames:
            saveSF._save_contour(videoName, storeName, compression = compression)
        elif self.name == 'H':
            saveSF._save_homography(videoName, storeName)
        elif self.name == 'P':
            saveSF._save_projection(videoName, storeName)
        elif self.name in regionNames:
            saveSF._save_region(videoName, storeName)
        elif self.name.endswith('Ln'):
            saveSF._save_line(videoName, storeName)
        elif self.name.endswith('Lns'):
            saveSF._save_lines(videoName, storeName)
        elif self.name.endswith('Cnt'):
            saveSF._save_contour(videoName, storeName, compression = compression)
        elif self.name.endswith('Cnts'):
            saveSF._save_contours(videoName, storeName, compression = compression)
        elif self.name.endswith('3D'):
            saveSF._save_point(videoName, storeName)
        elif 'Mask' in self.name:
            saveSF._save_as_video(videoName, storeName, monochrome = True, lossless = lossless)
        elif 'Frm' in self.name:
            saveSF._save_as_video(videoName, storeName, monochrome = False, lossless = lossless)
        elif 'Flag' in self.name: ### all text has to be called Flag!
            saveSF._save_text(videoName, storeName)
        elif 'Table' in self.name:
            saveSF._save_table(videoName, storeName)
        else:
            print(f"Did not save {self.name}, don't know how!")
        
    def squeeze(self):
        return self
    
    def unsqueeze(self):
        return StreamFeatures({frame: {self.name: feat} for frame, feat in self.items()})
    
    def to_bool(self):
        return BoolFeature(self.streamDict, name = self.name)

    def to_series(self):
        return pd.Series(self.streamDict, name = self.name)
    
    def to_frame(self):
        ### ONLY IF THE FEATURES CONTAINED ARE THEMSELVES FRAMES!
        frameList = [prepend_index_level(myFrm, 'idx0', frmNum) for frmNum, myFrm in self.items() if myFrm is not None]
        if len(frameList) == 0:
            return pd.DataFrame()
        else:
            return pd.concat(frameList, axis = 0)

    def to_multiframe(self):
        def to_array(myFrm):
            outArr = myFrm.array
            return outArr
        frmArr = self.apply(to_array).to_array().squeeze()
        
        fsNum, fsFrm = self.first()
        framesType = framesTypeMap[fsFrm.get_color()]
        return framesType(frmArr, name = self.name)
        
    def value_counts(self):
        return self.to_series().value_counts()

@dataclass(frozen = True)
class StreamFeatures(StreamFeaturesBase):
    ####### (like pd.DataFrame) -> frames in index, featDict in values -> multiple features
    name: str = None ### usually, the name of a StreamFeatures instance is ignored.
    
    @classmethod
    def from_store(cls, video_id, store_name, featnameList = None):
        featDict = ff.load_stored_features(video_id, store_name, featnameList = featnameList)
        return cls(featDict)
    
    @staticmethod
    def type_squeeze(): ### weird!
        return StreamFeature
    
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
                return self.__class__.type_squeeze()({frame: featDict for frame, featDict in self.items() if frame in item})
            elif all([isinstance(el, str) for el in item]):
                missing_in_feature = [el for el in item if el not in self.get_features()]
                if len(missing_in_feature) > 0:
                    raise KeyError(f'{missing_in_feature} not contained in features of {self}')
                else:
                    return self.__class__({frame: {feat: val for feat, val in featDict.items() if feat in item} for frame, featDict in self.streamDict.items()})
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
                return self.__class__.type_squeeze()(streamDict, name = item)
            except KeyError:
                raise KeyError(f'No such feature: {item}')
        else:
            raise Exception('Pass in a frame (int) or an feature name (str) or a list of these.')

    def __eq__(self, otherObj):
        shared_features = [f for f in self.get_features() if f in otherObj.get_features()]
        if isinstance(otherObj, StreamFeatures):
            return BoolFeatures({frame: {feat + '_equal': eqFunc(self[frame][feat], otherObj[frame][feat]) for feat in shared_features} for frame in self.get_frames()})
        else:
            return BoolFeatures({frame: {feat + '_equal': eqFunc(self[frame][feat], otherObj) for feat in shared_features} for frame in self.get_frames()})
        
    def __ne__(self, otherObj):
        shared_features = [f for f in self.get_features() if f in otherObj.get_features()]
        if isinstance(otherObj, StreamFeatures):
            return BoolFeatures({frame: {feat + '_unequal': eqFunc(self[frame][feat], otherObj[frame][feat]) for feat in shared_features} for frame in self.get_frames()})
        else:
            return BoolFeatures({frame: {feat + '_unequal': not eqFunc(self[frame][feat], otherObj) for feat in shared_features} for frame in self.get_frames()})
        
    def maybe_get(self, item):
        try:
            return self[item]
        except KeyError:
            return {feat: None for feat in self.get_features()}
                
    def conform(self, conformList):
        if isinstance(conformList, StreamFeaturesBase):
            conformList = conformList.get_frames()
        
        streamDict = {}
        for frame in conformList:
            if frame in self.get_frames():
                streamDict[frame] = self[frame]
            else:
                streamDict[frame] = {feat: None for feat in self.get_features()}
        return StreamFeatures(streamDict, name = self.name)
      
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
    
    def dropna(self):
        none_feature = {feat: None for feat in self.get_features()}
        return self.__class__({frame: featDict for frame, featDict in self.items() if featDict != none_feature}, name = self.name)
    
    def replace(self, otherSF):
        outDict = self.streamDict.copy()
        other_features = otherSF.get_features()
        for frame, otherDict in otherSF.items():
            try:
                featDict = self[frame].copy()
            except:
                continue
            else:
                for featName, featVal in otherDict.items():
                    if featName in featDict:
                        featDict[featName] = featVal
                outDict[frame] = featDict
            
        return StreamFeatures(outDict, name = self.name)
    
    def rename_feature(self, old_name, new_name):
        rename_stringFunc = lambda inStr: new_name if inStr == old_name else inStr
        rename_dictFunc = lambda inDict: {rename_stringFunc(k): v for k, v in inDict.items()}
        
        renamedDict = {frame: rename_dictFunc(featDict) for frame, featDict in self.streamDict.items()}
        return StreamFeatures(renamedDict)

    def rescale(self, scale_factor):
        featList = [self[featName].rescale(scale_factor) for featName in self.get_features()]
        return SFList(featList).concatenate(axis = 1)

    def length(self):
        return len(self.streamDict)
    
    def get_features(self):
        if self.is_empty():
            return []
        else:
            first_frame = self.get_frames()[0]
            return list(self.streamDict[first_frame].keys())
    
    def apply(self, func, **kwargs): ###TODO: at some point, allow *args, **kwargs
        ### This method allows for function application to single features, element by element
        ### func takes in features (H, courtLines, etc) and returns features
        ### assignArgs --> List: give the option to directly assign function arguments, regardless of name
        ### assignVals --> List: give the option to rename the result features
        ### where --> BoolFeature: give the option to apply the function not everywhere
    
        funcArgs = get_positional_arguments(func)
        funcVals = get_return_names(func)
        
        assert set(funcArgs).issubset(set(self.get_features())), f'Expected function arguments ({funcArgs}) have to be contained in available features ({self.get_features()})'
        
        ### figure out name
        if len(funcVals) == 1:
            name = funcVals[0]
        else:
            name = func.__name__
            
        applyDict = {}
        for frame in self:
            
            funcOut = func(*[self[frame][arg] for arg in funcArgs])
            
            if len(funcVals) == 1:
                applyDict[frame] = {name: funcOut}
            else:
                applyDict[frame] = dict(zip(funcVals, funcOut))
    
        return StreamFeatures(applyDict, name = name).squeeze()
    
    def unsqueeze(self):
        return self
    
    def squeeze(self):
        if len(self.get_features()) == 1:
            return self[self.get_features()[0]]
        else:
            return self
    
    def save(self, videoName, storeName, compression = None, lossless = False):
        for feat in self.get_features():
            self[feat].save(videoName, storeName, 
                            compression = compression, lossless = lossless)
    
    def to_bool(self):
        return BoolFeatures(self.streamDict, name = self.name)
    
    def to_dataframe(self):
        return pd.DataFrame.from_dict(self.streamDict, orient = 'index')
    
    def value_counts(self):
        return pd.concat([self[feat].value_counts() for feat in self.get_features()], axis = 1)
    
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
class ConstFeaturesBase:
    
    def get_frames(self):
        return self.frameList
    
    def is_empty(self):
        return len(self.frameList) == 0
    
    def first(self):
        frame = self.get_frames()[0]
        return frame, self[frame]
    
    def last(self):
        frame = self.get_frames()[-1]
        return frame, self[frame]
   
    def filter(self, boolSF):
        frameList = [frame for frame in self.frameList if boolSF[frame]]
        return self.__class__(self.name, self.feature, frameList)
    
    def merge(self, otherSF):
        features, features_other = self.get_features(), otherSF.get_features()
        
        features_overlap = [f for f in features if f in features_other]
        if features_overlap != []:
            raise Exception(f'features {features_overlap} exist in {self} and {otherSF}, can not combine')
            
        if isinstance(otherSF, ConstFeaturesBase):
            selfDict = self.unsqueeze().featDict
            otherDict = otherSF.unsqueeze().featDict
            return ConstFeatures(selfDict | otherDict, self.frameList)
        else:
            return self.to_SF().merge(otherSF)
        
@dataclass(frozen = True)
class ConstFeature(ConstFeaturesBase):
    ### a feature that is always the same, associated with a set of frames.
    name: str
    feature: np.ndarray
    frameList: List
    
    def __repr__(self):
        if self.is_empty():
            return 'ConstFeature()'
        else:
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'ConstFeature({self.name}, frames: {min_frame} to {max_frame})' ### looks like it's contiguous, but it need not be!
    
    def __getitem__(self, item):
        if item in self.frameList:
            return self.feature
        else:
            raise KeyError(f'Frame {item} not part of {self}')
    
    def maybe_get(self, item):
        try:
            return self[item]
        except KeyError:
            return None
        
    def get_features(self):
        return [self.name]
    
    def to_SF(self):
        return StreamFeature({frame: self.feature for frame in self.frameList}, name = self.name)
              
    def apply(self, func, **kwargs): ### TODO: at some point, allow *args, **kwargs
        ### This method allows for function application to single features, element by element
        ### func takes in features (H, courtLines, etc) and returns features
        funcVals = get_return_names(func)
    
        ### figure out name
        if len(funcVals) == 1:
            name = funcVals[0]
        else:
            name = func.__name__
         
        funcOut = func(self.feature, **kwargs)
        
        if len(funcVals) == 1:
            return ConstFeature(name, funcOut, self.frameList)
        else:
            featDict = dict(zip(funcVals, funcOut))
            return ConstFeatures(featDict, self.frameList)
    
    def split(self, min_feature_length = 0):
        prev_frame, frames = self.frameList[0], self.frameList[1:]
        
        curr_frame_list = [prev_frame]
        list_of_frame_lists = []
        
        for frame in frames:
            if frame > prev_frame + 1: ### if there is a gap between last and current frame
                if len(curr_frame_list) > min_feature_length: ### if the current accumulated sf is large enough
                    list_of_frame_lists.append(curr_frame_list)    
                curr_frame_list = []
            
            curr_frame_list.append(frame)
            prev_frame = frame
        
        ### finally, add in the remaining features (if long enough)
        if len(curr_frame_list) > min_feature_length: 
            list_of_frame_lists.append(curr_frame_list)
        
        return SFList([ConstFeature(self.name, self.feature, frame_list) for frame_list in list_of_frame_lists])
    
    def squeeze(self):
        return self
    
    def unsqueeze(self):
        return ConstFeatures({self.name: self.feature}, self.frameList)
    
    def conform(self, frameList):
        return ConstFeature(self.name, self.feature, frameList)
    
    
    
@dataclass(frozen = True)
class ConstFeatures(ConstFeaturesBase):
    ### a dict of features that is always the same, associated with a set of frames. 
    featDict: dict
    frameList: List
    
    @staticmethod
    def from_store(video_id, store_name, featnameList, featTypeList = None, 
                   frameList = None):
        if frameList is None:
            frameList = []
        
        pathDict = get_folder_struct('features_stored')
        storePath = os.path.join(pathDict['main'], video_id, store_name)
        
        ### if no list is given, read all available features
        if featnameList is None: 
            featnameList = [f for f in os.listdir(storePath) if not f.endswith('.csv')]
        else:
            featnameList = [f for f in featnameList if f in os.listdir(storePath)]
        
        ### figure out types
        if featTypeList is None:
            featTypeList = [None] * len(featnameList)
        else:
            assert len(featnameList) == len(featTypeList)
        
        ### collect all feature frames
        featDict = {}
        for featName, featType in zip(featnameList, featTypeList):
            featPath = os.path.join(storePath, featName)
            last_file = os.listdir(featPath)[-1] ### this has to be improved! Along with feature IO in general
            fullPath = os.path.join(featPath, last_file)
            
            myArr = np.load(fullPath) 
            if featType is None:
                featDict[featName] = myArr
            else:
                featDict[featName] = featType.from_array(myArr)
    
        return ConstFeatures(featDict, frameList)
    
    def __repr__(self):
        if self.is_empty():
            return 'ConstFeatures()'
        else:
            features = self.get_features()
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'ConstFeatures({features}, frames: {min_frame} to {max_frame})'
        
    def __getitem__(self, item):
        if isinstance(item, str):
            return ConstFeature(item, self.featDict[item], self.frameList)
        else:
            if item in self.frameList:
                return self.featDict
            else:
                raise KeyError(f'Frame {item} not part of {self}')
        
    def maybe_get(self, item):
        try:
            return self[item]
        except KeyError:
            return {feat: None for feat in self.get_features()}
                
    def get_features(self):
        return list(self.featDict.keys())
    
    def to_SF(self):
        return StreamFeatures({frame: self.featDict for frame in self.frameList})
    
    def apply(self, func, **kwargs): ###TODO: at some point, allow *args, **kwargs
        ### This method allows for function application to single features, element by element
        ### func takes in features (H, courtLines, etc) and returns features
        ### assignArgs --> List: give the option to directly assign function arguments, regardless of name
        ### assignVals --> List: give the option to rename the result features
        ### where --> BoolFeature: give the option to apply the function not everywhere
        funcArgs = get_positional_arguments(func)
        funcVals = get_return_names(func)
        
        assert set(funcArgs).issubset(set(self.get_features())), f'Expected function arguments ({funcArgs}) have to be contained in available features ({self.get_features()})'
        
        ### figure out name
        if len(funcVals) == 1:
            name = funcVals[0]
        else:
            name = func.__name__
            
        funcOut = func(*[self.featDict[arg] for arg in funcArgs])
        
        if len(funcVals) == 1:
            return ConstFeature(name, funcOut, self.frameList)
        else:
            featDict = dict(zip(funcVals, funcOut))
            return ConstFeatures(featDict, self.frameList) 
    
    def split(self, min_feature_length = 0):
        prev_frame, frames = self.frameList[0], self.frameList[1:]
        
        curr_frame_list = [prev_frame]
        list_of_frame_lists = []
        for frame in frames:
            if frame > prev_frame + 1: ### if there is a gap between last and current frame
                if len(curr_frame_list) > min_feature_length: ### if the current accumulated sf is large enough
                    list_of_frame_lists.append(curr_frame_list)    
                curr_frame_list = []
            
            curr_frame_list.append(frame)
            prev_frame = frame
        
        ### finally, add in the remaining features (if long enough)
        if len(curr_frame_list) > min_feature_length: 
            list_of_frame_lists.append(curr_frame_list)
        
        return SFList([ConstFeatures(self.featDict, frame_list) for frame_list in list_of_frame_lists])
        
    def squeeze(self):
        if len(self.featDict) == 1:
            name = self.get_features()[0]
            feature = self.featDict[name]
            return ConstFeature(name, feature, self.frameList)
        else:
            return self
    
    def unsqueeze(self):
        return self
    
    def conform(self, frameList):
        return ConstFeatures(self.featDict, frameList)
    
    
@dataclass(frozen = True)
class BoolFeature(StreamFeature):
    def __repr__(self):
        if self.is_empty():
            return 'BoolFeature()'
        else:
            frames = self.get_frames()
            min_frame, max_frame = frames[0], frames[-1]
            return f'BoolFeature({self.name}, frames: {min_frame} to {max_frame})'
    
    def fillna(self, fill_value):
        assert isinstance(fill_value, bool), 'Filling a BoolFeature can only be done with True or False'
        return BoolFeature({frame: fill_value if feat is None else feat for frame, feat in self.items()}, name = self.name)
       
    def neg(self):
        return BoolFeature({frame: not bool_feat for frame, bool_feat in self.items()})
    
    def true(self):
        return sum(self.values())
    def false(self):
        return self.length() - self.true()
    
    def save(self, videoName, storeName, compression = None, lossless = None):
        featFrm = self.to_series()
        featFrm.index.name = 'idx0'
        ff.save_feature_frame(featFrm, self.name, videoName, storeName)

    def split(self, min_feature_length = 0):
        sfList = ff.split_joint_features(self.streamDict, min_feature_length = min_feature_length)
        return SFList([self.__class__(streamDict, name = self.name) for streamDict in sfList])
        

@dataclass(frozen = True)
class BoolFeatures(StreamFeatures):
    
    @staticmethod
    def type_squeeze():
        return BoolFeature
    
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
    
    def true(self):
        return {feat: self[feat].true() for feat in self.get_features()}
    def false(self):
        return {feat: self[feat].false() for feat in self.get_features()}
        
    def neg(self):
        return BoolFeatures({frame: {feat_name: not bool_feat for feat_name, bool_feat in featDict.items()} for frame, featDict in self.items()})
       
@dataclass(frozen = True)
class SFList:
    sfList: List ### a list of stream features
    
    def __repr__(self):
        return f'SFList({self.length()} StreamFeatures from {self[0].first()[0]} to {self[0].last()[0]})'
    
    def __iter__(self):
        return self.sfList.__iter__()
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.sfList[item]
        else:
            return SFList(self.sfList[item])
    
    def get_feature(self, feature):
        return SFList([mySF[feature] for mySF in self])
    
    def length(self):
        return len(self.sfList)
    
    def rename(self, new_name):
        featList = [mySF.rename(new_name) for mySF in self]
        return SFList(featList)
    
    def rescale(self, scale_factor):
        featList = [mySF.rescale(scale_factor) for mySF in self]
        return SFList(featList)
    
    def apply(self, func, **kwargs):
        return SFList([mySF.apply(func, **kwargs) for mySF in self])
    
    def apply_whole(self, func, *args, **kwargs):
        outList = [func(mySF, *args, **kwargs) for mySF in self] ### these dont necessarily have to be StreamFeatures!
        if isinstance(outList[0], StreamFeature):
            return SFList(outList)
        else:
            return outList
    
    @extract_mp_timer
    def apply_whole_mp(self, func, num_workers, **kwargs):
        from multiprocessing.pool import Pool
        partFunc = partial(func, **kwargs)
        
        with Pool(num_workers) as p:
            outList = p.map(partFunc, self.sfList)
    
        return SFList(outList)
    
    def add(self, key, value):
        return SFList([mySF.add(key, value) for mySF in self])
    
    def merge(self, otherSFList):
        assert self.length() == otherSFList.length(), 'Lists to be merged have to be of same length!'
        return SFList([aSF.merge(bSF) for aSF, bSF in zip(self, otherSFList)])
    
    def concatenate(self):
        ### extends a list of features (along index)
        if self.length() == 0:
            return StreamFeatures({})
        else:    
            x, xs = self[0], self[1:]
        return StreamFeaturesBase.extend(x, xs.concatenate())
    
    def join(self): 
        ### merges a list of features (along columns)
        if self.length() == 0:
            return StreamFeatures({})
        else:    
            x, xs = self[0], self[1:]
        return StreamFeaturesBase.merge(x, xs.join(), how = 'outer')
    
    def save(self, videoName, storeName, compression = 24, lossless = False):
        for mySF in self:
            mySF.save(videoName, storeName, compression = compression, lossless = lossless)
        
        
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








