# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:03:08 2023

@author: gianl
"""

from ATMANpy.types.Frames import BGRFrame, GreyFrame, HSVFrame
from ATMANpy.types.Features import StreamFeatures, StreamFeature
from ATMANpy.types.Caches import Cache

from ATMANpy.funcs.draw_funcs import draw

from dataclasses import dataclass
import cv2 as cv

from ATMANpy.funcs.util_funcs import get_positional_arguments, get_return_names

@dataclass(frozen = True)
class Stream:
    cap: cv.VideoCapture
    start_frame: int
    stop_frame: int
    features: StreamFeatures = StreamFeatures({})
        
    def __post_init__(self):
        mySF = self.features
        if isinstance(self.features, StreamFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        ### conform StreamFeatures to start and stop frame
        # mySF = mySF.conform(list(range(self.start_frame, self.stop_frame + 1))) ### capture contains last frame!
        object.__setattr__(self, 'features', mySF)
    
    def __getitem__(self, frame_num):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        return BGRFrame(frame)
         
    def apply(self, mySF: StreamFeatures):
        return self.__class__(self.cap, self.start_frame, self.stop_frame, features = mySF)

    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame)
        frmNum = self.start_frame
        
        while self.cap.isOpened() and (self.start_frame <= frmNum < self.stop_frame):
            frmNum = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
            ret, frame = self.cap.read()
            
            frame = yield frmNum, frame # take in frame, possibly altered, to show 
            # frame = yield frame 
            ## TODO: include featDict as a thing the generator yields!
            
            if frame_counter:
                cv.putText(frame, str(frmNum), org = (40, 40), 
                           fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                           color = (0, 255, 0), thickness = 3)
            
            
            if frame is not None:
                cv.imshow('frame', frame)
                if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                    cv.destroyAllWindows()  
                    # yield None
                    break
        
        else:
            cv.destroyAllWindows()

    def show(self, ms_between_frames = 25, frame_counter = False):
        gen = self.generator(ms_between_frames = ms_between_frames, frame_counter = frame_counter)
        frmNum, rawArr = next(gen) ### prime the generator
        drawFunc = draw(self.features.get_features())
        
        while True:
            cFrm = BGRFrame(rawArr)
            
            try:
                featDict = self.features[frmNum]
            except KeyError:
                sendArr = rawArr
            else:
                sendFrm = drawFunc({'cFrm': cFrm}, featDict)
                sendArr = sendFrm.array  
                 
            try:
                frmNum, rawArr = gen.send(sendArr)
            except StopIteration:
                cv.destroyAllWindows()
                break 
               
    def extract(self, featureFunc, **kwargs): ### cv_funcs.get_stream_features
        
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)

        gen = self.generator()
        frmNum, rawArr = next(gen) ### prime the generator
        
        streamDict = {}
        while True:
            
            ### get arguments to plug into feature extraction function
            args = []
            for argStr in funcArgs:
                if argStr == 'cFrm':
                    argObj = BGRFrame(rawArr)
                elif argStr == 'gFrm':
                    argObj = BGRFrame(rawArr).to_grey()
                elif argStr == 'hFrm':
                    argObj = BGRFrame(rawArr).to_HSV()
                else:
                    try:
                        argObj = self.features[frmNum][argStr]
                    except KeyError as e:
                        raise e
                args.append(argObj)
                
            ### assert that all arguments that we expect (funcArgs) we have been able to extract (args).
            assert len(args) == len(funcArgs)
            
            ### assign to a dictionary, depending on whether we get one or multiple return values
            if len(funcVals) == 1:
                myFeature = featureFunc(*args, **kwargs)
                streamDict[frmNum] = myFeature
            else:
                myFeatures = featureFunc(*args, **kwargs)
                streamDict[frmNum] = dict(zip(funcVals, myFeatures))
            
            ### iterate the generator.
            try:
                frmNum, rawArr = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
            except StopIteration:
                break 
        
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        if len(funcVals) == 1:
            return StreamFeature(streamDict, name = funcVals[0])
        else:
            return StreamFeatures(streamDict)
            
        
@dataclass(frozen = True)
class CachedStream(Stream):
    cache_size: int = 1 ### number of frames before and after the current frame (e.g.: cache size 1 -> three frames in cache.)
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
        ### as opposed to the simple stream, the cached stream has to generate two caches which are constantly updated:
            # 1) the frameCache -> current frame at center, with future and past frames around
            # 2) the featCache -> current feature at center, with future and past features around
        frameCache = Cache(self.cache_size, [])
        # featCache = Cache(self.cache_size, [])
        ### Update: The featCache is not updated, but rebuilt, so that changes to past features can be used by featFunc.
        featIndices = frameCache.get_indices()
    
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame - self.cache_size)
        frmNum = self.start_frame - self.cache_size
        
        while self.cap.isOpened() and (self.start_frame - self.cache_size <= frmNum < self.stop_frame + self.cache_size):
            frmNum = int(self.cap.get(cv.CAP_PROP_POS_FRAMES)) ### the latest frame being read
            cntNum = frmNum - self.cache_size ### the central frame in the current iteration

            ret, currArr = self.cap.read()
            # feat = self._mut_features.maybe_get(frmNum)
            
            ### feed the cache
            frameCache = frameCache.add(currArr)
            # featCache = featCache.add(feat) ### this only ever changes a future feature! we can update changed past features.
            features = [self._mut_features.maybe_get(idx + cntNum) for idx in featIndices]
            featCache = Cache.from_list(features)
            
            ### check if we are supposed to yield -> only once cache is full!
            if frameCache.is_full():
                inArr = yield frmNum - frameCache.size, frameCache, featCache # take in frame, possibly altered, to show
            else:
                continue
            
            ### using the frame that comes back in, do something.
            ### if None is sent to generator, do nothing.
            if inArr is not None:
                if frame_counter:
                    cv.putText(inArr, str(frmNum), org = (40, 40), 
                               fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                               color = (0, 255, 0), thickness = 3)
                
                cv.imshow('frame', inArr)
                if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                    cv.destroyAllWindows() 
                    break
                    # yield None
        else:
            cv.destroyAllWindows()
        
    def extract(self, featureFunc, **kwargs):
        
        ### Establish function arguments and return shape
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)
        assert funcArgs == ['frameCache', 'featCache'], f'Function passed in to CachedStream.extract should have *["frameCache", "featCache"] as positional arguments, not *{funcArgs}'
        
        ### set the mutable features attribute
        ### used internally to dynamically update features and use them subsequently in extraction.
        _mut_features = self.features
        ### to allow this, they have to ...
        # ... conform to the beginning and end of the Stream (including cache).
        frameList = list(range(self.start_frame - self.cache_size, self.stop_frame + self.cache_size + 1))
        _mut_features = _mut_features.conform(frameList)
        # ... ensure that return values of the featureFunc are included in the mut_features, 
        # so that they may be written to and read out by subsequent iterations of the feature func.
        to_add = [f for f in funcVals if f not in _mut_features.get_features()]
        for feature in to_add:
            _mut_features = _mut_features.add(feature, None)
        
        object.__setattr__(self, '_mut_features', _mut_features)
        
        ### Initialize Generator
        gen = self.generator()
        frmNum, frameCache, featCache = next(gen) ### prime the generator
        
        streamDict = {}
        while True:
            ### call extraction function
            funcOut = featureFunc(frameCache, featCache, **kwargs)
            
            ### assign to a dictionary, depending on whether we get one or multiple return values
            if len(funcVals) == 1:
                updDict = {funcVals[0]: funcOut} 
            else:
                updDict = dict(zip(funcVals, funcOut))
            streamDict[frmNum] = updDict
            
            ### update the mutable features with the features just created
            featDict = self._mut_features[frmNum] ### these are all the features, those that are updated by featureFunc and those that aren't
            combDict = {}
            for feat in featDict:
                if feat in updDict:
                    combDict[feat] = updDict[feat]
                else:
                    combDict[feat] = featDict[feat]
            object.__setattr__(self, '_mut_features', self._mut_features.update(frmNum, combDict))
            
            ### iterate the generator.
            try:
                frmNum, frameCache, featCache = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
            except StopIteration:
                break 
        
        object.__delattr__(self, '_mut_features')
        
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        return StreamFeatures(streamDict).squeeze()
    
    def decache(self):
        return Stream(self.cap, self.start_frame, self.stop_frame, self.features)
    
    def show(self):
        self.decache().show()
    
    
    