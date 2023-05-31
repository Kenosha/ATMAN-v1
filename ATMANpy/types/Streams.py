# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:03:08 2023

@author: gianl
"""

import time
from dataclasses import dataclass
import cv2 as cv
from multiprocessing.pool import Pool, ThreadPool
from collections import deque
from random import randrange
from typing import List

from ATMANpy.types.Frames import BGRFrame, GreyFrame, HSVFrame
from ATMANpy.types.MultiFrames import GreyFrames
from ATMANpy.types.Features import StreamFeatures, StreamFeature, SFList
from ATMANpy.types.Caches import Cache

from ATMANpy.funcs.draw_funcs import draw
from ATMANpy.funcs.util_funcs import get_positional_arguments, get_return_names

# @dataclass(frozen = True)
# class StreamBase:
#     start_frame: int
#     stop_frame: int

#     def length(self):
#         return stop_frame - start_frame + 1 ### stream contains both start and stop frame!


def extract_mp_timer(extractMethod):

    def timed(*args, **kw):

        time_before = time.time()
        result = extractMethod(*args, **kw)
        time_after = time.time()

        timeStr = round(time_after - time_before, 2) 
        selfObj, otherArgs, exFunc = args[0], args[1:-1], args[-1]
        
        if len(otherArgs) == 1:
            num_workers = otherArgs[0]
            paramStr = f'{num_workers} workers'
        if len(otherArgs) == 2:
            num_workers, chunksize = otherArgs
            paramStr = f'{num_workers} workers and chunks of {chunksize}'
        print(f'{extractMethod.__name__} on {selfObj.__class__.__name__} with {exFunc.__name__} using {paramStr}: took {timeStr} seconds')

        return result

    return timed

@dataclass(frozen = True)
class Stream:
    cap: cv.VideoCapture
    start_frame: int
    stop_frame: int
    features: StreamFeatures = StreamFeatures({})
        
    @staticmethod
    def from_SF(mySF, cap):
        return Stream(cap, mySF.get_frames()[0], mySF.get_frames()[-1], mySF)
    
    def __post_init__(self):
        total_frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        if self.start_frame < 0:
            print(f'Negative start frame ({self.start_frame}) not available. Reinitializing with start_frame = 0...')
            object.__setattr__(self, 'start_frame', 0)
        
        if self.stop_frame > total_frames:
            print(f'{self.stop_frame} not available in cap of length {total_frames}. Reinitializing with stop_frame = {total_frames}...')
            object.__setattr__(self, 'stop_frame', total_frames)
        
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
       
    def argstrings_to_args(self, funcArgs, frmNum, rawArr):
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
        return args
    
    def attach(self, mySF: StreamFeatures):
        return self.__class__(self.cap, self.start_frame, self.stop_frame, features = mySF)

    def decap(self):
        myFrms = GreyFrames.from_video(self.cap, self.start_frame, self.stop_frame) ### adapt to all color spaces
        return DecapStream(myFrms, self.start_frame, self.stop_frame, self.features)

    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame)
        frmNum = self.start_frame
        
        while self.cap.isOpened() and (self.start_frame <= frmNum <= self.stop_frame):
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
            
            frmNum = int(self.cap.get(cv.CAP_PROP_POS_FRAMES))
        
        else:
            cv.destroyAllWindows()

    def chunk_generator(self, chunksize = 100):
        ### generate decapped chunks of the stream, one by one lazily
        chunk_start = self.start_frame
        chunk_stop = min(chunk_start + chunksize - 1, self.stop_frame) 
        ### since the last frame is included, a chunk of size 10 with chunk_start 5 ends at 14 = 5 + 10 - 1
        
        while not chunk_stop == self.stop_frame:
            yield Stream(self.cap, chunk_start, chunk_stop, self.features).decap()
            chunk_start = chunk_stop + 1
            chunk_stop = min(chunk_start + chunksize - 1, self.stop_frame)
        else:
            yield Stream(self.cap, chunk_start, chunk_stop, self.features).decap() ### yield one last time

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
          
    def extract(self, featureFunc, where = None, **kwargs): ### cv_funcs.get_stream_features
        ### where --> BoolFeature: give the option to apply the function not everywhere
        
        ### Establish function arguments and return shape
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)

        ### Initialize Generator
        gen = self.generator()
        try:
            frmNum, rawArr = next(gen) ### prime the generator
        except StopIteration:
            return StreamFeatures({})
        
        if where is not None:
            where = where.conform(self.features)
        
        streamDict = {}
        while True:
            
            ### get arguments to plug into feature extraction function
            args = Stream.argstrings_to_args(self, funcArgs, frmNum, rawArr) ### 
                
            ### assert that all arguments that we expect (funcArgs) we have been able to extract (args).
            assert len(args) == len(funcArgs)
            
            # ### assign to a dictionary, depending on whether we get one or multiple return values
            # if len(funcVals) == 1:
            #     myFeature = featureFunc(*args, **kwargs)
            #     streamDict[frmNum] = {funcVals[0]: myFeature}
            # else:
            #     myFeatures = featureFunc(*args, **kwargs)
            #     streamDict[frmNum] = dict(zip(funcVals, myFeatures))
            
            if where is None or where[frmNum]: ### if where is not given or is true at this position, we calculate the function value
                ### assign to a dictionary, depending on whether we get one or multiple return values
                if len(funcVals) == 1:
                    myFeature = featureFunc(*args, **kwargs)
                    streamDict[frmNum] = {funcVals[0]: myFeature}
                else:
                    myFeatures = featureFunc(*args, **kwargs)
                    streamDict[frmNum] = dict(zip(funcVals, myFeatures))
            else: ### if where is false, we just take the features in the stream
                selfDict = self.features.maybe_get(frmNum)
                combDict = {}
                for feat in funcVals:
                    if feat in selfDict:
                        combDict[feat] = selfDict[feat]
                    else:
                        combDict[feat] = None
                streamDict[frmNum] = combDict
                
            ### iterate the generator.
            try:
                frmNum, rawArr = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
            except StopIteration:
                break 
        
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        return StreamFeatures(streamDict).squeeze()
    
    def _extract_pool(self, poolCls, num_workers, featureFunc, where = None, **kwargs):
        """
        Auxiliary extract method for parallelisation 
            -> pass in a threadPool to do threaded extraction
            -> pass in a Pool to do multiprocessed extraction
            (see extract_threaded and extract_mp below)
        """
        WAIT_SECS = 0.001
        
        ### Set up the pool
        pool = poolCls(processes = num_workers)
        pending = deque()
        
        ### Establish function arguments and return shape
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)

        ### Initialize Generator
        gen = self.generator()
        try:
            frmNum, rawArr = next(gen) ### prime the generator
            generator_is_exhausted = False ### since the break condition is not StopIteration exception, we need this boolean.
        except StopIteration:
            pool.close()
            generator_is_exhausted = True
        
        if where is not None:
            where = where.conform(self.features)
            
        streamDict = {}
        while True:
            
            if not generator_is_exhausted and len(pending) < num_workers: #if there is space in the queue...
                if where is None or where[frmNum]: #and we actually want to calculate the function...
                    ### get arguments to plug into feature extraction function
                    args = Stream.argstrings_to_args(self, funcArgs, frmNum, rawArr)
                        
                    ### assert that all arguments that we expect (funcArgs) we have been able to extract (args).
                    assert len(args) == len(funcArgs)
                    
                    task = pool.apply_async(featureFunc, args, kwargs)
                    pending.append((frmNum, task))
            else:
                time.sleep(WAIT_SECS) ### need this to give the leftmost task time to finish!
            
            while len(pending) > 0 and pending[0][1].ready():
                taskNum, task = pending.popleft()
                
                if len(funcVals) == 1:
                    myFeature = task.get()
                    streamDict[taskNum] = {funcVals[0]: myFeature}
                else:
                    myFeatures = task.get()
                    streamDict[taskNum] = dict(zip(funcVals, myFeatures))
                
            if not generator_is_exhausted and len(pending) < num_workers: ### only when there is space in the queue
                ### iterate the generator.
                try:
                    frmNum, rawArr = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
                except StopIteration:
                    pool.close() ### pool closed once generator has nothing to yield --> no new elements will be added.
                    generator_is_exhausted = True
                    continue ### stop condition is not here anymore! we have to keep going until pending is empty and all tasks are done
                
                
            ### stop condition
            if generator_is_exhausted and len(pending) == 0:
                pool.terminate() ### once pending queue is zero, we can terminate the pool.
                break
        
        if where is not None:
            outDict = {}
            for frmNum in self.features.get_frames():
                if where[frmNum]:
                    outDict[frmNum] = streamDict[frmNum]
                else:
                    outDict[frmNum] = {featName: feat for featName, feat in self.features[frmNum].items() if featName in funcVals}
            streamDict = outDict.copy()
         
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        return StreamFeatures(streamDict).squeeze()
    
    def _extract_chunked(self, poolCls, num_workers, chunksize, featureFunc, _join = True, **kwargs):
        ### IDEA: Instead of create a pool of jobs that workers get to in parallel, 
        # create a pool of chunks: 
        # chunk the Stream and let each chunk be worked by one worker.
        # This is especially important for CachedStream, below, where we need sequential doing of tasks
        # but still want the speedup of parallel computing. 
        # With pooled parallelism, we can not guarantee that the frame before is already done
        # when the task for a given frame is dispatched, so that functions that track (needing prior frame information)
        # can not be applied.
        WAIT_SECS = 0.001
        
        ### Set up the pool
        pool = Pool(processes = num_workers)
        pending = deque()
        
        ### Initialize Generator
        gen = self.chunk_generator(chunksize = chunksize)
        try:
            chunkStm = next(gen) ### prime the generator
            generator_is_exhausted = False ### since the break condition is not StopIteration exception, we need this boolean.
        except StopIteration:
            pool.close()
            generator_is_exhausted = True
    
        sfList = []
        while True:
            if not generator_is_exhausted and len(pending) < num_workers:                
                ### if there is space in the queue, add a task!
                # print(f'Queueing {chunkStm}...')
                task = pool.apply_async(chunkStm.extract, [featureFunc], kwargs)
                pending.append(task)
            else:
                time.sleep(WAIT_SECS) ### need this to give the leftmost task time to finish!
            
            while len(pending) > 0 and pending[0].ready():
                mySF = pending.popleft().get()
                sfList.append(mySF)
            
            if not generator_is_exhausted and len(pending) < num_workers: ### only when there is space in the queue
                ### iterate the generator.
                try:
                    chunkStm = next(gen) ### iterate the generator
                    generator_is_exhausted = False ### since the break condition is not StopIteration exception, we need this boolean.
                except StopIteration:
                    pool.close() ### pool closed once generator has nothing to yield --> no new elements will be added.
                    generator_is_exhausted = True
                    continue ### stop condition is not here anymore! we have to keep going until pending is empty and all tasks are done
                
            ### stop condition
            if generator_is_exhausted and len(pending) == 0:
                pool.terminate() ### once pending queue is zero, we can terminate the pool.
                break
         
        if _join:
            return SFList(sfList).concatenate(axis = 0)
        else:
            return SFList(sfList)
    
    @extract_mp_timer
    def extract_threaded(self, num_workers, featureFunc, where = None, **kwargs):
        return Stream._extract_pool(self, ThreadPool, num_workers, featureFunc, where = None, **kwargs)
    
    @extract_mp_timer
    def extract_mp(self, num_workers, featureFunc, where = None, **kwargs):
        return Stream._extract_pool(self, Pool, num_workers, featureFunc, where = None, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_threaded(self, num_workers, chunksize, featureFunc, **kwargs):
        return Stream._extract_chunked(self, ThreadPool, num_workers, chunksize, featureFunc, _join = True, **kwargs)
            
    @extract_mp_timer
    def extract_chunked_mp(self, num_workers, chunksize, featureFunc, **kwargs):
        return Stream._extract_chunked(self, Pool, num_workers, chunksize, featureFunc, _join = True, **kwargs)
            
@dataclass(frozen = True)
class StreamList:
    streams: List[Stream]
    
    @staticmethod
    def from_SFList(sfList, cap):
        return StreamList([Stream.from_SF(mySF, cap) for mySF in sfList])
    
    def __repr__(self):
        return f'StreamList({self.length()} Streams from {self[0].start_frame} to {self[-1].stop_frame})'
    
    def __iter__(self):
        return self.streams.__iter__()
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.streams[item]
        else:
            return StreamList(self.streams[item])
        
    def length(self):
        return len(self.streams)
        
    def chunk_generator(self, chunksize = None): ### chunks are simply the elements!
        for myStm in self:
            yield myStm.decap()
        
    def extract(self, featureFunc, **kwargs):
        outList = []
        for myStm in self:
            mySF = myStm.extract(featureFunc, **kwargs)
            outList.append(mySF)
        return SFList(outList)
        
    def extract_chunked_threaded(self, num_workers, featureFunc, **kwargs):
        chunksize = None
        return Stream._extract_chunked(self, ThreadPool, num_workers, chunksize, featureFunc, _join = False, **kwargs)
    
    def extract_chunked_mp(self, num_workers, featureFunc, **kwargs):
        chunksize = None
        return Stream._extract_chunked(self, Pool, num_workers, chunksize, featureFunc, _join = False, **kwargs)
        
        
@dataclass(frozen = True)
class DecapStream:
    ### here, the Stream is dissociated from its videoCapture, which is turned into a MultiFrames object.
    frames: GreyFrames
    start_frame: int
    stop_frame: int
    features: StreamFeatures = StreamFeatures({})

    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        for idx, frame in enumerate(self.frames):
            
            frmNum = self.start_frame + idx
            frame = yield frmNum, frame # take in frame, possibly altered, to show 
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
            
    def extract(self, featureFunc, **kwargs):
        return Stream.extract(self, featureFunc, **kwargs)
    
    def extract_mp(self, num_workers, featureFunc, **kwargs):
        return Stream.extract_mp(self, num_workers, featureFunc, **kwargs)
    
@dataclass(frozen = True)
class NCDecapStream:
    ### Non-Consecutive Decapped Stream -> a collection of frames which do not have to be consecutive
    frames: GreyFrames
    frameNums: List
    features: StreamFeatures = StreamFeatures({})
    
    def __repr__(self):
        if self.is_empty():
            return 'NCDecapStream({})'
        else:
            return f'NCDecapStream(frames = {self.frames}, frameNums = {self.frameNums[0]} ... {self.frameNums[-1]}, features = {self.features})'
    
    def length(self):
        return self.frames.length()
    
    def is_empty(self):
        return self.length() == 0
    
    @staticmethod
    def from_bool(boolSF, cap):
        filterSF = boolSF.filter(boolSF) ## keep only true entries
        frameNums = list(filterSF.keys())
        frames = GreyFrames.from_list(cap, frameNums) ### TODO: speed up the get_framelist function by threading.
        return NCDecapStream(frames, frameNums, filterSF)
    
    def attach(self, mySF: StreamFeatures):
        return NCDecapStream(self.frames, self.frameNums, features = mySF)
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        for frame, frameNum in zip(self.frames, self.frameNums):
            
            frame = yield frameNum, frame # take in frame, possibly altered, to show 
            ## TODO: include featDict as a thing the generator yields!
            
            if frame_counter:
                cv.putText(frame, str(frameNum), org = (40, 40), 
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
    
    def extract(self, featureFunc, **kwargs):
        return Stream.extract(self, featureFunc, **kwargs)
    
    def extract_mp(self, num_workers, featureFunc, **kwargs):
        return Stream.extract_mp(self, num_workers, featureFunc, **kwargs)
    
@dataclass(frozen = True) ### despite a frozen dataclass, within the functions below, the features attribute is treated as mutable, so we can update it and use it dynamically for subsequent iterations
class CachedStream(Stream):
    cache_size: int = 1 ### number of frames before and after the current frame (e.g.: cache size 1 -> three frames in cache.)
    
    def _set_mut_features(self, features_needed = None):
        ### set the mutable features attribute
        ### used internally to dynamically update features and use them subsequently in extraction.
        _mut_features = self.features
        ### to allow this, they have to ...
        # ... conform to the beginning and end of the Stream (including cache).
        frameList = list(range(self.start_frame - self.cache_size, self.stop_frame + self.cache_size + 1))
        _mut_features = _mut_features.conform(frameList)
        # ... ensure that return values of the featureFunc are included in the mut_features, 
        # so that they may be written to and read out by subsequent iterations of the feature func.
        if not features_needed is None:
            to_add = [f for f in features_needed if f not in _mut_features.get_features()]
            for feature in to_add:
                _mut_features = _mut_features.add(feature, None)
        
        object.__setattr__(self, '_mut_features', _mut_features)
    
    def _update_mut_features(self, frmNum, updDict):
        ### update the mutable features with the features just created
        featDict = self._mut_features[frmNum] ### these are all the features, those that are updated by featureFunc and those that aren't
        combDict = {}
        for feat in featDict:
            if feat in updDict:
                combDict[feat] = updDict[feat]
            else:
                combDict[feat] = featDict[feat]
        object.__setattr__(self, '_mut_features', self._mut_features.update(frmNum, combDict))
    
    def _delete_mut_features(self):
        object.__delattr__(self, '_mut_features')
        
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
        ### as opposed to the simple stream, the cached stream has to generate two caches which are constantly updated:
            # 1) the frameCache -> current frame at center, with future and past frames around
            # 2) the featCache -> current feature at center, with future and past features around    
        frameCache = Cache(self.cache_size, [])
        # featCache = Cache(self.cache_size, [])
        ### Update: The featCache is not updated, but rebuilt, so that changes to past features can be used by featFunc.
        featIndices = frameCache.get_indices()
    
        ### mutable features always have to exist here!
        assert hasattr(self, '_mut_features'), 'To create a generator for a CachedStream, mutable features always have to exist!'
    
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame - self.cache_size)
        frmNum = self.start_frame - self.cache_size
        
        while self.cap.isOpened() and (self.start_frame - self.cache_size <= frmNum <= self.stop_frame + self.cache_size):
            cntNum = frmNum - self.cache_size ### the central frame in the current iteration

            ret, currArr = self.cap.read()
            
            ### feed/build the caches
            frameCache = frameCache.add(currArr)
            features = [self._mut_features.maybe_get(idx + cntNum) for idx in featIndices]
            featCache = Cache.from_list(features)
            
            ### check if we are supposed to yield -> only once cache is full!
            if frameCache.is_full():
                inArr = yield frmNum - frameCache.size, frameCache, featCache # take in frame, possibly altered, to show
            else:
                inArr = None
            
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
                    
            frmNum = int(self.cap.get(cv.CAP_PROP_POS_FRAMES)) ### the latest frame being read
        else:
            cv.destroyAllWindows()
        
    def chunk_generator(self, chunksize):
        assert chunksize > 2 * self.cache_size, f'Chunks have to be large enough to accomodate the cache! With cache size {self.cache_size}, a minimum chunk size of {self.cache_size * 2 + 1} is needed!'
        
        start_frame_with_cache = self.start_frame - self.cache_size
        stop_frame_with_cache = self.stop_frame + self.cache_size
        
        ### generate decapped chunks of the stream, one by one lazily
        chunk_start = start_frame_with_cache
        chunk_stop = min(chunk_start + chunksize - 1, stop_frame_with_cache) 
        ### since the last frame is included, a chunk of size 10 with chunk_start 5 ends at 14 = 5 + 10 - 1
        
        while not chunk_stop == stop_frame_with_cache:
            yield CachedStream(self.cap, chunk_start, chunk_stop, self.features, cache_size = self.cache_size).decap()
            last_proper_frame = chunk_stop - self.cache_size ### this is the last frame that we can expect to be done properly, with full cache.
            next_proper_frame = last_proper_frame + 1 ### for the next chunk, this is the first frame we want to be done properly.
            chunk_start = next_proper_frame - self.cache_size ### importantly, the next chunk does not start right after the previous chunk, but there is some overlap, so we can include the cache specific frames!
            chunk_stop = min(chunk_start + chunksize - 1, stop_frame_with_cache)
        else:
            ### yield one last time
            yield CachedStream(self.cap, chunk_start, chunk_stop, self.features, cache_size = self.cache_size).decap() 
       
    def extract(self, featureFunc, **kwargs):
        
        ### Establish function arguments and return shape
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)
        assert funcArgs == ['frameCache', 'featCache'], f'Function passed in to CachedStream.extract should have *["frameCache", "featCache"] as positional arguments, not *{funcArgs}'
        
        self._set_mut_features(features_needed = funcVals)
        
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
            
            ### update mutable features with what we have just calculated
            self._update_mut_features(frmNum, updDict)
            
            ### iterate the generator.
            try:
                frmNum, frameCache, featCache = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
            except StopIteration:
                break 
        
        ### delete the mutable features -> they are not needed, and should always be deleted since they can be large.
        self._delete_mut_features()
        
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        return StreamFeatures(streamDict).squeeze()
       
    def extract_chunked_threaded(self, num_workers, chunksize, featureFunc, **kwargs):
        return Stream.extract_chunked_threaded(self, num_workers, chunksize, featureFunc, **kwargs)
            
    def extract_chunked_mp(self, num_workers, chunksize, featureFunc, **kwargs):
        return Stream.extract_chunked_mp(self, num_workers, chunksize, featureFunc, **kwargs)
                 
    def decache(self):
        return Stream(self.cap, self.start_frame, self.stop_frame, self.features)
    
    def decap(self):
        myFrms = GreyFrames.from_video(self.cap, self.start_frame, self.stop_frame) ### adapt to all color spaces
        return DecapCachedStream(myFrms, self.start_frame, self.stop_frame, self.features, cache_size = self.cache_size)

    def show(self):
        self.decache().show()
    
    
@dataclass(frozen = True)
class DecapCachedStream(DecapStream):
    cache_size: int = 1
    
    _set_mut_features = CachedStream._set_mut_features
    _update_mut_features = CachedStream._update_mut_features        
    _delete_mut_features = CachedStream._delete_mut_features
        
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
        ### as opposed to the simple stream, the cached stream has to generate two caches which are constantly updated:
            # 1) the frameCache -> current frame at center, with future and past frames around
            # 2) the featCache -> current feature at center, with future and past features around    
        frameCache = Cache(self.cache_size, [])
        # featCache = Cache(self.cache_size, [])
        ### Update: The featCache is not updated, but rebuilt, so that changes to past features can be used by featFunc.
        featIndices = frameCache.get_indices()
    
        ### mutable features always have to exist here!
        assert hasattr(self, '_mut_features'), 'To create a generator for a CachedStream, mutable features always have to exist!'
    
        for idx, frame in enumerate(self.frames):
            frmNum = self.start_frame + idx            
            cntNum = frmNum - self.cache_size ### the central frame in the current iteration

            ### feed/build the caches
            frameCache = frameCache.add(frame)
            features = [self._mut_features.maybe_get(idx + cntNum) for idx in featIndices]
            featCache = Cache.from_list(features)
            
            ### check if we are supposed to yield -> only once cache is full!
            if frameCache.is_full():
                inArr = yield frmNum - frameCache.size, frameCache, featCache # take in frame, possibly altered, to show
            else:
                inArr = None
            
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
        return CachedStream.extract(self, featureFunc, **kwargs)
        
@dataclass(frozen = True)
class RandStream(Stream):
    number_of_frames: int = 100
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        frmNum = randrange(self.start_frame, self.stop_frame)
        frames_yielded = 0
    
        while self.cap.isOpened() and frames_yielded < self.number_of_frames:
            
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
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
            
            frmNum = randrange(self.start_frame, self.stop_frame)
        
        else:
            cv.destroyAllWindows()
    
     
    