# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:03:08 2023

@author: gianl
"""

import time
import os
import cv2 as cv
import numpy as np
import psutil

from dataclasses import dataclass
import multiprocessing
# multiprocessing.set_start_method('spawn')
from multiprocessing.pool import Pool, ThreadPool
from collections import deque
from random import randrange
from typing import List, Union

from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, HSVFrame, BoolFrame, MaskFrame
from ATMANpy.types.MultiFrames import GreyFrames, BGRFrames, HSVFrames, BoolFrames, MaskFrames
from ATMANpy.types.Features import StreamFeatures, StreamFeature, SFList, ConstFeatures, ConstFeature
from ATMANpy.types.Caches import Cache

from ATMANpy.funcs.cv_funcs import get_frame
from ATMANpy.funcs.draw_funcs import draw
from ATMANpy.funcs.util_funcs import get_positional_arguments, get_return_names, get_folder_struct

# @dataclass(frozen = True)
# class StreamBase:
#     start_frame: int
#     stop_frame: int

#     def length(self):
#         return stop_frame - start_frame + 1 ### stream contains both start and stop frame!

nameMap = {'bgr': 'cFrm',
           'grey': 'gFrm',
           'hsv': 'hFrm',
           'bool': 'bFrm',
           'mask': 'mFrm'}

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

def color_converter(imgArr, cap_color_space, target_color_space):
    methodMap = {'bgr': 'to_BGR',
                 'grey': 'to_grey',
                 'hsv': 'to_HSV',
                 'bool': 'to_bool',
                 'mask': 'to_mask'} 
    
    frameType = frameTypeMap[cap_color_space]
    
    origFrm = frameType(imgArr)
    
    if cap_color_space == target_color_space:
        return origFrm
    else:
        conversionFuncStr = methodMap[target_color_space]
        return getattr(origFrm, conversionFuncStr)()

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

def get_updated_attribute_dictionary(someObj, updDict):
    currDict = someObj.__dict__
    
    def maybe_update(attr):
        try:
            return updDict[attr]
        except KeyError:
            return currDict[attr]
    
    return {attr: maybe_update(attr) for attr in currDict if not attr.startswith('_')}

@dataclass(frozen = True)
class Stream:
    cap: cv.VideoCapture
    start_frame: int
    stop_frame: int
    truncation_anchor: int = 0 ### instead of a truncation flag. this anchor frame allows comparison between truncated streams and non-truncated streams by specifying the first frame of truncation.
    cap_color_space: str = 'bgr'
    target_color_space: str = 'bgr'
    scale_factor: float = 1 ### ability to shrink the image relative to the cap dimensions
    warp_dims: tuple = None ### ability to warp the stream using an H in features and dimensions supplied.
    name: str = None ### if none is left, it defaults to cFrm for bgr, gFrm for grey, hFrm for hsv, bFrm for bool
    features: StreamFeatures = StreamFeatures({})
        
    @staticmethod
    def from_SF(mySF, cap, name = None,
                cap_color_space = 'bgr', target_color_space = 'bgr',
                scale_factor = 1, warp_dims = None):
        start, stop = mySF.get_frames()[0], mySF.get_frames()[-1]
        return Stream(cap, start, stop, 
                      truncation_anchor = 0,
                      cap_color_space = cap_color_space, target_color_space = target_color_space, 
                      scale_factor = scale_factor, warp_dims = warp_dims,
                      features = mySF, name = name)
    
    @staticmethod
    def from_file(filePath, name = None,
                  cap_color_space = 'bgr', target_color_space = 'bgr',
                  scale_factor = 1):
        cap = cv.VideoCapture(filePath)
        
        fileName = os.path.basename(filePath)
        
        ### find start and end frame
        fileStump = fileName.split('.')[0]
        _, start, _, stop = fileStump.split('_')
        start, stop = int(start), int(stop)
        
        return Stream(cap, start, stop, 
                      truncation_anchor = start, ### NOTE: truncation anchor is the first frame given in file name! 
                      cap_color_space = cap_color_space, target_color_space = target_color_space, 
                      scale_factor = scale_factor,
                      name = name)
    
    @staticmethod
    def from_store(fileName, featureName, videoName, storeName, 
                   cap_color_space = 'bgr', target_color_space = 'bgr',
                   scale_factor = 1):
        featDict = get_folder_struct('features_stored')
        featPath = featDict['main']
        ### find video
        videoPath = os.path.join(featPath, videoName, storeName, featureName, fileName)
        return Stream.from_file(videoPath, name = featureName,
                                cap_color_space = cap_color_space, target_color_space = target_color_space,
                                scale_factor = scale_factor)
        
    def __post_init__(self):
        ### CHECK FRAMES
        total_frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        if self.start_frame < 0:
            print(f'Negative start frame ({self.start_frame}) not available. Reinitializing with start_frame = 0...')
            object.__setattr__(self, 'start_frame', 0)
        
        if self.stop_frame > self.start_frame + total_frames:
            print(f'{self.stop_frame} not available in cap of length {total_frames}. Reinitializing with stop_frame = {total_frames}...')
            object.__setattr__(self, 'stop_frame', total_frames)
        
        ### CHECK NAME
        if self.name is None:
            object.__setattr__(self, 'name', nameMap[self.target_color_space])
        
        ### CHECK FEATURES
        mySF = self.features
        if isinstance(self.features, StreamFeature) or isinstance(self.features, ConstFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        ### conform StreamFeatures to start and stop frame
        # mySF = mySF.conform(list(range(self.start_frame, self.stop_frame + 1))) ### capture contains last frame!
        object.__setattr__(self, 'features', mySF)
    
        if self.warp_dims is not None:
            assert 'H' in self.features.get_features(), 'If warp dimensions are specified, we need a Homography in features to warp with!'
            H = self.features['H']
            assert H.dropna().length() == H.length() == self.length(), 'When warping is specified, all frames in stream must have a homography.'
    
    
    def __getitem__(self, frmNum):
        ### TODO: allow for item to be a slice also! (this would yield a multiframes)
        assert self.start_frame <= frmNum <= self.stop_frame, f'Frame requested has to be between {self.start_frame} and {self.stop_frame}'
        outType = frameTypeMap[self.target_color_space]
        
        get_num = frmNum - self.truncation_anchor
        
        outFrm = outType.from_video(self.cap, get_num, 
                                    cap_color_space = self.cap_color_space, 
                                    scale_factor = self.scale_factor,
                                    name = self.name)
    
        for feat in self.features.get_features():
            outFrm = outFrm.put(self.features[frmNum][feat])
            
        if self.warp_dims is not None:
            width, height = self.warp_dims
            H = self.features['H'][frmNum]
            H_inv = np.linalg.inv(H)
            outFrm = outFrm.warp(H_inv, cols = width, rows = height)
        
        return outFrm
        
    @staticmethod
    def argstrings_to_args(funcArgs, myFrm, myDict):
        ### get arguments to plug into feature extraction function
        
        # differentiate between function arguments that need a stream and those that need a feature
        is_from_stream = lambda myStr: myStr.endswith('Mask') or myStr.endswith('Frm')
        
        args = []
        for argStr in funcArgs:
            if is_from_stream(argStr):
                ### In the MulStream case, this assertion will be weakened, 
                ### but when extracting with a Stream, only one frame argument can be supplied, 
                ### and it needs to match the name of the frame to be generated by the Stream.
                assert myFrm.name == argStr, f'Extracting function that needs {argStr} with frame {myFrm.name}'
                argObj = myFrm
            else:
                try:
                    argObj = myDict[argStr]
                except KeyError as e:
                    raise e
            args.append(argObj)
        return args
    
    def length(self):
        return self.stop_frame - self.start_frame + 1
    
    def dims(self):
        width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)) 
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def size(self, unit = 'GB'):
        ### size of raw video in stream
        unitMap = {'B': 1, 'KB': 1000, 'MB': 1000 ** 2, 'GB': 1000 ** 3}
        
        if self.target_color_space in ['hsv', 'bgr']:
            mult = 3
        else:
            mult = 1
            
        width, height = self.dims()
        length = self.length()
        return round(length * width * height * mult / unitMap[unit], 2)
    
    def conform(self, start_frame, stop_frame):
        updDict = {'start_frame': start_frame, 'stop_frame': stop_frame}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def recolor(self, new_target_color_space):
        updDict = {'target_color_space': new_target_color_space}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def rescale(self, new_scale_factor):
        updDict = {'scale_factor': new_scale_factor}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def attach(self, mySF: StreamFeatures):
        updDict = {'features': mySF}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)

    def decap(self):
        framesType = framesTypeMap[self.target_color_space]
        
        cap_start = self.start_frame - self.truncation_anchor
        cap_stop = self.stop_frame - self.truncation_anchor
        
        myFrms = framesType.from_video(self.cap, cap_start, cap_stop,
                                       cap_color_space = self.cap_color_space,
                                       scale_factor = self.scale_factor,
                                       name = self.name)
        ### warp if necessary
        if self.warp_dims is not None:
            width, height = self.warp_dims
            HInvArr = self.features['H'].apply(np.linalg.inv).to_array()
            myFrms = myFrms.warp(HInvArr, cols = width, rows = height)
            
        return DecapStream(myFrms, self.start_frame, self.stop_frame, self.features)

    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.start_frame - self.truncation_anchor)        
        frmNum = self.start_frame
        
        while self.cap.isOpened() and (self.start_frame <= frmNum <= self.stop_frame):
            ret, frame = self.cap.read()
            
            ### Get the frame in the correct color type
            myFrm = color_converter(frame, self.cap_color_space, self.target_color_space)
            myFrm = myFrm.rename(self.name)
            
            ### Warp if necessary
            if self.warp_dims is not None:
                width, height = self.warp_dims
                H = self.features['H'][frmNum]
                H_inv = np.linalg.inv(H)
                myFrm = myFrm.warp(H_inv, cols = width, rows = height)
                
            ### rescale if necessary
            if self.scale_factor != 1:
                myFrm = myFrm.rescale(self.scale_factor)
            
            ### Get the feature dictionary
            myDict = self.features.maybe_get(frmNum)
            ### if frmNum not found, we get {feat: None for feat in self.get_features()}
            
            myFrm = yield frmNum, myFrm, myDict # take in frame, possibly altered, to show 
            
            if frame_counter:
                myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
            
            if myFrm is not None:
                still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                if not still_showing:
                    break
            
            frmNum += 1
        
        else:
            cv.destroyAllWindows()

    def maxchunk(self, critical_value = 4, safety_margin = 0.1): 
        ### critical_value = 4 -> larger than 4 GB, we can not pickle!
        ### safety_margin = 0.1 -> want to be 10% below critical value
        size_per_unit = self.size(unit = 'GB') / self.length()
        admissible_units = int((1 - safety_margin) * critical_value / size_per_unit)
        return admissible_units
    
    def chunk_generator(self, chunksize = 100):
        chunksize = min(chunksize, self.maxchunk())
        
        ### generate decapped chunks of the stream, one by one lazily
        chunk_start = self.start_frame
        chunk_stop = min(chunk_start + chunksize - 1, self.stop_frame) 
        ### since the last frame is included, a chunk of size 10 with chunk_start 5 ends at 14 = 5 + 10 - 1
        
        while not chunk_stop == self.stop_frame:
            yield self.conform(chunk_start, chunk_stop).decap()
            chunk_start = chunk_stop + 1
            chunk_stop = min(chunk_start + chunksize - 1, self.stop_frame)
        else:
            yield self.conform(chunk_start, chunk_stop).decap() ### yield one last time

    def show(self, ms_between_frames = 25, frame_counter = False):
        gen = self.generator(ms_between_frames = ms_between_frames, frame_counter = frame_counter)
        frmNum, myFrm, myDict = next(gen) ### prime the generator
        drawFunc = draw(self.features.get_features())
        
        while True:
            sendFrm = drawFunc({'cFrm': myFrm}, myDict) ### TODO: Clean this up! myFrm is not always cFrm!
                 
            try:
                frmNum, myFrm, myDict = gen.send(sendFrm)
            except StopIteration:
                cv.destroyAllWindows()
                break 
          
    def extract(self, featureFunc, **kwargs): ### cv_funcs.get_stream_features
        ### Establish function arguments and return shape
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)

        ### Initialize Generator
        gen = self.generator()
        try:
            frmNum, myFrm, myDict = next(gen) ### prime the generator
        except StopIteration:
            return StreamFeatures({})
        
        streamDict = {}
        while True:
            
            ### get arguments to plug into feature extraction function
            args = self.__class__.argstrings_to_args(funcArgs, myFrm, myDict) ### 
                
            ### assert that all arguments that we expect (funcArgs) we have been able to extract (args).
            assert len(args) == len(funcArgs), f'Required arguments ({funcArgs}) have to match supplied arguments ({args})'
            
            ### assign to a dictionary, depending on whether we get one or multiple return values
            if len(funcVals) == 1:
                myFeature = featureFunc(*args, **kwargs)
                streamDict[frmNum] = {funcVals[0]: myFeature}
            else:
                myFeatures = featureFunc(*args, **kwargs)
                streamDict[frmNum] = dict(zip(funcVals, myFeatures))
            
            ### iterate the generator.
            try:
                frmNum, myFrm, myDict = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
            except StopIteration:
                break 
        
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        return StreamFeatures(streamDict).squeeze()
    
    def extract_to_disc(self, featureFunc, videoName, storeName, savekwargs, **kwargs):
        outSF = self.extract(featureFunc, **kwargs)
        outSF.save(videoName, storeName, **savekwargs)
    
    def _extract_pool(self, featureFunc, poolCls, num_workers, **kwargs):
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
            frmNum, myFrm, myDict = next(gen) ### prime the generator
            generator_is_exhausted = False ### since the break condition is not StopIteration exception, we need this boolean.
        except StopIteration:
            pool.close()
            generator_is_exhausted = True
        
        streamDict = {}
        while True:
            
            if not generator_is_exhausted and len(pending) < num_workers: #if there is space in the queue...
                ### get arguments to plug into feature extraction function
                args = self.__class__.argstrings_to_args(funcArgs, myFrm, myDict)
                    
                ### assert that all arguments that we expect (funcArgs) we have been able to extract (args).
                assert len(args) == len(funcArgs), f'Required arguments ({funcArgs}) have to match supplied arguments ({args})'
                
                
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
                    frmNum, myFrm, myDict = gen.send(None) ### sending None -> nothing is shown! TODO: "Loud" option -> extract + show.
                except StopIteration:
                    pool.close() ### pool closed once generator has nothing to yield --> no new elements will be added.
                    generator_is_exhausted = True
                    continue ### stop condition is not here anymore! we have to keep going until pending is empty and all tasks are done
                
                
            ### stop condition
            if generator_is_exhausted and len(pending) == 0:
                pool.terminate() ### once pending queue is zero, we can terminate the pool.
                break
        
        ### write to StreamFeature or StreamFeatures, depending on the number of return values.
        return StreamFeatures(streamDict).squeeze()
    
    def _extract_pool_to_disc(self, featureFunc, poolCls, num_workers, videoName, storeName, savekwargs, **kwargs):
        outSF = self._extract_pool(self, featureFunc, poolCls, num_workers, **kwargs)
        outSF.save(videoName, storeName, **savekwargs)
    
    def _extract_chunked(self, featureFunc, poolCls, num_workers, chunksize, _split = False, **kwargs):
        ### IDEA: Instead of create a pool of jobs that workers get to in parallel, 
        # create a pool of chunks: 
        # chunk the Stream and let each chunk be worked by one worker.
        # This is especially important for CachedStream, below, where we need sequential doing of tasks
        # but still want the speedup of parallel computing. 
        # With pooled parallelism, we can not guarantee that the frame before is already done
        # when the task for a given frame is dispatched, so that functions that track (needing prior frame information)
        # can not be applied.
                
        def ram_is_ok(max_size_of_stream = 4, buffer_needed_gb = 8):
            ### max size of stream is due to the impossibility of pickling an object larger than 4GB
            ### check if an stream, which we want to decap, fits into RAM, allowing for a buffer.
            available_gb = psutil.virtual_memory().available / 1024 ** 3
            return max_size_of_stream < available_gb - buffer_needed_gb
        
        WAIT_SECS = 0.001
        MESSAGE_FREQUENCY = 10
        VERBOSE = False
        if VERBOSE:
            time_of_last_message = time.time()
        
        ### Set up the pool
        pool = poolCls(processes = num_workers)
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
            ram_ok = ram_is_ok() ### have to check it once! 
                                 ### so we dont queue the generator twice without scheduling 
                                 ### or queue the same task twice...
            
            if not generator_is_exhausted and len(pending) < num_workers and ram_ok:                
                ### if there is space in the queue, add a task!
                # print(f'Queueing {chunkStm}...')
                # print(kwargs)
                task = pool.apply_async(chunkStm.extract, [featureFunc], kwargs)
                pending.append(task)
            else:
                time.sleep(WAIT_SECS) ### need this to give the leftmost task time to finish!
            
            while len(pending) > 0 and pending[0].ready():
                mySF = pending.popleft().get()
                sfList.append(mySF)
            
            if not generator_is_exhausted and len(pending) < num_workers and ram_ok: ### only when there is space in the queue
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
            else: # not done! communicate status
                if VERBOSE:
                    time_since_last_message = time.time() - time_of_last_message
                    if time_since_last_message > MESSAGE_FREQUENCY:
                        len_queue = len(pending)
                        ram_pct = psutil.virtual_memory().percent
                        print(f'   ... length of queue: {len(pending)}. Ram utilization: {ram_pct}%')
                        time_of_last_message = time.time()
        
        allSFs = SFList(sfList).concatenate(axis = 0)
            
        if _split:
            return allSFs.split()
        else:
            return allSFs
    
    def _extract_chunked_to_disc(self, featureFunc, poolCls, num_workers, chunksize, videoName, storeName, 
                                 savekwargs, **kwargs):
        ### IDEA: Instead of create a pool of jobs that workers get to in parallel, 
        # create a pool of chunks: 
        # chunk the Stream and let each chunk be worked by one worker.
        # This is especially important for CachedStream, below, where we need sequential doing of tasks
        # but still want the speedup of parallel computing. 
        # With pooled parallelism, we can not guarantee that the frame before is already done
        # when the task for a given frame is dispatched, so that functions that track (needing prior frame information)
        # can not be applied.
                
        def ram_is_ok(max_size_of_stream = 4, buffer_needed_gb = 8):
            ### max size of stream is due to the impossibility of pickling an object larger than 4GB
            ### check if an stream, which we want to decap, fits into RAM, allowing for a buffer.
            available_gb = psutil.virtual_memory().available / 1024 ** 3
            return max_size_of_stream < available_gb - buffer_needed_gb
        
        WAIT_SECS = 0.001
        MESSAGE_FREQUENCY = 10
        VERBOSE = False
        if VERBOSE:
            time_of_last_message = time.time()
        
        ### Set up the pool
        pool = poolCls(processes = num_workers)
        pending = deque()
        
        ### Initialize Generator
        gen = self.chunk_generator(chunksize = chunksize)
        try:
            chunkStm = next(gen) ### prime the generator
            generator_is_exhausted = False ### since the break condition is not StopIteration exception, we need this boolean.
        except StopIteration:
            pool.close()
            generator_is_exhausted = True
    
        while True:
            ram_ok = ram_is_ok() ### have to check it once! 
                                 ### so we dont queue the generator twice without scheduling 
                                 ### or queue the same task twice...
            
            if not generator_is_exhausted and len(pending) < num_workers and ram_ok:                
                ### if there is space in the queue, add a task!
                # print(f'Queueing {chunkStm}...')
                # print(kwargs)
                task = pool.apply_async(chunkStm.extract_to_disc, [featureFunc, videoName, storeName, savekwargs], kwargs)
                pending.append(task)
            else:
                time.sleep(WAIT_SECS) ### need this to give the leftmost task time to finish!
            
            while len(pending) > 0 and pending[0].ready():
                pending.popleft().get()
                
            if not generator_is_exhausted and len(pending) < num_workers and ram_ok: ### only when there is space in the queue
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
            else: # not done! communicate status
                if VERBOSE:
                    time_since_last_message = time.time() - time_of_last_message
                    if time_since_last_message > MESSAGE_FREQUENCY:
                        len_queue = len(pending)
                        ram_pct = psutil.virtual_memory().percent
                        print(f'   ... length of queue: {len(pending)}. Ram utilization: {ram_pct}%')
                        time_of_last_message = time.time()
        
        
    @extract_mp_timer
    def extract_threaded(self, featureFunc, num_workers, **kwargs):
        return Stream._extract_pool(self, featureFunc, ThreadPool, num_workers, **kwargs)
    
    @extract_mp_timer
    def extract_mp(self, featureFunc, num_workers, **kwargs):
        return Stream._extract_pool(self, featureFunc, Pool, num_workers, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_threaded(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, ThreadPool, num_workers, chunksize, _split = False, **kwargs)
            
    @extract_mp_timer
    def extract_chunked_mp(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, Pool, num_workers, chunksize, _split = False, **kwargs)
       
    @extract_mp_timer
    def extract_chunked_threaded_to_disc(self, featureFunc, num_workers, chunksize, 
                                         videoName, storeName, savekwargs, **kwargs):
        return Stream._extract_chunked_to_disc(self, featureFunc, ThreadPool, num_workers, chunksize, 
                                               videoName, storeName, savekwargs, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_mp_to_disc(self, featureFunc, num_workers, chunksize, 
                                   videoName, storeName, savekwargs, **kwargs):
        return Stream._extract_chunked_to_disc(self, featureFunc, Pool, num_workers, chunksize, 
                                               videoName, storeName, savekwargs, **kwargs)
        
       
@dataclass(frozen = True)
class StillStream:
    ### instead of a cap, a StillStream has a single frame!
    ### it displays a single picture for a certain duration
    ### to initialize it: save a single picture frame lossless, and read it as a StillStream
    stillFrm: Union[GreyFrame, BGRFrame, HSVFrame, BoolFrame]
    start_frame: int
    stop_frame: int
    scale_factor: int = 1
    name: str = None ### if none is left, it defaults to cFrm for bgr, gFrm for grey, hFrm for hsv, bFrm for bool
    features: StreamFeatures = StreamFeatures({})
       
    def __post_init__(self):
        ### CHECK FEATURES
        mySF = self.features
        if isinstance(self.features, StreamFeature) or isinstance(self.features, ConstFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        ### conform StreamFeatures to start and stop frame
        # mySF = mySF.conform(list(range(self.start_frame, self.stop_frame + 1))) ### capture contains last frame!
        object.__setattr__(self, 'features', mySF)
    
    @staticmethod
    def from_SF(mySF, stillFrm, name = None,
                cap_color_space = 'bgr', target_color_space = 'bgr',
                scale_factor = 1):
        ### figure out name
        if name is None:
            name = stillFrm.name
        ### color conversion
        imgArr = stillFrm.array
        stillFrm = color_converter(imgArr, cap_color_space, target_color_space)
        ### figure out start and stop
        start, stop = mySF.get_frames()[0], mySF.get_frames()[-1]
        return StillStream(stillFrm, start, stop, 
                           scale_factor = scale_factor,
                           features = mySF, name = name)
    
    @staticmethod
    def from_file(framePath, name = None, 
                  cap_color_space = 'bgr', target_color_space = 'bgr',
                  scale_factor = 1):
        imgArr = np.load(framePath)
        stillFrm = color_converter(imgArr, cap_color_space, target_color_space)
        stillFrm = stillFrm.rename(name)
        
        fileName = os.path.basename(framePath)
        
        ### find start and end frame
        fileStump = fileName.split('.')[0]
        _, start, _, stop = fileStump.split('_')
        start, stop = int(start), int(stop)
        
        return StillStream(stillFrm, start, stop, 
                           scale_factor = scale_factor,
                           name = name)
    
    @staticmethod
    def from_store(fileName, featureName, videoName, storeName, 
                   cap_color_space = 'bgr', target_color_space = 'bgr',
                   scale_factor = 1):
        featDict = get_folder_struct('features_stored')
        featPath = featDict['main']
        ### find video
        videoPath = os.path.join(featPath, videoName, storeName, featureName, fileName)
        return StillStream.from_file(videoPath, name = featureName,
                                     cap_color_space = cap_color_space, target_color_space = target_color_space,
                                     scale_factor = scale_factor)
    
    @staticmethod
    def argstrings_to_args(funcArgs, myFrm, myDict):
        return Stream.argstrings_to_funcs(funcArgs, myFrm, myDict)
    
    def __getitem__(self, frmNum):
        assert self.start_frame <= frmNum <= self.stop_frame, f'Frame requested has to be between {self.start_frame} and {self.stop_frame}'
        if self.scale_factor != 1:
            outFrm = self.stillFrm.rescale(self.scale_factor)
        else:
            outFrm = self.stillFrm
        return outFrm
    
    def length(self):
        return self.stop_frame - self.start_frame + 1
    
    def dims(self):
        width  = self.stillFrm.cols
        height = self.stillFrm.rows
        return width, height
    
    def size(self, unit = 'GB'):
        ### size of raw video in stream, if decapped
        unitMap = {'B': 1, 'KB': 1000, 'MB': 1000 ** 2, 'GB': 1000 ** 3}
        
        try:
            mult = self.stillFrm.shape[2]
        except IndexError:
            mult = 1
            
        width, height = self.dims()
        length = self.length()
        return round(length * width * height * mult / unitMap[unit], 2)
    
    def conform(self, start_frame, stop_frame):
        return Stream.conform(self, start_frame, stop_frame)
        
    def recolor(self, new_color):
        oldFrm = self.stillFrm
        old_color = oldFrm.__class__.get_color()
        newFrm = color_converter(oldFrm.array, old_color, new_color)
        
        updDict = {'stillFrm': newFrm}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return StillStream(**newDict)
        
    def attach(self, mySF: StreamFeatures):
        return Stream.attach(self, mySF)
    
    def decap(self):
        framesType = framesTypeMap[self.stillFrm.__class__.get_color()]
        
        ### get length
        num_elems = self.stop_frame - self.start_frame + 1
        
        ### get array
        stillArr = self.stillFrm.array
        ### expand array by repeating
        expArr = np.array([stillArr for i in range(num_elems)])
        ### turn into proper MultiFrame type
        expFrms = framesType(expArr, name = self.name)
        
        return DecapStream(expFrms, self.start_frame, self.stop_frame, self.features)
    
    def generator(self, ms_between_frames = 25, frame_counter = False):
        
        frmNum = self.start_frame
    
        while self.start_frame <= frmNum <= self.stop_frame:
            ### get the frame
            myFrm = self.stillFrm
            
            ### get the features
            myDict = self.features.maybe_get(frmNum)
            
            myFrm = yield frmNum, myFrm, myDict # take in frame, possibly altered, to show 
            
            if frame_counter:
                myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
            
            if myFrm is not None:
                still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                if not still_showing:
                    break
            
            frmNum += 1
            
        else:
            cv.destroyAllWindows()
    
    def show(self, ms_between_frames = 25, frame_counter = False):
        return Stream.show(self,  ms_between_frames = ms_between_frames, frame_counter = frame_counter)
    
    def extract(self, featureFunc, **kwargs):
        return Stream.extract(self, featureFunc, **kwargs)
    
    
@dataclass(frozen = True)
class MulStream:
    ### The MulStream holds many streams in parallel:
        # all contained streams differ (different cap, features, color space) 
        # EXCEPT for their start and end frames, which are the same.
    streams: List[Stream]
    
    def __post_init__(self):
        ### NOTE: The features, start and end frame taken for the entire MulStream are stored in the first Stream!
        # leadStm = self[0]
        # object.__setattr__(self, 'leadStm', leadStm)
        ### TODO: Something smarter. 
            # canonical features -> union of all.
            # ... start_frame -> check if they all agree (mindful of caching), take any 
            # ... stop_frame -> same 
            # ... cache_size -> check all CachedStreams, take largest
        
        canonical_features = self.streams[0].features
        canonical_start_frame = self.streams[0].start_frame ### this fucks things if there is a difference 
        canonical_stop_frame = self.streams[0].stop_frame   ### between starts and stops, but the generator will pick this up.

        object.__setattr__(self, 'features', canonical_features)
        object.__setattr__(self, 'start_frame', canonical_start_frame)
        object.__setattr__(self, 'stop_frame', canonical_stop_frame)
        
        try:
            canonical_cache_size = self.streams[0].cache_size
        except:
            pass
        else:
            object.__setattr__(self, 'cache_size', canonical_cache_size)
        
    @staticmethod
    def from_streams(*streams):
        ### TODO: maybe include some integrity checks (same start and end, etc.)
        return MulStream(streams)
    
    def __iter__(self):
        return self.streams.__iter__()
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return [myStm[item] for myStm in self]
        else:
            return MulStream([myStm[item] for myStm in self])
    
    ### So that a MulStream may be extracted from with caching
    def _set_mut_features(self, features_needed = None):
        for myStm in self:
            try: ### this will work for all cached streams
                myStm._set_mut_features(features_needed = features_needed)
            except AttributeError:
                pass
    
    def _update_mut_features(self, frmNum, updDict):
        for myStm in self:
            try: ### this will work for all cached streams
                myStm._update_mut_features(frmNum, updDict)
            except AttributeError:
                pass
    
    def _delete_mut_features(self):
        for myStm in self:
            try: ### this will work for all cached streams
                myStm._delete_mut_features()
            except AttributeError:
                pass
        
    def length(self):
        return self.stop_frame - self.start_frame + 1
        
    def dims(self):
        return self.streams[0].dims()
    
    def size(self, unit = 'GB'):
        ### size of raw video in stream, if decapped
        return sum([myStm.size(unit = unit) for myStm in self])
    
    def maxchunk(self, critical_value = 4, safety_margin = 0.1): 
        ### critical_value = 4 -> larger than 4 GB, we can not pickle!
        ### safety_margin = 0.1 -> want to be 10% below critical value
        size_per_unit = self.size(unit = 'GB') / self.length()
        admissible_units = int((1 - safety_margin) * critical_value / size_per_unit)
        return admissible_units
    
    def attach(self, mySF):
        fsStm, othStms = self.streams[0], self.streams[1:]
        return MulStream([fsStm.attach(mySF), *othStms])
        
    def conform(self, start_frame, stop_frame):
        return MulStream([myStm.conform(start_frame, stop_frame) for myStm in self])
        
    def decap(self):
        return MulStream([myStm.decap() for myStm in self])
    
    @staticmethod
    def argstrings_to_args(funcArgs, frameList, featDict_or_Cache):
        ### get arguments to plug into feature extraction function
        # differentiate between function arguments that need a stream and those that need a feature
        is_from_stream = lambda myStr: myStr.endswith('Mask') or myStr.endswith('Frm')
        is_a_cache = lambda myStr: myStr.endswith('Cache')
        
        args = []
        for argStr in funcArgs:
            if is_from_stream(argStr):
                matching_frames = [myFrm for myFrm in frameList if myFrm.name == argStr]
                assert len(matching_frames) == 1, f'There has to be one (and only one) frame corresponding to {argStr}, instead found {len(matching_frames)}!'
                argObj = matching_frames[0]
            elif is_a_cache(argStr):
                if argStr == 'featCache':
                    argObj = featDict_or_Cache ### typecheck here? or just let it fail?
                else:
                    matching_caches = [myCache for myCache in frameList if myCache.name == argStr]
                    assert len(matching_caches) == 1, f'There has to be one (and only one) cache corresponding to {argStr}, instead found {len(matching_caches)}!'
                    argObj = matching_caches[0]
            else:
                try:
                    argObj = featDict_or_Cache[argStr]
                except KeyError as e:
                    raise e
            args.append(argObj)
        return args
    
    def chunk_generator(self, chunksize = 100):
        chunksize = min(chunksize, self.maxchunk())
        
        ### generate decapped chunks of the stream, one by one lazily
        chunk_start = self.start_frame
        chunk_stop = min(chunk_start + chunksize - 1, self.stop_frame) 
        ### since the last frame is included, a chunk of size 10 with chunk_start 5 ends at 14 = 5 + 10 - 1
        
        while not chunk_stop == self.stop_frame:
            yield self.conform(chunk_start, chunk_stop).decap()
            chunk_start = chunk_stop + 1
            chunk_stop = min(chunk_start + chunksize - 1, self.stop_frame)
        else:
            yield self.conform(chunk_start, chunk_stop).decap() ### yield one last time
            
    def generator(self, ms_between_frames = 25, frame_counter = False):
        def continue_generators(generator_states):
            if all(generator_states):
                return True
            elif all([not state for state in generator_states]):
                return False
            else: ### somethign is messed up
                raise Exception('Some generators are running, while others are not...')
            
        def step_generators(genList):
            numList = []
            yieldList_frames = []
            yieldList_feats  = []
            generator_states = []
            for gen in genList:
                try:
                    frmNum, myFrm, myFeat = next(gen)    ### the yielded feature dictionaries are ignored! 
                                                         ### all that matters is the first feature (see post_init).
                except StopIteration:
                    generator_states.append(False)
                else:
                    generator_states.append(True)
                    numList.append(frmNum)
                    yieldList_frames.append(myFrm)
                    yieldList_feats.append(myFeat)
                   
            return numList, yieldList_frames, yieldList_feats, generator_states
        
        ### initialize the generators
        genList = [myStm.generator() for myStm in self]
        
        ### prime the generator and get the generator states
        numList, yieldList_frames, yieldList_feats, generator_states = step_generators(genList)
        
        while continue_generators(generator_states):
            ### figure out unique frame number
            assert len(set(numList)) == 1, f'Different frame numbers scheduled in parallel for {self}: {numList}'
            frmNum = numList[0]
            
            ### yield previous result (guaranteed to be there, because continue generators is true!)
            yield frmNum, yieldList_frames, yieldList_feats[0] ### first stream is canonical!
                                                                ## this could be a featDict or a Cache of featDicts.
            
            ### get the new results (or a stop iteration)
            numList, yieldList_frames, yieldList_feats, generator_states = step_generators(genList)
            
        
    def extract(self, featureFunc, **kwargs):
        canonicalType = self.streams[0].__class__ ### if the first object is a Stream, we use normal extraction
                                            # if the first object is a CachedStream, we use cached extraction
                                            # (with mutable features, etc.)
        return canonicalType.extract(self, featureFunc, **kwargs)
        
    def extract_to_disc(self, featureFunc, videoName, storeName, savekwargs, **kwargs):
        outSF = self.extract(featureFunc, **kwargs)
        outSF.save(videoName, storeName, **savekwargs)
    
    @extract_mp_timer
    def extract_chunked_threaded(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, ThreadPool, num_workers, chunksize, _split = False, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_mp(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, Pool, num_workers, chunksize, _split = False, **kwargs)
        
    @extract_mp_timer
    def extract_chunked_threaded_to_disc(self, featureFunc, num_workers, chunksize, 
                                         videoName, storeName, savekwargs, **kwargs):
        return Stream._extract_chunked_to_disc(self, featureFunc, ThreadPool, num_workers, chunksize, 
                                               videoName, storeName, savekwargs, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_mp_to_disc(self, featureFunc, num_workers, chunksize, 
                                   videoName, storeName, savekwargs, **kwargs):
        return Stream._extract_chunked_to_disc(self, featureFunc, Pool, num_workers, chunksize,  
                                               videoName, storeName, savekwargs, **kwargs)
        
    
@dataclass(frozen = True)
class StreamList:
    ### The StreamList holds many streams in series:
        # all contained streams are the same (same cap, features, color_space etc) 
        # EXCEPT for their start and end frames, which differ.
    streams: List[Stream]
    
    @staticmethod
    def from_SFList(sfList, cap, name = None,
                    cap_color_space = 'bgr', 
                    target_color_space = 'bgr',
                    scale_factor = 1,
                    warp_dims = None,
                    cache_size = None, force_fill_cache = False):
        if isinstance(cap, cv.VideoCapture):
            if cache_size is None:
                return StreamList([Stream.from_SF(mySF, cap, name = name,
                                                  cap_color_space = cap_color_space, 
                                                  target_color_space = target_color_space,
                                                  scale_factor = scale_factor,
                                                  warp_dims = warp_dims) for mySF in sfList])
            else:
                return StreamList([CachedStream.from_SF(mySF, cap, name = name,
                                                        cache_size = cache_size, force_fill_cache = force_fill_cache,
                                                        cap_color_space = cap_color_space, 
                                                        target_color_space = target_color_space,
                                                        scale_factor = scale_factor,
                                                        warp_dims = warp_dims) for mySF in sfList])
        elif isinstance(cap, Frame): ### in this case, we have a list of StillStreams!
                return StreamList([StillStream.from_SF(mySF, cap, name = name,
                                                       cap_color_space = cap_color_space, 
                                                       target_color_space = target_color_space) for mySF in sfList])
    
    @staticmethod
    def from_store(featureName, videoName, storeName, 
                   cache_size = None, force_fill_cache = False,
                   cap_color_space = 'bgr', 
                   target_color_space = 'bgr',
                   scale_factor = 1):
        featDict = get_folder_struct('features_stored')
        featPath = featDict['main']
        storePath = os.path.join(featPath, videoName, storeName, featureName)
        
        ### find and sort files to read
        files = os.listdir(storePath)
        sortFunc = lambda myStr: int(myStr.split('_')[1]) ### sort by first frame
        sorted_files = sorted(files, key = sortFunc)
        
        ### understand file ending
        suffixes = [file.split('.')[1] for file in sorted_files]
        assert len(set(suffixes)) == 1, f'Mixed file types found in {storePath}'
        suffix = suffixes[0]
        
        ### figure out arguments for reading
        params = {'name': featureName,
                  'cap_color_space': cap_color_space,
                  'target_color_space': target_color_space,
                  'scale_factor': scale_factor}
        
        if suffix == 'npy':
            readType = StillStream
        elif cache_size is not None:
            params['cache_size'] = cache_size
            params['force_fill_cache'] = force_fill_cache
            readType = CachedStream
        else:
            readType = Stream
            
        
        outList = []
        for videoFile in sorted_files:
            videoPath = os.path.join(storePath, videoFile)
            outList.append(readType.from_file(videoPath, **params))
        return StreamList(outList)
    
    @staticmethod
    def combine(*streamLists):
        ### With this method, we can combine many streamLists into a single streamList of MulStreams
        outList = []
        for streams in zip(*streamLists):
            mulStm = MulStream.from_streams(*streams)
            outList.append(mulStm)
        return StreamList(outList)
    
    def __repr__(self):
        repStm = self[0]
        if isinstance(repStm, Stream):
            stmType = 'Stream'
            nameStr = repStm.name
        elif isinstance(repStm, MulStream):
            stmType = 'MulStream'
            nameStr = ', '.join([subStm.name for subStm in repStm])
        elif isinstance(repStm, StillStream):
            stmType = 'StillStream'
            nameStr = repStm.name
        return f'StreamList({self.length()} {stmType}s (named {nameStr}) from {self[0].start_frame} to {self[-1].stop_frame})'
    
    def __iter__(self):
        return self.streams.__iter__()
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.streams[item]
        else:
            return StreamList(self.streams[item])
        
    def length(self):
        return len(self.streams)
    
    def recolor(self, new_target_color_space):
        return StreamList([myStm.recolor(new_target_color_space) for myStm in self])
    
    def rescale(self, new_scale_factor):
        return StreamList([myStm.rescale(new_scale_factor) for myStm in self])
    
    def attach(self, sfList):
        assert self.length() == sfList.length(), f'Trying to attach SFList of length {sfList.length()} to StreamList of length {self.length()}'
        return StreamList([myStm.attach(mySF) for myStm, mySF in zip(self, sfList)])
    
    # def chunk_generator(self, chunksize = None): 
    #     ### chunks are simply the elements!
    #     ### UPDATE: If the chunks get too large, they can not be pickled anymore! In this case,
    #     ### we get the weird 'OS ERROR: parameter is incorrect'
    #     for myStm in self:
    #         sub_generator = myStm.chunk_generator(chunksize = chunksize)
    #         for chunkStm in sub_generator:
    #             yield chunkStm
    
    def chunk_generator(self, chunksize = 100): 
        ### chunks are simply the elements!
        CRITICAL_VALUE = 4 ### beyond 4 GB, we can not pickle!
        for myStm in self:
            if myStm.size(unit = 'GB') >= CRITICAL_VALUE:
                sub_generator = myStm.chunk_generator(chunksize = chunksize)
                for chunkStm in sub_generator:
                    yield chunkStm
            else:
                yield myStm.decap()
    
    def extract(self, featureFunc, **kwargs):
        outList = []
        for myStm in self:
            mySF = myStm.extract(featureFunc, **kwargs)
            outList.append(mySF)
        return SFList(outList)
        
    @extract_mp_timer
    def extract_chunked_threaded(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, ThreadPool, num_workers, chunksize, _split = True, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_mp(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, Pool, num_workers, chunksize, _split = True, **kwargs)
        
    @extract_mp_timer
    def extract_chunked_threaded_to_disc(self, featureFunc, num_workers, chunksize, 
                                         videoName, storeName, savekwargs, **kwargs):
        return Stream._extract_chunked_to_disc(self, featureFunc, ThreadPool, num_workers, chunksize, 
                                               videoName, storeName, savekwargs, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_mp_to_disc(self, featureFunc, num_workers, chunksize, 
                                   videoName, storeName, savekwargs, **kwargs):
        return Stream._extract_chunked_to_disc(self, featureFunc, Pool, num_workers, chunksize,  
                                               videoName, storeName, savekwargs, **kwargs)
        
        
@dataclass(frozen = True)
class DecapStream:
    ### here, the Stream is dissociated from its videoCapture, which is turned into a MultiFrames object.
    frames: Union[GreyFrames, BGRFrames, HSVFrames, BoolFrames] # GreyFrames | BGRFrames | HSVFrames | BoolFrames
    start_frame: int
    stop_frame: int
    features: StreamFeatures = StreamFeatures({})
    name: str = None

    def __getitem__(self, item):
        assert self.start_frame <= item <= self.stop_frame, f'Frame requested has to be between {self.start_frame} and {self.stop_frame}'
        outFrm = self.frames[item - self.start_frame]
        
        for feat in self.features.get_features():
            outFrm = outFrm.put(self.features[item][feat])
        
        return outFrm
    
    @staticmethod
    def argstrings_to_args(funcArgs, myFrm, myDict):
        return Stream.argstrings_to_args(funcArgs, myFrm, myDict)
    
    def length(self):
        return self.stop_frame - self.start_frame + 1
    
    def dims(self):
        height, width = self.frames.shape()
        return width, height
    
    def size(self, unit = 'GB'):
        return self.frames.size(unit = unit)
        
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
    
        for idx, myFrm in enumerate(self.frames):
            frmNum = self.start_frame + idx
            
            myDict = self.features.maybe_get(frmNum)
            
            myFrm = yield frmNum, myFrm, myDict # take in frame, possibly altered, to show 

            if frame_counter:
                myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
            
            if myFrm is not None:
                still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                if not still_showing:
                    break
        
        else:
            cv.destroyAllWindows()
           
    def show(self, ms_between_frames = 30, frame_counter = False):
        return Stream.show(self, ms_between_frames = ms_between_frames, frame_counter = frame_counter)
           
    def extract(self, featureFunc, **kwargs):
        return Stream.extract(self, featureFunc, **kwargs)
    
    def extract_to_disc(self, featureFunc, videoName, storeName, savekwargs, **kwargs):
        return Stream.extract_to_disc(self, featureFunc, videoName, storeName, savekwargs, **kwargs)
    
    def extract_mp(self, featureFunc, num_workers, **kwargs):
        return Stream.extract_mp(self, featureFunc, num_workers, **kwargs)
    
    
@dataclass(frozen = True)
class NCDecapStream:
    ### Non-Consecutive Decapped Stream -> a collection of frames which do not have to be consecutive
    frames: Union[GreyFrames, BGRFrames, HSVFrames]
    frmNums: List
    features: StreamFeatures = StreamFeatures({})
    name: str = None
    
    def __post_init__(self):
        ### CHECK FEATURES
        mySF = self.features
        if isinstance(self.features, StreamFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        object.__setattr__(self, 'features', mySF)
    
    def __repr__(self):
        if self.is_empty():
            return 'NCDecapStream({})'
        else:
            return f'NCDecapStream(frames = {self.frames}, frmNums = {self.frmNums[0]} ... {self.frmNums[-1]}, features = {self.features})'
    
    def length(self):
        return self.frames.length()
    
    def dims(self):
        height, width = self.frames.shape()
        return width, height
    
    def size(self, unit = 'GB'):
        return self.frames.size(unit = unit)
    
    def is_empty(self):
        return self.length() == 0
    
    @staticmethod
    def from_bool(boolSF, cap, 
                  cap_color_space = 'bgr',
                  target_color_space = 'bgr',
                  scale_factor = 1,
                  name = None):
        
        framesType = framesTypeMap[target_color_space]
        filterSF = boolSF.filter(boolSF) ## keep only true entries
        frmNums = list(filterSF.keys())
        frames = framesType.from_list(cap, frmNums, 
                                      cap_color_space = cap_color_space,
                                      scale_factor = scale_factor,
                                      name = name) ### TODO: speed up the get_framelist function by threading.
        return NCDecapStream(frames, frmNums, filterSF, name = name)
    
    @staticmethod
    def argstrings_to_args(funcArgs, myFrm, myDict):
        return Stream.argstrings_to_args(funcArgs, myFrm, myDict)

    def __getitem__(self, item):
        assert item in self.frmNums, f'Frame requested has to be in self.frmNums!'
        outFrm = self.frames[self.frmNums.index(item)]
        
        for feat in self.features.get_features():
            outFrm = outFrm.put(self.features[item][feat])
            
        return outFrm
    
    def attach(self, mySF: StreamFeatures):
        return Stream.attach(self, mySF)
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
    
        for frmNum, myFrm in zip(self.frmNums, self.frames):
            myDict = self.features.maybe_get(frmNum)
            
            myFrm = yield frmNum, myFrm, myDict # take in frame, possibly altered, to show 
            if frame_counter:
                myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
            
            if myFrm is not None:
                still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                if not still_showing:
                    break
        
        else:
            cv.destroyAllWindows()
    
    def show(self, ms_between_frames = 30, frame_counter = False):
        return Stream.show(self, ms_between_frames = ms_between_frames, frame_counter = frame_counter)
           
    def extract(self, featureFunc, **kwargs):
        return Stream.extract(self, featureFunc, **kwargs)
    
    def extract_mp(self, featureFunc, num_workers, **kwargs):
        return Stream.extract_mp(self, featureFunc, num_workers, **kwargs)
    
@dataclass(frozen = True) ### despite a frozen dataclass, within the functions below, the features attribute is treated as mutable, so we can update it and use it dynamically for subsequent iterations
class CachedStream(Stream):
    cache_size: int = 1 ### number of frames before and after the current frame (e.g.: cache size 1 -> three frames in cache.)
    force_fill_cache: bool = False ### whether the initial cache may be repeated instances of the first frame,
                                    # or whether we just go before and after stop frames.
                                    # THE ONLY WAY THIS INTERACTS WITH ANYTHING IS VIA TRUNCATE_FRAME_NUMBER!!!
    
    def __post_init__(self):
        
        ### Define a helper function which is only relevant if we want to force fill the cache.
        def truncate_frame_number(frmNum):
            if not self.force_fill_cache:
                return frmNum
            else:
                if frmNum < self.start_frame:
                    return self.start_frame
                elif frmNum > self.stop_frame:
                    return self.stop_frame
                else:
                    return frmNum
        
        ### Here, important to check that the cache size is actually available within the stream!
        total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))  ### total number of frames
        fs_avail = self.truncation_anchor
        ls_avail = self.truncation_anchor + total_frames - 1 ### available frames, in absolute terms (relative to the master cap). Note that if there are 2 frames, the indices are 0 and 1 -> hence, we subtract 1 from num_frames
        
        start_cap = truncate_frame_number(self.start_frame - self.cache_size)
        stop_cap = truncate_frame_number(self.stop_frame + self.cache_size)
        
        if not fs_avail <= start_cap <= ls_avail:
            print(f'Starting frame {self.start_frame} not available in {self.cap} from {fs_avail} to {ls_avail} and cache_size {self.cache_size}')
            earliest_start = fs_avail + self.cache_size
            object.__setattr__(self, 'start_frame', earliest_start)
            print(f'Reset starting frame to {earliest_start}.')
        
        if not fs_avail <= stop_cap <= ls_avail:
            print(f'Stop frame {self.stop_frame} not available in {self.cap} from {fs_avail} to {ls_avail} and cache_size {self.cache_size}')
            latest_stop = ls_avail - self.cache_size
            object.__setattr__(self, 'stop_frame', latest_stop)
            print(f'Reset stopping frame to {latest_stop}.')
        
        # if not fs_avail <= start_cap <= ls_avail:
        #     raise Exception(f'Starting frame {start_cap} not available in {self.cap} from {fs_avail} to {ls_avail}')
        # if not fs_avail <= stop_cap <= ls_avail:
        #     raise Exception(f'Stop frame {stop_cap} not available in {self.cap} from {fs_avail} to {ls_avail}')
            
        ### CHECK NAME
        if self.name is None:
            object.__setattr__(self, 'name', nameMap[self.target_color_space])
        
        ### CHECK FEATURES
        mySF = self.features
        if isinstance(self.features, StreamFeature) or isinstance(self.features, ConstFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        ### conform StreamFeatures to start and stop frame
        # mySF = mySF.conform(list(range(self.start_frame, self.stop_frame + 1))) ### capture contains last frame!
        object.__setattr__(self, 'features', mySF)
            
    @staticmethod
    def from_SF(mySF, cap, name = None,
                cache_size = 1, force_fill_cache = False,
                cap_color_space = 'bgr', target_color_space = 'bgr',
                scale_factor = 1, warp_dims = None):
        start, stop = mySF.get_frames()[0], mySF.get_frames()[-1]
        return CachedStream(cap, start, stop, cache_size = cache_size, force_fill_cache = force_fill_cache,
                            truncation_anchor = 0,
                            cap_color_space = cap_color_space, target_color_space = target_color_space, 
                            scale_factor = scale_factor, warp_dims = warp_dims,
                            features = mySF, name = name)

    @staticmethod
    def from_stream(someStm, cache_size = 1, force_fill_cache = False):
        return CachedStream(someStm.cap, someStm.start_frame, someStm.stop_frame,
                            truncation_anchor = someStm.truncation_anchor, 
                            cap_color_space = someStm.cap_color_space,
                            target_color_space = someStm.target_color_space, 
                            scale_factor = someStm.scale_factor,
                            name = someStm.name, features = someStm.features, 
                            cache_size = cache_size, force_fill_cache = force_fill_cache)
    
    @staticmethod
    def from_file(filePath, name = None,
                  cache_size = 1, force_fill_cache = False,
                  cap_color_space = 'bgr', target_color_space = 'bgr',
                  scale_factor = 1):
        cap = cv.VideoCapture(filePath)
        
        fileName = os.path.basename(filePath)
        
        ### find start and end frame
        fileStump = fileName.split('.')[0]
        _, start, _, stop = fileStump.split('_')
        start, stop = int(start), int(stop)
        
        return CachedStream(cap, start, stop, cache_size = cache_size, force_fill_cache = force_fill_cache,
                            truncation_anchor = start, ### NOTE: truncation anchor is the first frame given in file name! 
                            cap_color_space = cap_color_space, target_color_space = target_color_space, 
                            scale_factor = scale_factor,
                            name = name)
    
    @staticmethod
    def from_store(fileName, featureName, videoName, storeName, 
                   cache_size = 1, force_fill_cache = False,
                   cap_color_space = 'bgr', target_color_space = 'bgr',
                   scale_factor = 1):
        featDict = get_folder_struct('features_stored')
        featPath = featDict['main']
        ### find video
        videoPath = os.path.join(featPath, videoName, storeName, featureName, fileName)
        return CachedStream.from_file(videoPath, cache_size = cache_size, force_fill_cache = force_fill_cache,
                                      name = featureName,
                                      cap_color_space = cap_color_space, target_color_space = target_color_space,
                                      scale_factor = scale_factor)
        
    
    def _set_mut_features(self, features_needed = None):
        ### set the mutable features attribute
        ### used internally to dynamically update features and use them subsequently in extraction.
        _mut_features = self.features
        ### to allow this, they have to ...
        # ... conform to the beginning and end of the Stream (including cache).
        frameList = list(range(self.start_frame - self.cache_size, self.stop_frame + self.cache_size + 1))
        _mut_features = _mut_features.conform(frameList)
        ## convert constant features to stream features, to which we can write!
        if isinstance(_mut_features, ConstFeatures):
            _mut_features = _mut_features.to_SF()
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
    
    def length(self, including_cache = True):
        if including_cache:
            return Stream.length(self) + self.cache_size * 2
        else:
            return Stream.length(self)
    
    @staticmethod
    def argstrings_to_args(funcArgs, frameCache, featCache):
        ### get arguments to plug into feature extraction function
        
        # differentiate between function arguments that need a stream and those that need a feature
        is_from_stream = lambda myStr: myStr.endswith('Mask') or myStr.endswith('Frm')
        
        args = []
        for argStr in funcArgs:
            if argStr == frameCache.name:
                args.append(frameCache)
            elif argStr == featCache.name:
                args.append(featCache)
        return args
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
        ### as opposed to the simple stream, the cached stream has to generate two caches which are constantly updated:
            # 1) the frameCache -> current frame at center, with future and past frames around
            # 2) the featCache -> current feature at center, with future and past features around   
        framecacheName = self.name.replace('Frm', '').replace('Mask', '') + 'Cache'
        frameCache = Cache(self.cache_size, [], name = framecacheName)
        # featCache = Cache(self.cache_size, [])
        ### Update: The featCache is not updated, but rebuilt, so that changes to past features can be used by featFunc.
        featIndices = frameCache.get_indices()
        
        ### Define a helper function which is only relevant if we want to force fill the cache.
        def truncate_frame_number(frmNum):
            if not self.force_fill_cache:
                return frmNum
            else:
                if frmNum < self.start_frame:
                    return self.start_frame
                elif frmNum > self.stop_frame:
                    return self.stop_frame
                else:
                    return frmNum
        
        ### mutable features always have to exist here!
        assert hasattr(self, '_mut_features'), 'To create a generator for a CachedStream, mutable features always have to exist!'
    
        cap_start = self.start_frame - self.cache_size
        cap_stop = self.stop_frame + self.cache_size
    
        frmNum = cap_start ### think of this as a counter in the background -> goes from cap_start to cap_stop, increment of 1
        getNum = truncate_frame_number(frmNum) ### if force_fill_cache -> this truncates frames before start and after stop to start and stop resp.
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, getNum - self.truncation_anchor)       

        while self.cap.isOpened() and (cap_start <= frmNum <= cap_stop):
            cntNum = frmNum - self.cache_size ### the central frame in the current iteration

            ret, frame = self.cap.read()
            
            myFrm = color_converter(frame, self.cap_color_space, self.target_color_space)
            myFrm = myFrm.rename(self.name)
            
            ### warp if necessary
            if self.warp_dims is not None:
                width, height = self.warp_dims
                H = self.features['H'][getNum]
                H_inv = np.linalg.inv(H)
                myFrm = myFrm.warp(H_inv, cols = width, rows = height)
                
            ### rescale if necessary
            if self.scale_factor != 1:
                myFrm = myFrm.rescale(self.scale_factor)
                
            ### feed/build the caches
            frameCache = frameCache.add(myFrm)
            features = [self._mut_features.maybe_get(truncate_frame_number(idx + cntNum)) for idx in featIndices]
            featCache = Cache.from_list(features, name = 'featCache')
            
            ### check if we are supposed to yield -> only once cache is full!
            if frameCache.is_full():
                myFrm = yield cntNum, frameCache, featCache # take in frame, possibly altered, to show
            else:
                myFrm = None
            
            ### using the frame that comes back in, do something.
            ### if None is sent to generator, do nothing.
            if myFrm is not None:
                if frame_counter:
                    myFrm = myFrm.put(str(getNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
                
                if myFrm is not None:
                    still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                    if not still_showing:
                        break
                    
            if frmNum != getNum: ### this will only occur if we force_fill_cache, at the beginning and end.
                                 ### none of this is intuitive... there must be a better way.
                self.cap.set(cv.CAP_PROP_POS_FRAMES, getNum - self.truncation_anchor)      
                
            frmNum += 1  ### the counter is incremented
            getNum = truncate_frame_number(frmNum) ### the latest frame being read
            
        else:
            cv.destroyAllWindows()
        
    def chunk_generator(self, chunksize = 100):
        return Stream.chunk_generator(self, chunksize = chunksize)
        
    def extract(self, featureFunc, **kwargs):
        
        ### Establish function arguments and return shape
        funcArgs = get_positional_arguments(featureFunc)
        funcVals = get_return_names(featureFunc)
        # assert funcArgs == ['frameCache', 'featCache'], f'Function passed in to CachedStream.extract should have *["frameCache", "featCache"] as positional arguments, not *{funcArgs}'
        
        self._set_mut_features(features_needed = funcVals)
        
        ### Initialize Generator
        gen = self.generator()
        frmNum, frameCache, featCache = next(gen) ### prime the generator
        
        streamDict = {}
        while True:
            ### get arguments to plug into feature extraction function
            args = self.__class__.argstrings_to_args(funcArgs, frameCache, featCache) ### 
                
            ### assert that all arguments that we expect (funcArgs) we have been able to extract (args).
            assert len(args) == len(funcArgs), f'Required arguments ({funcArgs}) have to match supplied arguments ({args})'
            
            
            ### call extraction function
            funcOut = featureFunc(*args, **kwargs)
            
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
       
    def extract_chunked_threaded(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream.extract_chunked_threaded(self, featureFunc, num_workers, chunksize, **kwargs)
            
    def extract_chunked_mp(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream.extract_chunked_mp(self, featureFunc, num_workers, chunksize, **kwargs)
                 
    def decache(self):
        return Stream(**{k: v for k, v in self.__dict__.items() if k in Stream.__annotations__})
    
    def decap(self):
        ### Update: When decapping a cached stream, the start_frame and stop_frame refer
        # to the first and last yielded frames. Hence, the Frames object needs to include 
        # a cache to the left and right of start_frame and stop_frame resp.
        # in principle, this is how the CachedStream behaves: 
        # Its generator begins cycling before start frame, and ends after stop_frame. 
        
        ### get starting and ending frames including cache
        # (the existence of which is guaranteed by post_init!)
        start_cap = self.start_frame - self.cache_size - self.truncation_anchor
        stop_cap = self.stop_frame + self.cache_size - self.truncation_anchor
            
        framesType = framesTypeMap[self.target_color_space]
        
        myFrms = framesType.from_video(self.cap, start_cap, stop_cap,
                                       cap_color_space = self.cap_color_space,
                                       scale_factor = self.scale_factor,
                                       name = self.name)
        ### warp if necessary
        if self.warp_dims is not None:
            width, height = self.warp_dims
            HInvArr = self.features['H'].apply(np.linalg.inv).to_array()
            myFrms = myFrms.warp(HInvArr, cols = width, rows = height)
        
        return DecapCachedStream(myFrms, self.start_frame, self.stop_frame, self.features, 
                                 cache_size = self.cache_size, name = self.name)

    def show(self):
        self.decache().show()
    
    
@dataclass(frozen = True)
class DecapCachedStream(DecapStream):
    cache_size: int = 1
    
    _set_mut_features = CachedStream._set_mut_features
    _update_mut_features = CachedStream._update_mut_features        
    _delete_mut_features = CachedStream._delete_mut_features
        
    @staticmethod
    def argstrings_to_args(funcArgs, myFrm, myDict):
        return CachedStream.argstrings_to_args(funcArgs, myFrm, myDict)
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
        ### as opposed to the simple stream, the cached stream has to generate two caches which are constantly updated:
            # 1) the frameCache -> current frame at center, with future and past frames around
            # 2) the featCache -> current feature at center, with future and past features around    
        framecacheName = self.name.replace('Frm', '').replace('Mask', '') + 'Cache'
        frameCache = Cache(self.cache_size, [], name = framecacheName)
        # featCache = Cache(self.cache_size, [])
        ### Update: The featCache is not updated, but rebuilt, so that changes to past features can be used by featFunc.
        featIndices = frameCache.get_indices()
    
        ### mutable features always have to exist here!
        assert hasattr(self, '_mut_features'), 'To create a generator for a CachedStream, mutable features always have to exist!'
    
        for idx, myFrm in enumerate(self.frames):
            frmNum = self.start_frame + idx - self.cache_size ### the frame we are currently getting!
            cntNum = frmNum - self.cache_size ### the central frame in the current iteration

            ### feed/build the caches
            frameCache = frameCache.add(myFrm)
            features = [self._mut_features.maybe_get(idx + cntNum) for idx in featIndices]
            featCache = Cache.from_list(features, name = 'featCache')
            
            ### check if we are supposed to yield -> only once cache is full!
            if frameCache.is_full():
                myFrm = yield cntNum, frameCache, featCache # take in frame, possibly altered, to show
            else:
                myFrm = None
            
            ### using the frame that comes back in, do something.
            ### if None is sent to generator, do nothing.
            if myFrm is not None:
                if frame_counter:
                    myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
                
                if myFrm is not None:
                    still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                    if not still_showing:
                        break
            
        else:
            cv.destroyAllWindows()
            
    def extract(self, featureFunc, **kwargs):
        return CachedStream.extract(self, featureFunc, **kwargs)
        
@dataclass(frozen = True)
class RandStream(Stream):
    number_of_frames: int = 100
    
    def length(self):
        return self.number_of_frames
    
    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
            
        frmNum = randrange(self.start_frame, self.stop_frame)
        frames_yielded = 0
    
        while self.cap.isOpened() and frames_yielded < self.number_of_frames:
            
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
            ret, frame = self.cap.read()
            
            ### Get the frame in the correct color type
            myFrm = color_converter(frame, self.cap_color_space, self.target_color_space)
            myFrm = myFrm.rename(self.name)
            
            ### rescale if necessary
            if self.scale_factor != 1:
                myFrm = myFrm.rescale(self.scale_factor)
                
            ### Get the feature dictionary
            myDict = self.features.maybe_get(frmNum)
            ### if frmNum not found, we get {feat: None for feat in self.get_features()}
            
            myFrm = yield frmNum, myFrm, myDict # take in frame, possibly altered, to show 

            if frame_counter:
                myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
            
            if myFrm is not None:
                still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                if not still_showing:
                    break
                
            frmNum = randrange(self.start_frame, self.stop_frame)
            frames_yielded += 1
        
        else:
            cv.destroyAllWindows()
    
@dataclass(frozen = True)
class NCStream:
    cap: cv.VideoCapture
    frmNums: List
    truncation_anchor: int = 0 ### instead of a truncation flag. this anchor frame allows comparison between truncated streams and non-truncated streams by specifying the first frame of truncation.
    cap_color_space: str = 'bgr'
    target_color_space: str = 'bgr'
    scale_factor: float = 1 ### ability to shrink the image relative to the cap dimensions
    name: str = None ### if none is left, it defaults to cFrm for bgr, gFrm for grey, hFrm for hsv, bFrm for bool
    features: StreamFeatures = StreamFeatures({})
        
    def __repr__(self):
        return f'NCStream(cap = {self.cap}, frmNums = {self.frmNums[0]} ... {self.frmNums[-1]} (length = {self.length()}), features = {self.features})'
    
    
    @staticmethod
    def from_SF(mySF, cap, name = None,
                cap_color_space = 'bgr', target_color_space = 'bgr',
                scale_factor = 1):
        frmNums = mySF.get_frames()
        return NCStream(cap, frmNums, 
                      truncation_anchor = 0,
                      cap_color_space = cap_color_space, target_color_space = target_color_space, 
                      scale_factor = scale_factor,
                      features = mySF, name = name)
    
    def __post_init__(self):
        ### CHECK NAME
        if self.name is None:
            object.__setattr__(self, 'name', nameMap[self.target_color_space])
        
        ### CHECK FEATURES
        mySF = self.features
        if isinstance(self.features, StreamFeature):
            ### transform StreamFeature into StreamFeatures
            mySF = mySF.unsqueeze()
        ### conform StreamFeatures to start and stop frame
        # mySF = mySF.conform(list(range(self.start_frame, self.stop_frame + 1))) ### capture contains last frame!
        object.__setattr__(self, 'features', mySF)
    
    def __getitem__(self, frmNum):
        ### TODO: allow for item to be a slice also! (this would yield a multiframes)
        assert frmNum in self.frmNums, f'Frame requested has to be in frmNums: [{self.frmNums[0]} ... {self.frmNums[-1]}]'
        outType = frameTypeMap[self.target_color_space]
        
        get_num = frmNum - self.truncation_anchor
        
        outFrm = outType.from_video(self.cap, get_num, 
                                    cap_color_space = self.cap_color_space, 
                                    scale_factor = self.scale_factor,
                                    name = self.name)
        
        for feat in self.features.get_features():
            outFrm = outFrm.put(self.features[frmNum][feat])
            
        return outFrm
        
    @staticmethod
    def argstrings_to_args(funcArgs, myFrm, myDict):
        return Stream.argstrings_to_args(funcArgs, myFrm, myDict)
    
    def length(self):
        return len(self.frmNums)
    
    def dims(self):
        return Stream.dims(self)
    
    def size(self, unit = 'GB'):
        return Stream.size(self, unit = unit)
    
    def conform(self, new_frmNums):
        updDict = {'frmNums': new_frmNums}
        newDict = get_updated_attribute_dictionary(self, updDict)
        return self.__class__(**newDict)
    
    def recolor(self, new_target_color_space):
        return Stream.recolor(self, new_target_color_space)
    
    def rescale(self, new_target_color_space):
        return Stream.rescale(self, new_target_color_space)
    
    def attach(self, mySF):
        return Stream.attach(self, mySF)

    def decap(self):
        framesType = framesTypeMap[self.target_color_space]
        frames = framesType.from_list(self.cap, self.frmNums, 
                                      cap_color_space = self.cap_color_space,
                                      scale_factor = self.scale_factor,
                                      name = self.name) ### TODO: speed up the get_framelist function by threading.
        return NCDecapStream(frames, self.frmNums, self.features, name = self.name)

    def generator(self, ms_between_frames = 25, frame_counter = False): ### cv_funcs.video_stream
    
        for frmNum in self.frmNums:
            
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
            ret, frame = self.cap.read()
            
            ### Get the frame in the correct color type
            myFrm = color_converter(frame, self.cap_color_space, self.target_color_space)
            myFrm = myFrm.rename(self.name)
            
            ### rescale if necessary
            if self.scale_factor != 1:
                myFrm = myFrm.rescale(self.scale_factor)
            
            ### Get the feature dictionary
            myDict = self.features.maybe_get(frmNum)
            ### if frmNum not found, we get {feat: None for feat in self.get_features()}
            
            myFrm = yield frmNum, myFrm, myDict # take in frame, possibly altered, to show 
            
            if frame_counter:
                myFrm = myFrm.put(str(frmNum), where = (40, 40), size = 2, color = (0, 255, 0), thickness = 3)
            
            if myFrm is not None:
                still_showing = myFrm.show_continuous(ms_between_frames = ms_between_frames)
                if not still_showing:
                    break
            
            frmNum += 1
        
        else:
            cv.destroyAllWindows()

    def maxchunk(self, critical_value = 4, safety_margin = 0.1): 
        ### critical_value = 4 -> larger than 4 GB, we can not pickle!
        ### safety_margin = 0.1 -> want to be 10% below critical value
        size_per_unit = self.size(unit = 'GB') / self.length()
        admissible_units = int((1 - safety_margin) * critical_value / size_per_unit)
        return admissible_units
    
    def chunk_generator(self, chunksize = 100):
        chunksize = min(chunksize, self.maxchunk())

        ### generate decapped chunks of the stream, one by one lazily
        slice_start, slice_stop = 0, chunksize
        while slice_stop < self.length():
            chunkNums = self.frmNums[slice_start: slice_stop]
            yield self.conform(chunkNums).decap()
            slice_start, slice_stop = slice_stop, slice_stop + chunksize
        else:
            chunkNums = self.frmNums[slice_start: slice_stop]
            yield self.conform(chunkNums).decap()

    def show(self, ms_between_frames = 25, frame_counter = False):
        Stream.show(self, ms_between_frames = ms_between_frames, frame_counter = frame_counter)
          
    def extract(self, featureFunc, **kwargs): ### cv_funcs.get_stream_features
        return Stream.extract(self, featureFunc, **kwargs)
    
    @extract_mp_timer
    def extract_threaded(self, featureFunc, num_workers, **kwargs):
        return Stream._extract_pool(self, featureFunc, ThreadPool, num_workers, **kwargs)
    
    @extract_mp_timer
    def extract_mp(self, featureFunc, num_workers, **kwargs):
        return Stream._extract_pool(self, featureFunc, Pool, num_workers, **kwargs)
    
    @extract_mp_timer
    def extract_chunked_threaded(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, ThreadPool, num_workers, chunksize, _split = False, **kwargs)
            
    @extract_mp_timer
    def extract_chunked_mp(self, featureFunc, num_workers, chunksize, **kwargs):
        return Stream._extract_chunked(self, featureFunc, Pool, num_workers, chunksize, _split = False, **kwargs)
       

    
    