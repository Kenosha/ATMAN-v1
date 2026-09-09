# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:17:12 2021

Utility Functions

@author: gianl
"""

import os
import time
import numpy as np
import cv2 as cv
import ATMANpy.funcs.util_funcs as atu

def show(frame, name = 'frame', num_seconds = None):
    cv.imshow(name, frame)
    keypress = cv.waitKey(0 if num_seconds is None else round(num_seconds * 1000))
    cv.destroyAllWindows()  
    return keypress

def show_continuous(frame, name = 'frame', ms_between_frames = 30):
    cv.imshow('frame', frame)
    wait_return = cv.waitKey(ms_between_frames) ### wait, and check for key input.
    if wait_return & 0xFF == ord('q'): ### if key input corresponds to character q, break.
        cv.destroyAllWindows()  
        return False ### needed for signalling a possible break condition
    return True

def get_frame(cap, frame_num):
    
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)  ### total number of frames
    
    if frame_num < 0 or frame_num > total_frames:
        print(f'No such frame, choose number between 0 and {total_frames}')
        return None
    
    # Avoid decoding the GOP again when this is already the next frame.
    if cap.get(cv.CAP_PROP_POS_FRAMES) != frame_num:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    return frame

def get_frames(cap, start_frame, stop_frame):
    
    outFrames = []
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    while cap.isOpened() and (start_frame <= frmNum <= stop_frame):
        ret, frame = cap.read()
        
        outFrames.append(frame)
        
        frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        
    return np.array(outFrames)

def get_framelist(cap, frameList):
    
    outFrames = []
    
    for frmNum in frameList:
        if cap.get(cv.CAP_PROP_POS_FRAMES) != frmNum:
            cap.set(cv.CAP_PROP_POS_FRAMES, frmNum)
        ret, frame = cap.read()
        
        # if color_space == 'hsv':
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # if color_space == 'grey':
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
        outFrames.append(frame)
        
    return np.array(outFrames)

def get_width_height(cap):
    return cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)

def join_video_captures(cap_feature, videoName, storeName):
    ### Due to chunking, it can happen that contiguous video captures are stored in different files
    ### when extracting straight to disc. Hence, we might want to join these.
    ### TODO: Why are the files that are created so large, compared to the sum of their constituents?
    print('Joining contiguous, but separate files...')
    
    featDict = atu.get_folder_struct('features_stored')
    featPath = featDict['main']
    storePath = os.path.join(featPath, videoName, storeName, cap_feature)
    
    ### find files to join
    files = os.listdir(storePath)
    
    ### understand dimensions, fps, codec.
    fs_cap = cv.VideoCapture(os.path.join(storePath, files[0]))
    width = int(fs_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(fs_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = width, height
    fps = int(fs_cap.get(cv.CAP_PROP_FPS))
    codec = int(fs_cap.get(cv.CAP_PROP_FOURCC))
    fs_cap.release()
    del fs_cap
    # codecStr = chr(codec&0xff) + chr((codec>>8)&0xff) + chr((codec>>16)&0xff) + chr((codec>>24)&0xff)
    # codec = cv.VideoWriter_fourcc(*codecStr)
    
    ### sort files to join
    sortFunc = lambda myStr: int(myStr.split('_')[1]) ### sort by first frame
    sorted_files = sorted(files, key = sortFunc)
    
    ### understand file ending
    filenames, suffixes = [file.split('.')[0] for file in sorted_files], [file.split('.')[1] for file in sorted_files]
    assert len(set(suffixes)) == 1, f'Mixed file types found in {storePath}'
    suffix = suffixes[0]
    
    ### define helpers ###
    startFunc = lambda fileStr: int(fileStr.split('_')[1])
    endFunc = lambda fileStr: int(fileStr.split('_')[3])
    
    ### group files if contiguous
    groupList = []
    current_group = []
    
    for file in filenames:
        if current_group == []:
            current_group.append(file)
        else:
            last_file = current_group[-1]
            if endFunc(last_file) + 1 == startFunc(file):
                current_group.append(file)
            elif endFunc(last_file) + 1 < startFunc(file):
                groupList.append(current_group)
                current_group = [file]
            else:
                raise Exception(f'Previous file ending: {endFunc(last_file)}; current file beginning: {startFunc(file)}')
            
    groupList.append(current_group)

    ### merge grouped files
    for group in groupList:
        if len(group) > 1:
            group_start, group_end = startFunc(group[0]), endFunc(group[-1])
            groupName = f'from_{group_start}_to_{group_end}'
            groupPath = os.path.join(storePath, groupName + f'.{suffix}')
            writer = cv.VideoWriter(groupPath, codec, fps, size)
            for fileName in group:
                filePath = os.path.join(storePath, fileName + f'.{suffix}')
                
                cap = cv.VideoCapture(filePath)
                while cap.isOpened():
                    r, frame = cap.read()
                    if not r:
                        break
                    writer.write(frame)

                cap.release()
                del cap
                os.remove(filePath) 
                
            writer.release()
    
    
def show_video(cap, start_frame = None, stop_frame = None, ms_between_frames = None,
               frame_counter = False):
       
    if ms_between_frames is None:
        ms_between_frames = 25
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    
    while cap.isOpened() and (start_frame <= frame_num <= stop_frame):
        ret, frame = cap.read()
        
        if frame_counter:
            cv.putText(frame, str(frame_num), org = (40, 40), 
                       fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 2,
                       color = (0, 255, 0), thickness = 3)
        
        if ret:
            cv.imshow('frame', frame)
            frame_num += 1
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                break
        else:
            break
        
    # cap.release()
    cv.destroyAllWindows()
    
def video_stream(cap, start_frame = None, stop_frame = None, ms_between_frames = None):
    if ms_between_frames is None:
        ms_between_frames = 25
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    while cap.isOpened() and (start_frame <= frmNum <= stop_frame):
        ret, frame = cap.read()
        
        frame = yield frmNum, frame # take in frame, possibly altered, to show
        # print(frame)
        
        if frame is not None:
            cv.imshow('frame', frame)
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                cv.destroyAllWindows()  
                yield None
                
        frmNum = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    
    else:
        cv.destroyAllWindows()
                
       

def random_or_next_stream(cap, ms_between_frames = None, verbose = False):
    from random import randrange
    
    if ms_between_frames is None:
        ms_between_frames = 25
    total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        
    doFlag = 'random' ### this flag signals to the generator what to do.
                        # it takes three values. 'random', 'next', 'stop'
                   
    while doFlag != 'stop':
        
        if doFlag == 'random':
            
            randFrame = randrange(0, total_frames - 1)
            cap.set(cv.CAP_PROP_POS_FRAMES, randFrame)
            ret, frame = cap.read()
            
            doFlag, frame = yield randFrame, frame 
        
        elif doFlag == 'next':
            
            frmNum = cap.get(cv.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            
            doFlag, frame = yield frmNum, frame
        
        if verbose:
            cv.imshow('frame', frame)
            if cv.waitKey(ms_between_frames) & 0xFF == ord('q'):
                cv.destroyAllWindows()  


def get_stream_features_from_generator(gen,
                                       existing_features = None,
                                       cache_length = None,
                                       init_feat_func = None, init_frame_func = None,
                                       cached_line_func = None,
                                       court_line_func = None, court_feature_func = None,
                                       draw_func = None,
                                       features_of_interest = None,
                                       verbose = True):
    
     ### CACHING ###
    if cache_length is None:
        cache_length = 1 ### cache holds only the current frame.
    curr_frame_idx = cache_length // 2 ### the offset with which features are entered.
                                        ## note that due to caching of past and future frames, 
                                        ## the video generator is ahead by curr_frame_idx frames, 
                                        ## so we adjust frmNum (which comes from the video generator) 
                                        ## by that amount curr_frame_idx.
    
    ### init outputList
    streamFeatures = {}
    
    #### init feature dictionary
    featDict = None
    
    #### init cache ###
    frmCache = []
    
    from ATMANpy.types.Frames import BGRFrame
    from copy import deepcopy
    
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        ### prepare and feed the cache ###
        # cache frames, so we can frame difference 
        # without having to re-generate the frames from the video capture
        latestFrame = BGRFrame(rawArr)    
        frmCache.append(latestFrame)
        if len(frmCache) < cache_length:
            frmNum, rawArr = gen.send(None)
            continue
        if len(frmCache) > cache_length:
            print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
            break
        
        ### prepare frame objects of interest
        frmDict = init_frame_func(frmCache)
        frmNum_on_display = frmNum - curr_frame_idx
        
        ### initialize the feature dictionary with possible stored features
        if existing_features is not None:
            try:
                stored_featDict = existing_features[frmNum_on_display]
            except KeyError:
                stored_featDict = None
        else:
            stored_featDict = None
            
        featDict = init_feat_func(current_featDict = featDict, 
                                  stored_featDict = stored_featDict)
        
        ### Get the features of interest on the frame
        if not court_line_func is None:
            featDict = court_line_func(frmDict, featDict)
        
        if not court_feature_func is None:
            featDict = court_feature_func(frmDict, featDict)
            
            
        ### extract out stream features of interest
        if features_of_interest is None:
            streamFeatures[frmNum_on_display] = deepcopy(featDict)
        else:
            saveDict = {k: v for k, v in featDict.items() if k in features_of_interest}
            streamFeatures[frmNum_on_display] = deepcopy(saveDict)
    
        if verbose:
            ### Send Frame Back to Generator
            sendFrm = draw_func(frmDict, featDict, frameNum = frmNum_on_display)
            sendArr = sendFrm.array  
        else:
            sendArr = None
            
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
        
        ### drop oldest element from cache ### 
        frmCache = frmCache[1:] # drop oldest item
        
    else:
        cv.destroyAllWindows()

    return streamFeatures
    
def get_stream_features(cap, start_frame, stop_frame, 
                        existing_features = None,
                        cache_length = None,
                        init_feat_func = None, init_frame_func = None,
                        cached_line_func = None,
                        court_line_func = None, court_feature_func = None,
                        draw_func = None,
                        features_of_interest = None,
                        verbose = True):
    
    if isinstance(cap, str):
        cap = atu.get_video(cap) ### this is needed for multiprocessing
            # video captures can not be pickled, therefore, we create it inside of the function
     
    from copy import deepcopy
    from ATMANpy.types.Frames import BGRFrame
    
    ### CACHING ###
    if cache_length is None:
        cache_length = 1 ### cache holds only the current frame.
    curr_frame_idx = cache_length // 2 ### the offset with which features are entered.
                                        ## note that due to caching of past and future frames, 
                                        ## the video generator is ahead by curr_frame_idx frames, 
                                        ## so we adjust frmNum (which comes from the video generator) 
                                        ## by that amount curr_frame_idx.
    
    if start_frame is not None:
        start_frame_with_caching = start_frame - curr_frame_idx
    else:
        start_frame_with_caching = 0
        
    if stop_frame is not None:
        stop_frame_with_caching = stop_frame + curr_frame_idx
    else:
        stop_frame_with_caching = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    ### init outputList
    streamFeatures = {}
    
    #### init feature dictionary
    featDict = None
    
    #### init cache ###
    frmCache = []
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, 
                       start_frame = start_frame_with_caching, 
                       stop_frame = stop_frame_with_caching)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        ### prepare and feed the cache ###
        # cache frames, so we can frame difference 
        # without having to re-generate the frames from the video capture
        latestFrame = BGRFrame(rawArr)    
        frmCache.append(latestFrame)
        if len(frmCache) < cache_length:
            frmNum, rawArr = gen.send(None)
            continue
        if len(frmCache) > cache_length:
            print(f'Cache FUBAR, {len(frmCache)} items') ### this happens if the cache drop at the end fails
            break
        
        ### prepare frame objects of interest
        frmDict = init_frame_func(frmCache)
        frmNum_on_display = frmNum - curr_frame_idx
        
        ### initialize the feature dictionary with possible stored features
        if existing_features is not None:
            try:
                stored_featDict = existing_features[frmNum_on_display]
            except KeyError:
                stored_featDict = None
        else:
            stored_featDict = None
            
        featDict = init_feat_func(current_featDict = featDict, 
                                  stored_featDict = stored_featDict)
        
        ### Get the features of interest on the frame
        if not court_line_func is None:
            featDict = court_line_func(frmDict, featDict)
        
        if not court_feature_func is None:
            featDict = court_feature_func(frmDict, featDict)
            
            
        ### extract out stream features of interest
        if features_of_interest is None:
            streamFeatures[frmNum_on_display] = deepcopy(featDict)
        else:
            saveDict = {k: v for k, v in featDict.items() if k in features_of_interest}
            streamFeatures[frmNum_on_display] = deepcopy(saveDict)
    
        if verbose:
            ### Send Frame Back to Generator
            sendFrm = draw_func(frmDict, featDict, frameNum = frmNum_on_display)
            sendArr = sendFrm.array  
        else:
            sendArr = None
            
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
        
        ### drop oldest element from cache ### 
        frmCache = frmCache[1:] # drop oldest item
        
    else:
        cv.destroyAllWindows()

    return streamFeatures
    

def replay(cap, stream_features, 
           start_frame = None, stop_frame = None,
           ms_between_frames = None,
           drawFunc = None):
    
    if start_frame is None:
        start_frame = min(stream_features.keys())
        
    if stop_frame is None:
        stop_frame = max(stream_features.keys())
    
    from ATMANpy.types.Frames import BGRFrame
    
    ### init video stream ###
    from ATMANpy.funcs.cv_funcs import video_stream
    gen = video_stream(cap, 
                       start_frame = start_frame, 
                       stop_frame = stop_frame,
                       ms_between_frames = ms_between_frames)
    frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
    ######
    
    while True:
        
        cFrm = BGRFrame(rawArr)
        
        if not stream_features is None:
            featDict = stream_features[frmNum]
        else:
            featDict = {}
        
        ### Send Frame Back to Generator
        if drawFunc is not None:
            sendFrm = drawFunc({'cFrm': cFrm}, featDict, frameNum = frmNum)
            sendArr = sendFrm.array  
        else:
            sendArr = None
        
        try:
            frmNum, rawArr = gen.send(sendArr)
        except:
            cv.destroyAllWindows()
            break 
            
        
if __name__ == '__main__':
    array = np.random.randint(256, size = (400, 400, 3)).astype('uint8')
    keypress = show(array)
    print(keypress)
        
        