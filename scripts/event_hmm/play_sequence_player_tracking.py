# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:45:04 2022

@author: gianl
"""

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.funcs.util_funcs as atu
from ATMANpy.types.Points import Point, SafePoint
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame
from ATMANpy.types.Contours import Contour

# image operation
import cv2 as cv
import numpy as np

import pandas as pd

import os
from random import randrange

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################


########## READ-IN PASSAGES OF PLAY ##########

evtDict = atu.get_folder_struct('ball_event_hmm')

try:
    readPath = evtDict[VIDEO_ID]
except KeyError:
    print('Data not found.')
    
playseqPath = os.path.join(readPath, 'play_sequences')    
start_stop_timestamp = '2022-04-28-19-09-45'
file = f'play_start_stop_{start_stop_timestamp}.csv'

seqFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)

# frmList = []
# for file in os.listdir(playseqPath):
#     frm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)
#     frmList.append(frm)
# seqFrm = pd.concat(frmList, ignore_index = True)
    
homography_timestamp = '2022-05-03-21-53-31'
file = f'homographies_{homography_timestamp}.csv'

homFrm = pd.read_csv(os.path.join(playseqPath, file), index_col = 0)


###############################################
        


#%% ############# STREAM FUNCTIONS ########

import ATMANpy.funcs.high_level_funcs as hlf
import ATMANpy.funcs.particle_funcs_vectorized as pf ### in the best case, we ONLY import from hlf

def init_featDict():
    # H = None
    # config = None
    # topplayerRegion, botplayerRegion = None, None
    # particles, idxHistories, closeHistories, modelHistories = None, None, None, None    
    
    featDict = {'H': None, 'config': None,
                'courtLines': None, 'netLines': None, 'trackFlag': None,
                'playerBot': None, 'playerTop': None,
                'ballContours': None, 'ballParticles': None, 'ballIdxHistory': None, 
                'ballCloseHistory': None, 'ballModelHistory': None}
    return featDict

### CACHE ######
FRAME_DIFF_LIST = [4, 6, 8]
CACHE_LENGTH = max(FRAME_DIFF_LIST) * 2 + 1 # we have a frame diff of 8 on both sides, so we need 8 * 2 + 1 = 17
CURR_FRAME_IDX = CACHE_LENGTH // 2 ### the middle frame
FRAME_DIFF_INDICES = [CURR_FRAME_IDX - f for f in reversed(FRAME_DIFF_LIST)] + [CURR_FRAME_IDX + f for f in FRAME_DIFF_LIST]
################

def init_frameDict(frmCache, 
                   curr_frame_idx = CURR_FRAME_IDX, 
                   frame_diff_indices = FRAME_DIFF_INDICES):
    
    cFrm = frmCache[curr_frame_idx]
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    diffFrames = [frmCache[idx].to_grey() for idx in frame_diff_indices]
    gFrm_before = frmCache[curr_frame_idx - 1].to_grey()
    
    frameDict = {'cFrm': cFrm, 'gFrm': gFrm, 'hFrm': hFrm, 
                 'diffFrames': diffFrames, 'gFrm_before': gFrm_before}
    
    return frameDict
    
def get_court_lines(frmDict, featDict, cached_lines = None):
    
    gFrm = frmDict['gFrm']
    H = featDict['H']
    config = featDict['config']
    
    if not cached_lines is None:
        courtLines, H, trackFlag = cached_lines
        from ATMANpy.funcs.court_funcs import find_netlines
        netLines = find_netlines(gFrm, courtLines)     
    else:
        try:
            courtLines, netLines, H, config, trackFlag = hlf.track_courtlines(gFrm, H, config,
                                                                              only_225 = False)
        except Exception as e:
            print('Exception in track_courtlines:')
            print(e)
            courtLines, netLines, H, config = None
            trackFlag = 'Not Found'
    
    featDict['courtLines'] = courtLines
    featDict['netLines'] = netLines
    featDict['H'] = H
    featDict['config'] = config
    featDict['trackFlag'] = trackFlag
    
    return featDict

# def get_court_features(frmDict, featDict): ### these two quantities will be used in simulations
    
#     ### handle the case if the court is not found ---> simply reset all features!
#     if featDict['trackFlag'] == 'Not Found':
#         outDict = init_featDict()
#         outDict['trackFlag'] = 'Not Found'
#         return outDict
        
#     hFrm = frmDict['hFrm']
    
#     H = featDict['H']
#     courtLines = featDict['courtLines']
#     netLines = featDict['netLines']
    
#     outDict = featDict.copy() 
    
#     if featDict['trackFlag'] == 'Tracked':
#         roiMask_bot, roiMask_top = featDict['roiMask_bot'], featDict['roiMask_top']
#     else:
#         ### Project Regions of Interest
#         projROI_bot, projROI_top = hlf.schema_specific_roi_projection_func(H)
#         roiMask_bot, roiMask_top = hFrm.within_region(projROI_bot), hFrm.within_region(projROI_top)
#         outDict['roiMask_bot'], outDict['roiMask_top'] = roiMask_bot, roiMask_top

#     ### Top Player, with updated averages
#     topplayerRegion = hlf.find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_top)
#     outDict['playerTop'] = topplayerRegion
#     ### Bottom Player, with updated averages
#     botplayerRegion = hlf.find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_bot)
#     outDict['playerBot'] = botplayerRegion
    
#     return outDict
  

#%% Court features with smart(er) player detection

######## GET SCHEMA ########
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

initial_width_factor = 1
initial_depth_factor = 1
additional_depth_factor = 0.3

from ATMANpy.funcs.player_funcs import get_service_ROIs
initial_ROI_lower, initial_ROI_upper = get_service_ROIs(schemaLines,
                                                         width_factor = initial_width_factor, 
                                                         depth_factor = initial_depth_factor, 
                                                         add_depth_factor_bottom = additional_depth_factor)

def get_court_features(frmDict, featDict): ### these two quantities will be used in simulations
    
    PLAYER_TRACKING_SCALE_FACTOR = 0.2
    BOT_MIN_BOUNDING_ROI_HEIGHT = 40 ### bounding region has to be 40 pixels wide at least...
    BOT_MIN_BOUNDING_ROI_WIDTH  = 100 ## ... and 100 pixels tall
    TOP_MIN_BOUNDING_ROI_HEIGHT = 30 ## reduce values a bit for top because player is farther away.
    TOP_MIN_BOUNDING_ROI_WIDTH  = 80 

    ### handle the case if the court is not found ---> simply reset all features!
    if featDict['trackFlag'] == 'Not Found':
        outDict = init_featDict()
        outDict['trackFlag'] = 'Not Found'
        return outDict
        
    hFrm = frmDict['hFrm']
    
    H = featDict['H']
    courtLines = featDict['courtLines']
    netLines = featDict['netLines']
    
    outDict = featDict.copy() 
    
    topPlayer = featDict['playerTop']
    botPlayer = featDict['playerBot']
    
    if topPlayer is None or botPlayer is None:
        initial_projROI_lower, initial_projROI_upper = hlf.get_projected_ROIs(initial_ROI_lower, initial_ROI_upper, H)
    
    if topPlayer is None:
        roiMask_top = hFrm.within_region(initial_projROI_upper)
    else:
        enclReg = topPlayer.enclosing_region()
        updated_projROI_upper = enclReg.scale(PLAYER_TRACKING_SCALE_FACTOR).enlarge(to_min_height = TOP_MIN_BOUNDING_ROI_HEIGHT, 
                                                                                    to_min_width = TOP_MIN_BOUNDING_ROI_WIDTH)
        roiMask_top = hFrm.within_region(updated_projROI_upper)
        
    if botPlayer is None:
        roiMask_bot = hFrm.within_region(initial_projROI_lower)
    else:
        enclReg = botPlayer.enclosing_region()
        updated_projROI_lower = enclReg.scale(PLAYER_TRACKING_SCALE_FACTOR).enlarge(to_min_height = BOT_MIN_BOUNDING_ROI_HEIGHT, 
                                                                                    to_min_width = BOT_MIN_BOUNDING_ROI_WIDTH)
        roiMask_bot = hFrm.within_region(updated_projROI_lower)
        
    ### Players
    topplayerRegion, botplayerRegion = hlf.find_both_players(hFrm, courtLines, netLines, H, 
                                                             roiMask_top = roiMask_top, roiMask_bot = roiMask_bot,
                                                             remove_shadow = True)
    outDict['playerTop'] = topplayerRegion
    outDict['playerBot'] = botplayerRegion
    
    # ### Players without shadow removal
    # topplayerRegion, botplayerRegion = hlf.find_both_players(hFrm, courtLines, netLines, H, 
    #                                                          roiMask_top = roiMask_top, roiMask_bot = roiMask_bot,
    #                                                          remove_shadow = False)
    # outDict['playerTop_noshadow'] = topplayerRegion
    # outDict['playerBot_noshadow'] = botplayerRegion
    
    outDict['roiMask_top'] = roiMask_top
    outDict['roiMask_bot'] = roiMask_bot
    
    return outDict




  
    

#%%### DRAW FUNCTIONS #####  
def make_sendframe(frmDict, featDict):
    cFrm = frmDict['cFrm']
        
    if featDict['trackFlag'] == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
    else:
        courtLines = featDict['courtLines']
        playerTop = featDict['playerTop']
        playerBot = featDict['playerBot']
        # playerTop_noshad = featDict['playerTop_noshad']
        # playerBot_noshad = featDict['playerBot_noshad']
        sendFrm = cFrm.put(courtLines, thickness = 2, color = (255, 0, 0))
        sendFrm = sendFrm.put(playerTop, thickness = 2, color = (0, 0, 255))
        sendFrm = sendFrm.put(playerBot, thickness = 2, color = (0, 0, 255))
        # sendFrm = sendFrm.put(playerTop_noshad, thickness = 2, color = (0, 255, 0))
        # sendFrm = sendFrm.put(playerBot_noshad, thickness = 2, color = (0, 255, 0))

    return sendFrm

def null_replay(frmDict, featDict):
    cFrm = frmDict['cFrm']
    return cFrm

#%% STREAM AND EXTRACT BALL FEATURES


from functools import partial
from ATMANpy.funcs.cv_funcs import get_stream_features, replay
from ATMANpy.funcs.high_level_funcs import cached_courtlines 


# features_of_interest = ['ball', 'ballDetected',
#                         'playerTop', 'playerBot']

my_cached_line_func = partial(cached_courtlines, homFrm = homFrm)

my_stream_func = partial(get_stream_features, 
                        cache_length = CACHE_LENGTH,
                        init_feat_func = init_featDict, 
                        init_frame_func = init_frameDict,
                        court_line_func = get_court_lines, 
                        court_feature_func = get_court_features,
                        draw_func = make_sendframe,
                        verbose = True)



#### Simulation 
# video_segments = seqFrm.loc[[9]]
video_segments = seqFrm.head(10)

featList = []

for idx, start_stop_frames in video_segments.iterrows():
    start_frame, stop_frame = start_stop_frames
    stream_features_cached = my_stream_func(cap, start_frame, stop_frame,
                                            cached_line_func = my_cached_line_func, verbose = False)
    featList.append(stream_features_cached)
    # replay(cap, stream_features_cached, drawFunc = make_sendframe)
    
    

for SF in featList:    
    replay(cap, SF, drawFunc = make_sendframe, ms_between_frames = 50)

# for frmNum, featDict in stream_features_cached.items():
#     cFrm = BGRFrame.from_video(cap, frmNum)

#     roi_lower = featDict['roiMask_bot']
#     cFrm.mask(roi_lower).show()
    
    
# for frmNum, featDict in stream_features_cached.items():
#     cFrm = BGRFrame.from_video(cap, frmNum)

#     roi_lower = featDict['roiMask_bot']
#     cFrm.mask(roi_lower).show()


#%% Shadow Removal

# myFeat = featList[9]

# myFeat.keys()
# myFrame = 16151

# featDict = myFeat[myFrame]
# playerBot = featDict['playerBot']
# H = featDict['H']

# hFrm = HSVFrame.from_video(cap, myFrame)
# imgFrm = hFrm
# verbose = True

# hFrm.put(playerBot).show()


# p1 = Point((1600, 830))
# p2 = Point((1750, 830))
# p3 = Point((1600, 950))
# p4 = Point((1750, 950))
# shadowReg = Region.from_points(p1, p2, p3, p4)

# hFrm.put(playerBot).crop(shadowReg).show()


# ######## READ IN AND PREPARE BURN-IN ######
# burnDict = atu.get_folder_struct('burn_in')

# try:
#     readPath = burnDict[VIDEO_ID]
# except KeyError:
#     print('Burn-In Data not found.')
    
# medFrm = HSVFrame.from_csv(readPath, name = 'median')
# stdFrm = HSVFrame.from_csv(readPath, name = 'std')

# ##### GET THE BURNED IN MASK ########
# normalized_deviation_threshold = 4
# channel_weights = (1, 50, 1) #interested only in hue and value



# # H_inv = np.linalg.inv(H)
# wrpFrm = medFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
# if verbose: wrpFrm.show()

# stdwrpFrm = stdFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
# if verbose: stdwrpFrm.show()

# ### Background Differencing ###
# diffFrm = imgFrm - wrpFrm
# normdiffFrm = diffFrm / stdwrpFrm.clip_lo(1)

# chk = diffFrm.crop(shadowReg)


# ### exclude everything_outside wrpFrms scope. this way, it will not bleed into the frame if we gaussian blur.
# normdiffFrm = normdiffFrm.mask((wrpFrm > 0).any_channel())

# ### Adding a blur stage!
# normdiffFrm = normdiffFrm.gaussian_blur((21, 21))

# backgroundMask = normdiffFrm < [c * normalized_deviation_threshold for c in channel_weights]
# backgroundMask = backgroundMask.all_channels()
# foregroundMask = ~backgroundMask

# ### Adding a closing stage
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
# foregroundMask = foregroundMask.close(kernel)

# if verbose: imgFrm.mask(foregroundMask).show()


# playerRegion = playerBot

# ltBool = (hFrm < wrpFrm)

# (hFrm - wrpFrm) / stdwrpFrm.clip_lo(1)

# hFrm.array.shape
# medFrm.array.shape


# hFrm.to_BGR().show()
        

# chk2 = ltBool.crop(shadowReg)
# hFrm.crop(shadowReg).show()

# hFrm.put(shadowReg).show()
