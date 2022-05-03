# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:28:50 2022

Replication of Lai and Hsieh (2009)

@author: gianl
"""


import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point, SafePoint
from ATMANpy.Lines import Line, Lines, CourtLines, Region
from ATMANpy.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame

# image operation
import cv2 as cv
import numpy as np


##### VIDEO FILE READ IN ####
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
####################



######## GET SCHEMA ########
from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)


###### DEFINE TRACK PROCEDURE ########
from ATMANpy.high_level_funcs import find_homography_225_only, update_homography
from ATMANpy.high_level_funcs import find_courtlines, find_netlines

def track_courtlines(gFrm, prev_H, prev_config, schemaLines = schemaLines):
    if prev_H is None:
        res = find_homography_225_only(gFrm)
    else:
        # print('Courtlines found, updating homography...')
        res = update_homography(gFrm, prev_H, prev_config)
        
    if not res is None:
        H, config = res
        courtLines = find_courtlines(H, schemaLines)
        netLines = find_netlines(gFrm, courtLines)       
    
        # outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        # outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        courtLines, netLines = None, None
        H, config = None, None
        
    return courtLines, netLines, H, config


#########################################

##### Read-In Burned-In Background ########
import os
burnDict = atu.get_folder_struct('burn_in')

try:
    readPath = burnDict[VIDEO_ID]
except KeyError:
    print('Burn-In Data not found.')
    
medFrm = HSVFrame.from_csv(readPath, name = 'median')
stdFrm = HSVFrame.from_csv(readPath, name = 'std')

medFrm_bgr = BGRFrame.from_csv(readPath, name = 'median')
stdFrm_bgr = BGRFrame.from_csv(readPath, name = 'std')

##### GET THE BURNED IN MASK ########
NORMALIZED_DEVIATION_THRESHOLD = 4
BURN_IN_HSV_CHANNEL_WEIGHTS = (1, 50, 1) #interested only in hue and value

from ATMANpy.player_funcs import get_burn_in_mask

from functools import partial
match_specific_burn_in_func = partial(get_burn_in_mask, medFrm, stdFrm) ### these are always the same!


########## GET REGIONS OF INTEREST ##########

WIDTH_FACTOR = 2 ### how much of the doubles alley width does a player maximally next to court
DEPTH_FACTOR = 0.2 ### how much of side length does a player maximally stand behind baseline
ADD_DEPTH_FACTOR = 1.25 ### top side depth adjustment - player figure is tall, so looks to be outside of ROI.
# we could adjust this with the camera matrix, but let's do something eyeballed here, for the moment.

from ATMANpy.player_funcs import get_initial_ROIs

ROI_lower, ROI_upper = get_initial_ROIs(schemaLines,
                                         width_factor = WIDTH_FACTOR, 
                                         depth_factor = DEPTH_FACTOR, 
                                         add_depth_factor_top = ADD_DEPTH_FACTOR)
schemaFrm.to_BGR().put(ROI_upper, color = (0, 0, 255), thickness = 2).put(ROI_lower, color = (0, 255, 0), thickness = 2).show()

############


def find_avg_and_std(frame, projROI, courtLines, which = 'Top'):
    
    roi_mask_in, roi_mask_out = get_initial_ROI_masks(frame, projROI, courtLines, which = which)
    
    white_mask = frame.to_grey() >= WHITE_PIXEL_THRESHOLD
    avg_in, std_in = frame.mean_std(mask = roi_mask_in & white_mask)
    avg_out, std_out = frame.mean_std(mask = roi_mask_out & white_mask)
    
    return avg_in, std_in, avg_out, std_out    

####### FIND PLAYER #########


#### GET ARBITRARY FRAME ####
FRAME_NUM = 11005
gFrm = GreyFrame.from_video(cap, FRAME_NUM)
gFrm.show()

cFrm = BGRFrame.from_video(cap, FRAME_NUM)
cFrm.show()

############################# 

H, config = None, None

courtLines, netLines, H, config = track_courtlines(gFrm, H, config)

hFrm = cFrm.to_HSV()

projROI_lower = ROI_lower.transform('apply_homography', H).transform('round_points')
projROI_upper = ROI_upper.transform('apply_homography', H).transform('round_points')
cFrm.put(projROI_upper, color = (0, 0, 255), thickness = 2).put(projROI_lower, color = (0, 255, 0), thickness = 2).show()

burninMask = match_specific_burn_in_func(hFrm, H,
                                normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                verbose = True)


from ATMANpy.high_level_funcs import COURT_LINE_WIDTH, WHITE_PIXEL_THRESHOLD
from ATMANpy.player_funcs import find_player_with_burn_in, get_initial_ROI_masks

verbose = True
COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1



initialROI = projROI_upper
which = 'Top'

avg_in, std_in, avg_out, std_out = find_avg_and_std(hFrm, initialROI, courtLines, which = which)

topRegion = find_player_with_burn_in(hFrm, burninMask, courtLines, netLines,
                                     avg_in, avg_out, std_in, std_out, initialROI, playerROI = None,
                                     line_width = COURT_LINE_WIDTH, 
                                     court_dilation_factor = COURT_DILATION,
                                     net_dilation_factor = NET_DILATION,
                                     verbose = verbose)

initialROI = projROI_lower
which = 'Bottom'

avg_in, std_in, avg_out, std_out = find_avg_and_std(hFrm, initialROI, courtLines, which = which)

botRegion = find_player_with_burn_in(hFrm, burninMask, courtLines, netLines,
                                     avg_in, avg_out, std_in, std_out, initialROI, playerROI = None,
                                     line_width = COURT_LINE_WIDTH, 
                                     court_dilation_factor = COURT_DILATION,
                                     net_dilation_factor = NET_DILATION,
                                     verbose = verbose)

cFrm.put(topRegion).put(botRegion).show()


from ATMANpy.player_funcs import find_player_only_burn_in


initialROI = projROI_upper
roiMask = hFrm.within(initialROI)
topRegion = find_player_only_burn_in(hFrm, burninMask, roiMask, verbose = False)


initialROI = projROI_lower
roiMask = hFrm.within(initialROI)
botRegion = find_player_only_burn_in(hFrm, burninMask, roiMask, verbose = False)

cFrm.put(topRegion).put(botRegion).show()

#############################
##### SNIPPETS


# # ### Background Differencing ###
# # NORMALIZED_DEVIATION_THRESHOLD = 2
# # channel_weights = (1, 1, 1) #interested only in hue and value
# # # hFrm = cFrm.to_HSV()

# # normdiffFrm = abs(cFrm - wrpFrm_bgr) / stdwrpFrm_bgr.clip_lo(1)

# # normdiffMask = normdiffFrm < [c * NORMALIZED_DEVIATION_THRESHOLD for c in channel_weights]
# # normdiffMask = normdiffMask.all(axes = 2)
# # cFrm.mask(~normdiffMask).show()


# ABSOLUTE_DEVIATION_THRESHOLD = 15

# absdiffFrm = abs(hFrm - wrpFrm)

# absdiffMask = absdiffFrm < [80, 1000, 80]
# absdiffMask = absdiffMask.all(axes = 2)
# cFrm.mask(~absdiffMask).show()
# ##############

# tightRegion = Region.from_points(Point((650, 650)), Point((750, 750)))
# h_crop = hFrm.crop(tightRegion)
# wrp_crop = wrpFrm.crop(tightRegion)

# ### check berdych's shoes...
# tightRegion = Region.from_points(Point((945, 250)), Point((965, 270)))
# cFrm.put(tightRegion).show()
# h_crop = hFrm.crop(tightRegion)
# wrp_crop = wrpFrm.crop(tightRegion)
# nrm_crop = normdiffFrm.crop(tightRegion)
# nrmMask = normdiffMask.crop(tightRegion)
# abs_crop = absdiffFrm.crop(tightRegion)

# ### check kia logo
# tightRegion = Region.from_points(Point((525, 380)), Point((600, 430)))
# cFrm.put(tightRegion).show()
# h_crop = hFrm.crop(tightRegion)
# wrp_crop = wrpFrm.crop(tightRegion)
# nrm_crop = normdiffFrm.crop(tightRegion)
# nrmMask = normdiffMask.crop(tightRegion)
# abs_crop = absdiffFrm.crop(tightRegion)
#### Find ROI ####

# as opposed to the paper, we'll define the ROI on the court schema, and 
# translate it to projection coordinates later

# courtFrm.show()
