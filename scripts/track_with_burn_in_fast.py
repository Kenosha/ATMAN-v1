import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point, SafePoint
from ATMANpy.Lines import Line, Lines, CourtLines, Region
from ATMANpy.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame

# image operation
import cv2 as cv
import numpy as np

### VIDEO ######
VIDEO_ID = 'pqxAmUWIoKc'
cap = atu.get_video(VIDEO_ID)
################

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
        outFlag = 'Found'
    else:
        # print('Courtlines found, updating homography...')
        res = update_homography(gFrm, prev_H, prev_config)
        outFlag = 'Tracked'
        
    if not res is None:
        H, config = res
        courtLines = find_courtlines(H, schemaLines)
        netLines = find_netlines(gFrm, courtLines)       
    
        # outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        # outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        outFlag = 'Not Found'
        courtLines, netLines = None, None
        H, config = None, None
        
    return courtLines, netLines, H, config, outFlag

######## READ IN AND PREPARE BURN-IN ######
burnDict = atu.get_folder_struct('burn_in')

try:
    readPath = burnDict[VIDEO_ID]
except KeyError:
    print('Burn-In Data not found.')
    
medFrm = HSVFrame.from_csv(readPath, name = 'median')
stdFrm = HSVFrame.from_csv(readPath, name = 'std')

##### GET THE BURNED IN MASK ########
NORMALIZED_DEVIATION_THRESHOLD = 4
BURN_IN_HSV_CHANNEL_WEIGHTS = (1, 50, 1) #interested only in hue and value

from ATMANpy.player_funcs import get_burn_in_mask
from functools import partial
match_specific_burn_in_func = partial(get_burn_in_mask, medFrm, stdFrm,
                                      normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD,
                                      channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS) ### these are always the same!
#######################

########## GET REGIONS OF INTEREST ON SCHEMA ##########

WIDTH_FACTOR = 2 ### how much of the doubles alley width does a player maximally next to court
DEPTH_FACTOR = 0.2 ### how much of side length does a player maximally stand behind baseline
ADD_DEPTH_FACTOR = 1.25 ### top side depth adjustment - player figure is tall, so looks to be outside of ROI.
# we could adjust this with the camera matrix, but let's do something eyeballed here, for the moment.

from ATMANpy.player_funcs import get_initial_ROIs
ROI_lower, ROI_upper = get_initial_ROIs(schemaLines,
                                         width_factor = WIDTH_FACTOR, 
                                         depth_factor = DEPTH_FACTOR, 
                                         add_depth_factor_top = ADD_DEPTH_FACTOR)

def get_projected_ROIs(ROI_lower, ROI_upper, H):
    projROI_lower = ROI_lower.transform('apply_homography', H).transform('round_points')
    projROI_upper = ROI_upper.transform('apply_homography', H).transform('round_points')
    # cFrm.put(projROI_upper, color = (0, 0, 255), thickness = 2).put(projROI_lower, color = (0, 255, 0), thickness = 2).show()
    return projROI_lower, projROI_upper

schema_specific_roi_projection_func = partial(get_projected_ROIs, ROI_lower, ROI_upper)

# schemaFrm.to_BGR().put(ROI_upper, color = (0, 0, 255), thickness = 2).put(ROI_lower, color = (0, 255, 0), thickness = 2).show()
##############

#### now, we get a frame and an h
verbose = False
COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1

from ATMANpy.high_level_funcs import COURT_LINE_WIDTH, WHITE_PIXEL_THRESHOLD
from ATMANpy.player_funcs import find_player_only_burn_in


def find_player(frame, courtLines, netLines, H, roiMask = None, verbose = False):
    
    ### get the burned in mask for this specific homography
    burn_foreMask = match_specific_burn_in_func(frame, H,
                                    normalized_deviation_threshold = NORMALIZED_DEVIATION_THRESHOLD, 
                                    channel_weights = BURN_IN_HSV_CHANNEL_WEIGHTS, 
                                    verbose = verbose)
    
    playerRegion = find_player_only_burn_in(hFrm, burn_foreMask, roiMask)
    
    return playerRegion


################ ################

START_FRAME = 11000
STOP_FRAME = 12000


H = None
config = None
    
### init video stream ###
from ATMANpy.cv_funcs import video_stream
gen = video_stream(cap, start_frame = START_FRAME, stop_frame = STOP_FRAME)
frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
######

while True:
    cFrm = BGRFrame(rawArr)
    gFrm = cFrm.to_grey()
    hFrm = cFrm.to_HSV()
    
    try:
        courtLines, netLines, H, config, trackFlag = track_courtlines(gFrm, H, config)
    except Exception as e:
        print('Exception in track_courtlines:')
        print(e)
        courtLines = None
    
    if trackFlag == 'Not Found':
        sendFrm = cFrm.put('No Lines!', thickness = 2, color = (0, 0, 255))
        

    elif trackFlag == 'Found':
        
        ### Project Regions of Interest
        projROI_lower, projROI_upper = schema_specific_roi_projection_func(H)
        roiMask_lower, roiMask_upper = hFrm.within(projROI_lower), hFrm.within(projROI_upper)

        ### Top Player, with updated averages
        topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
        
        ### Bottom Player, with updated averages
        botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
        
        sendFrm = cFrm.put(topplayerRegion, thickness = 2, color = (0, 0, 255)).put(botplayerRegion, thickness = 2, color = (0, 0, 255))
        
    elif trackFlag == 'Tracked':
        ### If the court was tracked, we don't change the projected ROI, 
        #   and we don't change the color averages and standard deviations.
        
        topplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_upper)
        botplayerRegion = find_player(hFrm, courtLines, netLines, H, roiMask = roiMask_lower)
        
        sendFrm = cFrm.put(topplayerRegion, thickness = 2, color = (0, 0, 255)).put(botplayerRegion, thickness = 2, color = (0, 0, 255))
    
    sendArr = sendFrm.array  
    frmNum, rawArr = gen.send(sendArr)
else:
    cv.destroyAllWindows()









