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
    
    playerRegion = find_player_only_burn_in(frame, burn_foreMask, roiMask)
    
    return playerRegion


################ ################


#### GET ARBITRARY FRAME ####
FRAME_NUM = 11019
gFrm = GreyFrame.from_video(cap, FRAME_NUM)
# gFrm.show()

cFrm = BGRFrame.from_video(cap, FRAME_NUM)
# cFrm.show()
############################# 

#### GET PREVIOUS FRAME #####
gFrm_p = GreyFrame.from_video(cap, FRAME_NUM - 1)
# gFrm_p.show()

cFrm_p = BGRFrame.from_video(cap, FRAME_NUM - 1)
# cFrm_p.show()
#############################

#### GET PREVIOUS FRAME #####
gFrm_a = GreyFrame.from_video(cap, FRAME_NUM + 1)
# gFrm_a.show()

cFrm_a = BGRFrame.from_video(cap, FRAME_NUM + 1)
# cFrm_a.show()
#############################

def get_motion_mask(frameNum):
    gFrm = GreyFrame.from_video(cap, frameNum)
    gFrm_bef = GreyFrame.from_video(cap, frameNum - 1)
    gFrm_aft = GreyFrame.from_video(cap, frameNum + 1)

    THRESH = 10
    fstdiff = gFrm - gFrm_bef
    secdiff = gFrm_aft - gFrm_bef
    return (fstdiff > THRESH) & ~(secdiff > THRESH)

frameNum = 11019
cFrm = BGRFrame.from_video(cap, frameNum)
motMask = get_motion_mask(frameNum)

cFrm.mask(motMask).show()

def postprocess_motion_mask(motMask):
    open_kernel = np.ones((4,4),np.uint8)
    maskFrm = motMask.to_grey().open(open_kernel)
    return maskFrm

motMask = postprocess_motion_mask(motMask)

cFrm.mask(motMask).show()

contours, hierarchy = cv.findContours(motMask.array, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# ### how large is a ball?
# ballReg = Region.from_points(Point((860, 567)), Point((870, 577)))
# cFrm.put(ballReg).show()
DIAMETER_OF_BALL = 10
ACCEPTABLE_DEV = 5
acceptable_area = ((DIAMETER_OF_BALL - ACCEPTABLE_DEV) ** 2, (DIAMETER_OF_BALL + ACCEPTABLE_DEV) ** 2)
acceptable_w2h = 3.5

def check_ball_color(object_channels, benchmark_channels, acceptable_dev):
    # TENNIS_BALL_BGR = (79, 255, 223)
    # measured_TENNIS_BALL_BGR = [176, 179, 142] 
    for objCh, ballCh in zip(object_channels, benchmark_channels):
        if not ballCh - acceptable_dev < objCh < ballCh + acceptable_dev:
            return False
    return True

def sieve_ball_candidates(contours, refFrame):
    sievedList = []
    blackArr = refFrame.zero().array
    for cnt in contours:
        cntMask = cv.fillPoly(blackArr.copy(), [cnt], 255)
        
        ### first sieve: size
        # idea: get the area of the object 
        area_lb, area_ub = acceptable_area
        if not area_lb < cntMask.sum() / 255 < area_ub:
            continue
        
        ### second sieve: shape
        # idea: rightmost and leftmost point can not be much farther apart than lowest and highest
        width = cnt[:, :, 0].max() - cnt[:, :, 0].min()
        height = cnt[:, :, 1].max() - cnt[:, :, 1].min()
        if not 1 / acceptable_w2h < width / height < acceptable_w2h:
            continue
        
        ### third sieve: color
        # cFrm.mask(BoolFrame(cntMask)).put(ballReg).show()
        # mean, std = cFrm.mean_std(mask = BoolFrame(cntMask))
        
        # print(mean)
        # if not check_ball_color(mean):
        #     continue
        
        sievedList.append(cnt)
    return sievedList
    


###### SET OF FRAMES #######
candDict = {}
for frameNum in range(11010, 11020, 1):
    cFrm = BGRFrame.from_video(cap, frameNum)
    cFrm.show()
    
    motMask = get_motion_mask(frameNum)
    motMask = postprocess_motion_mask(motMask)

    contours, hierarchy = cv.findContours(motMask.array, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    sieveList = sieve_ball_candidates(contours, motMask)
    
    candDict[frameNum] = sieveList
    
    # if len(sieveList) == 1:
    cntImg = cv.drawContours(cFrm.array.copy(), sieveList, -1, (0, 255, 0), thickness = 2)
    BGRFrame(cntImg).show()
    
    
######## SINGLE FRAME #########
frameNum = 11019

cFrm = BGRFrame.from_video(cap, frameNum)
hFrm = HSVFrame.from_video(cap, frameNum)
gFrm = GreyFrame.from_video(cap, frameNum)

motMask = get_motion_mask(frameNum)
motMask = postprocess_motion_mask(motMask)

hFrm.mask(motMask).show()

ball = contours[6]
cntMask = cv.fillPoly(gFrm.zero().array.copy(), [ball], 255)
cFrm.mask(BoolFrame(cntMask)).show()

cntImg = cv.drawContours(cFrm.array.copy(), [ball], -1, (0, 255, 0), thickness = 2)
BGRFrame(cntImg).show()
    
def plotHist(imgFrm, maskFrm, color = ('b', 'g', 'r')):
    from matplotlib import pyplot as plt
    
    splitArr = cv.split(imgFrm.array)
    for i, col in enumerate(color):
        histr = cv.calcHist(splitArr,[i], maskFrm, [256], [0,256])
        plt.plot(histr, color = col)
        plt.xlim([0,256])
    plt.show()
    
plotHist(hFrm, cntMask)
plotHist(cFrm, cntMask)


# TENNIS_BALL_BGR = (79, 255, 223)
# b_bin, g_bin, r_bin = [int(chn / (255 / bins)) for chn in TENNIS_BALL_BGR]

# b_hist[b_bin]

# sieveList
    

# cntFrm = cv.drawContours(cFrm.zero().array.copy(), contours, -1, (0, 255, 0), 3)



# cFrm.mask(BoolFrame(cntMask)).put(ballReg).show()
