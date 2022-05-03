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
video_id = 'pqxAmUWIoKc'
video_full_name = video_id + '.mp4'

videoDict = atu.get_folder_struct('video_files')


videoPath = videoDict[video_full_name]

cap = cv.VideoCapture(videoPath)

####################

from ATMANpy.high_level_funcs import find_homography, update_homography, find_courtlines, find_netlines
from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN, COURT_LINE_WIDTH
from ATMANpy.line_funcs import fit_frame_to_lines

schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
courtFrm = fit_frame_to_lines(schemaLines)

def track_courtlines(gFrm, prev_H, prev_config):
    if prev_H is None:
        print('No CourtLines, searching homography again...')
        res = find_homography(gFrm)
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
##################################

WIDTH_FACTOR = 2 ### how much of the doubles alley width does a player maximally next to court
DEPTH_FACTOR = 0.2 ### how much of side length does a player maximally stand behind baseline
ADD_DEPTH_FACTOR = 1.25

COURT_DILATION = 1.5
NET_DILATION = 2.5
ROI_DILATION = 1

### top side depth adjustment - player figure is tall, so looks to be outside of ROI.
# we could adjust this with the camera matrix, but let's do something eyeballed here, for the moment.

from ATMANpy.player_funcs import get_initial_ROIs, find_player

def track_players(cFrm, H, courtLines, netLines, verbose = False):
    ROI_lower, ROI_upper = get_initial_ROIs(schemaLines,
                                         width_factor = WIDTH_FACTOR, 
                                         depth_factor = DEPTH_FACTOR, 
                                         add_depth_factor_top = ADD_DEPTH_FACTOR)
    
    if verbose: courtFrm.to_BGR().put(ROI_upper, color = (0, 0, 255), thickness = 2).put(ROI_lower, color = (0, 255, 0), thickness = 2).show()
    
    projROI_lower = ROI_lower.transform('apply_homography', H).transform('round_points')
    projROI_upper = ROI_upper.transform('apply_homography', H).transform('round_points')
    
    if verbose: cFrm.put(projROI_upper, color = (0, 0, 255), thickness = 2).put(projROI_lower, color = (0, 255, 0), thickness = 2).show()

        
    playerRegion_bottom = find_player(cFrm, projROI_lower, courtLines, netLines, 
                                   line_width = COURT_LINE_WIDTH,
                                   court_dilation_factor = COURT_DILATION, 
                                   net_dilation_factor = NET_DILATION, 
                                   roi_dilation_factor = ROI_DILATION, 
                                   which = 'Bottom',
                                   verbose = verbose)
    
    playerRegion_top = find_player(cFrm, projROI_upper, courtLines, netLines, 
                                   line_width = COURT_LINE_WIDTH,
                                   court_dilation_factor = COURT_DILATION, 
                                   net_dilation_factor = NET_DILATION, 
                                   roi_dilation_factor = ROI_DILATION, 
                                   which = 'Top',
                                   verbose = verbose)

    return playerRegion_bottom, playerRegion_top

def put_list(frm, putList, **kwargs):
    if len(putList) == 0:
        return frm
    else:
        p, ps = putList[0], putList[1:]
        return put_list(frm.put(p, **kwargs), ps, **kwargs)

##############

H = None
config = None
    
START_FRAME = 11710
STOP_FRAME = 11940

from ATMANpy.cv_funcs import video_stream
gen = video_stream(cap, start_frame = START_FRAME, stop_frame = STOP_FRAME)
rawArr = next(gen) ### this is needed to "prime" the generator

########## JUST SHOW VIDEO ########
# while True:
#     try:
#         rawArr = next(gen)  
#     except:
#         cv.destroyAllWindows()  
#         break
    
#     if rawArr is not None:
#         # try:
#         rawFrm = BGRFrame(rawArr)
#         gFrm = rawFrm.to_grey()
        
#         try:
#             gen.send(gFrm.array)
#         except StopIteration:
#             print('DONE!')
#             cv.destroyAllWindows()  
#             break
#     else:
#         cv.destroyAllWindows()  
#         break
        
########## VIDEO with TRACK LINES #########
while True:
    try:
        rawArr = next(gen)  
    except:
        cv.destroyAllWindows()  
        break
    
    if rawArr is not None:
        # try:
        cFrm = BGRFrame(rawArr)
        gFrm = cFrm.to_grey()
        
        courtLines, netLines, H, config = track_courtlines(gFrm, H, config)
        playerRegion_bottom, playerRegion_top = track_players(cFrm, H, courtLines, netLines)
        
        ### produce 
        outFrm = put_list(cFrm, [courtLines, netLines, 
                                 playerRegion_bottom, playerRegion_top], thickness = 2)
        
        try:
            gen.send(outFrm.array)
        except StopIteration:
            print('DONE!')
            cv.destroyAllWindows()  
            break
    else:
        cv.destroyAllWindows()  
        break
        
        
    

#### a Stream decorator!!!!!!!!!!! unfinished.
def stream(func, **args):
        
    from ATMANpy.cv_funcs import video_stream
    gen = video_stream(cap, start_frame = START_FRAME, stop_frame = STOP_FRAME)
    rawArr = next(gen) ### this is needed to "prime" the generator

    while True:
        try:
            rawArr = next(gen)  
        except:
            cv.destroyAllWindows()  
            break
        
        if rawArr is not None:
            # try:
            rawFrm = BGRFrame(rawArr)
            gFrm = rawFrm.to_grey()
            
            func
            
            try:
                gen.send(gFrm.array)
            except StopIteration:
                print('DONE!')
                cv.destroyAllWindows()  
                break
        else:
            cv.destroyAllWindows()  
            break
    

