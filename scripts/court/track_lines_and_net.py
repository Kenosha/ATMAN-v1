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
from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN

schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()

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
    
        outFrm = gFrm.to_BGR().put(courtLines, thickness = 3)
        outFrm = outFrm.put(netLines, color = (255, 0, 0), thickness = 3)
    else:
        outFrm = gFrm
        H, config = None, None
        
    return outFrm, H, config
        #################################


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
        rawFrm = BGRFrame(rawArr)
        gFrm = rawFrm.to_grey()
        
        rawFrm = BGRFrame(rawArr)
        gFrm = rawFrm.to_grey()
        gFrm, H, config = track_courtlines(gFrm, H, config)
        
        try:
            gen.send(gFrm.array)
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
    

