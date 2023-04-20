

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import ATMANpy.utils as atu
from ATMANpy.Points import Point2D, SafePoint2D
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


from ATMANpy.high_level_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)

from ATMANpy.high_level_funcs import find_homography_225_only, update_homography, find_courtlines, find_netlines
def track_courtlines(gFrm, prev_H, prev_config):
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

def put_list(frm, putList, **kwargs):
    if len(putList) == 0:
        return frm
    else:
        p, ps = putList[0], putList[1:]
        return put_list(frm.put(p, **kwargs), ps, **kwargs)

##############

courtlineList = []
RETRY_TRACK_AFTER_FRAMES = 0 ### retry tracking after five seconds

H = None
config = None
    
START_FRAME = 11000
STOP_FRAME = 18000

from ATMANpy.cv_funcs import video_stream2
gen = video_stream2(cap, start_frame = START_FRAME, stop_frame = STOP_FRAME)


# next(gen)
# ######### JUST SHOW VIDEO ########
# while True:
#     try:
#         rawArr = next(gen)  
#     except:q
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
untracked_counter = RETRY_TRACK_AFTER_FRAMES

frmNum, rawArr = next(gen) ### this is needed to "prime" the generator
while True:    
    if rawArr is not None:
        # try:
        cFrm = BGRFrame(rawArr)
        gFrm = cFrm.to_grey()
        
        if H is not None or untracked_counter >= RETRY_TRACK_AFTER_FRAMES:
            untracked_counter = 0
            courtLines, netLines, H, config = track_courtlines(gFrm, H, config)
            
            ### produce 
            if courtLines is not None:
                courtlineList.append((frmNum, courtLines, H))
                outFrm = put_list(cFrm, [courtLines, netLines], thickness = 2)
            else:
                outFrm = cFrm
            
        else:
            untracked_counter += 1
            outFrm = cFrm
            
        try:
            outFrm = outFrm.put(str(frmNum), color = (0, 0, 255))
            frmNum, rawArr = gen.send(outFrm.array)
        except StopIteration:
            print('DONE!')
            cv.destroyAllWindows()  
            break
    else:
        cv.destroyAllWindows()  
        break
        
###########
len(courtlineList)
# how do we consolidate these courtlines? 
# what we need to derive is: 
    # a rule that tells us whether a given courtline corresponds to the 'normal' set of courtlines
    # the average color of all pixels on the frame in the 'normal' set of courtlines.



# frmNum, aLns, H = courtlineList[100]


NORM_DEV_THRESHOLD = 1

def court_points(courtLines):
    p1, p2 = courtLines['Top Baseline'].leftPt, courtLines['Top Baseline'].rightPt
    p3, p4 = courtLines['Bottom Baseline'].leftPt, courtLines['Bottom Baseline'].rightPt
    return [p1.coords, p2.coords, p3.coords, p4.coords]

courtpoints = np.array([court_points(cLns) for frmNum, cLns, H in courtlineList])

avg_points = np.average(courtpoints, axis = 0)
med_points = np.median(courtpoints, axis = 0)
std_points = np.std(courtpoints, axis = 0)

medRegion = Region.from_points(*[Point2D(tuple(c)) for c in med_points])
gFrm.to_BGR().put(medRegion).show()

avgRegion = Region.from_points(*[Point2D(tuple(c)) for c in avg_points])
gFrm.to_BGR().put(avgRegion).show() ### average is bad, median is great!

normalized_deviation = (courtpoints - med_points) / std_points
within_bool = np.abs(normalized_deviation) < NORM_DEV_THRESHOLD
all_corners_within = np.all(within_bool, axis = (1, 2))

print(f'Number of Court Lines Found:   {len(courtpoints)}')
print(f' ... of which judged normal:   {all_corners_within.sum()}')
print(f' ... of which judged abnormal: {(~all_corners_within).sum()}')
### we can actually be quite strict here, exclusion-wise, but it probably
# doesn't matter much. But if we're strict, we'll have very similar lines,
# which will make our colored schema more precise 
# (since there will be less aberration caused by flattening of objects with height)
# but our burn-in will be longer. 
# but the results are good even with small samples as long as we take the median color.
# so let's be strict. 
# In consequence, in the code below, many court regions will be labelled 'weird' which probably look completely ok.

for pts, normBool in zip(courtpoints, all_corners_within):
    myRegion = Region.from_points(*[Point2D(tuple(c)) for c in pts])
    rFrm = gFrm.to_BGR().put(myRegion, thickness = 2, color = (255, 0, 0))
    if normBool:
        rFrm = rFrm.put('Normal!', color = (0, 255, 0), thickness = 4)
    else:
        rFrm = rFrm.put('Weird!', color = (0, 0, 255), thickness = 4)
        rFrm.show()



####### Map back into court schema #######
# Assumption: Everything is flat. 
# using this, we can simply apply the inverse homography to the frame
# and so calculate point colors on the court schema!
# if court lines detected are 'normal', 
# we use them to calculate an average.
# TODO: At some point: Account for net NOT being flat! (for this, need 3D camera matrix from net post(s)... see notebook.)

#### GET ARBITRARY FRAME ####
FRAME_NUM = 11500
gFrm = GreyFrame.from_video(cap, FRAME_NUM)
gFrm.show()

cFrm = BGRFrame.from_video(cap, FRAME_NUM)
cFrm.show()

courtlines_to_check = len(courtlineList)
wrpList = []

for i in range(courtlines_to_check):
    
    if i % 100 == 0:
        print(f'checked {i} of {courtlines_to_check} sets of court lines...')
    
    frmNum, aLns, H = courtlineList[i]
    pts = courtpoints[i]
    normBool = all_corners_within[i]
    
    if normBool:
        H_inv = np.linalg.inv(H)
        cFrm = BGRFrame.from_video(cap, frmNum)
        # cFrm.show()
        cols_rows = (schemaFrm.cols, schemaFrm.rows)
        wrpArr = cv.warpPerspective(cFrm.array, H_inv, cols_rows)
        
        ### prune away parts that are superfluous because they are completely dark
        ## in the unimaginable case that we have black rows within the picture, (not at the edges), 
        ## we'd be fucked.
        row_is_not_dark = wrpArr.any(axis = 1).any(axis = 1) 
        wrpArr = wrpArr[row_is_not_dark]
        
        # wrpFrm = BGRFrame(wrpArr)
        # wrpFrm.show()
        wrpList.append(wrpArr)

wrpArrs = np.stack(wrpList, axis = 0)
wrpArrs.shape

# del courtlineList

### Due to memory intensive standard deviation calculation: limit the number of warped arrays we take.
NUM_ARRS = 1000
selected = np.random.choice(wrpArrs.shape[0], NUM_ARRS, replace=False) 

test = wrpArrs[selected]

avg_test = np.average(test, axis = 0)
med_test = np.median(test, axis = 0)
std_test = np.std(test, axis = 0) ### memory requirements for standard deviation are quite high.....
                                    # check task manager while it runs; RAM usage increases a lot.

avgFrm = BGRFrame(avg_test.astype('uint8'))
avgFrm.show()

medFrm = BGRFrame(med_test.astype('uint8'))
medFrm.show() ### BEAUTIFUL! 
### EVEN WITH first 100 Frames only, the median frame looks gorgeous!
### Maybe we can severely reduce the burn-in time.






# avg_frame = np.average(wrpArrs, axis = 0)
# med_frame = np.median(wrpArrs, axis = 0)
# std_frame = np.std(wrpArrs, axis = 0)

# from ATMANpy.line_funcs import calculate_homography
# calculate_homography(myRegion.points, schemaRegion.points)

  


