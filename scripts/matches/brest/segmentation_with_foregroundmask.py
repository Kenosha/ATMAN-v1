# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:16:32 2023

@author: gianl
"""


import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

### GET VIDEO ###
from ATMANpy.funcs.util_funcs import get_video
from ATMANpy.funcs.cv_funcs import join_video_captures

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points
schemaLines = CourtLines.make().add_singles_netline()
schemaFrm = fit_frame_to_lines(schemaLines)

conf = ut.read_config('brest')
vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
nwo, nch = conf['num_workers'], conf['chunksize']
cap = conf['cap']

if __name__ == '__main__':
    frameList = list(range(conf['start_frame'], conf['stop_frame'] + 1))
    
    burnStms = StreamList.from_store('foregroundMask', vid, sto, 
                                     cap_color_space = 'bool', target_color_space = 'mask')
    
    mFrm = burnStms[0].get(300)
    
    ratSFs = burnStms.extract_chunked_mp(ex.black_comparison, nwo, nch)
    ratSF = ratSFs.concatenate()
    ratSF.to_series().plot()
    
    THRESHOLD = 0.25 ### if more than 25% of the frame is lit, what is marked foreground is NOT foreground!
    boolSF = ratSF.apply(sa.exceeds_threshold, threshold = THRESHOLD).to_bool()
    
    boolSF = boolSF.conform(frameList).fillna(False) ### fill with False, since the segmented out sections 
                                                       # (which get created by conforming) are
                                                       # 1) not technically above threshold
                                                       # 2) can be taken care of by combining the isCourt boolean with OR
    boolSF = boolSF.neg().rename('isValidFore')
    
    boolSF.save(vid, sto)
    
    