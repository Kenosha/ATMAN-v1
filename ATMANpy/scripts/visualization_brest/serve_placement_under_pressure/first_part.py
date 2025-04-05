
import os
import numpy as np
import pandas as pd
        
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame, MaskFrame
from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Visual3D import Camera, View, Scene, DummyScene
from ATMANpy.types.Overlays import FramesRaster
from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D, Point3D

from ATMANpy.types.Abstract import NullPoint, Points, Shots, read_in_data_tables, read_in_placeholder_tables, read_in_colinear_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

#############################
#### READ IN CONFIGS ########
#############################
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature
cap = confs[0]['cap']


###################################
######### READ IN DATA ############
###################################
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)
coliTable = read_in_colinear_tables(confs)

### replace points suspected of co-linearity with the co-linear stand-in
to_replace = [(0, 110, 1), (0, 112, 2)] ### match index, point index, shot index

replTable = dataTable.copy()

for mtc_idx, pnt_idx, sht_idx in to_replace:
    mtc_coliTable = coliTable.loc[mtc_idx]
    mtc_replTable = replTable.loc[mtc_idx]

    sht_coliTable = mtc_coliTable[(mtc_coliTable['point_idx'] == pnt_idx) & (mtc_coliTable['shot_idx'] == sht_idx)]
    mtc_replTable.loc[sht_coliTable.index] = sht_coliTable

dataPnts = Points.from_datatable(replTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

##############################
########## Raster ############
##############################

def maybe_get_pressure(somePnt):
    try:
        press = somePnt.get_total_match_score_pressure()
    except:
        press = None
    return press

pressures = dataPnts.apply(maybe_get_pressure)

### find a high pressure point ###
max_press = pressures[~np.isnan(pressures)].max()
max_idxs = pressures == max_press
idxs = np.arange(len(dataPnts))
print(idxs[max_idxs])

### 
dataPnts[1131].get_game_score()
dataPnts[1131].get_point_score()

hi_pressure_idx = 1131
hiPnt = dataPnts[hi_pressure_idx]
# hiPnt.stream(confs = confs).show()

### find a low pressure point from the same side by the same player in the same match
mtc_idx = hiPnt.get_match_index()
mtcPnts = dataPnts.get(mtc_idx)

def sousa_serves_ad(somePnt):
    is_sousa = somePnt.get_server().get_name() == 'Joao Sousa'
    is_ad = somePnt.is_serve_ad()
    return is_sousa and is_ad

filtPnts = mtcPnts.filter(sousa_serves_ad)

pressures_in_mtc_filt = filtPnts.apply(maybe_get_pressure)

which_lowest_pressure = 2
min_press = np.sort(pressures_in_mtc_filt)[which_lowest_pressure]

min_idxs = pressures_in_mtc_filt == min_press
mtc_idxs = np.arange(len(filtPnts))
min_idx = int(mtc_idxs[min_idxs][0])

loPnt = filtPnts[min_idx]
# loPnt.stream(confs = confs).show()

### initial frame raster ###
widen_by = 100
hiStm = hiPnt.stream(confs = confs).widen(widen_by)
loStm = loPnt.stream(confs = confs).widen(widen_by)

pre_hiFrms = hiStm.decap().frames[:widen_by].rescale(3/2)
pre_loFrms = loStm.decap().frames[:widen_by].rescale(3/2)

pre_hiFrms = pre_hiFrms.framed(thickness = 3, color = const.BGR['bright_grey'])
pre_loFrms = pre_loFrms.framed(thickness = 3, color = const.BGR['bright_grey'])

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Arrow

### high leverage text frame
arrPt00 = Point2D((250, 235))
arrPt01 = Point2D((50, 235))
myArr0 = Arrow.make(arrPt00, arrPt01, hat_length = 20)

textFrm_hi = BGRFrame.make_black(cols = 960, rows = 540)
textFrm_hi_0 = textFrm_hi.put('High Leverage Point', color = const.BGR['intense_red'], 
                            size = 3, thickness = 2,
                            where = (300, 250))
textFrm_hi_1 = textFrm_hi_0.put(myArr0, color = const.BGR['intense_red'], thickness = 2)
textFrm_hi_2 = textFrm_hi_1.put('Server is well advised to employ\na serve that maximizes\nwin probability.', color = const.BGR['bright_grey'], 
                            size = 2, thickness = 1,
                            where = (300, 350))
textFrm_hi_2.show()

### lo leverage text frame
arrPt10 = Point2D((650, 235))
arrPt11 = Point2D((910, 235))
myArr1 = Arrow.make(arrPt10, arrPt11, hat_length = 20)

textFrm_lo = BGRFrame.make_black(cols = 960, rows = 540)
textFrm_lo_0 = textFrm_lo.put('Low Leverage Point', color = const.BGR['leafy_green'], 
                            size = 3, thickness = 2,
                            where = (100, 250))
textFrm_lo_1 = textFrm_lo_0.put(myArr1, color = const.BGR['leafy_green'], thickness = 2)
textFrm_lo_2 = textFrm_lo_1.put('Server might mix in a serve that is not optimal\nto retain unpredictability.', color = const.BGR['bright_grey'], 
                            size = 2, thickness = 1,
                            where = (100, 350))
textFrm_lo_2.show()

##########
myRst = FramesRaster(2, 2, borders = None)

myRst.add(pre_hiFrms, 0, 0)
rstFrms0 = myRst.render()
# DecapStream(rstFrms0, 0, 100).show()

myRst = myRst.to_frame_raster().to_frames_raster()
myRst.add(BGRFrames.from_frame(textFrm_hi_1, 50), 0, 1)
rstFrms1 = myRst.render()
# DecapStream(rstFrms1, 0, 100).show()

myRst.add(BGRFrames.from_frame(textFrm_hi_2, 50), 0, 1)
rstFrms2 = myRst.render()
# DecapStream(rstFrms2, 0, 100).show()

myRst.add(pre_loFrms, 1, 1)
rstFrms3 = myRst.render()
# DecapStream(rstFrms3, 0, 100).show()

myRst = myRst.to_frame_raster().to_frames_raster()
myRst.add(BGRFrames.from_frame(textFrm_lo_1, 50), 1, 0)
rstFrms4 = myRst.render()
# DecapStream(rstFrms4, 0, 100).show()

myRst.add(BGRFrames.from_frame(textFrm_lo_2, 50), 1, 0)
rstFrms5 = myRst.render()
# DecapStream(rstFrms5, 0, 100).show()

rst_frames = [rstFrms0, rstFrms1, rstFrms2, rstFrms3, rstFrms4, rstFrms5]

vis_ut.show(rst_frames)

