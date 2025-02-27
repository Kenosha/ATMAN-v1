import os
import numpy as np
import pandas as pd
        
from ATMANpy.funcs.util_funcs import get_folder_struct
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Frames import BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames, GreyFrames, MaskFrames
from ATMANpy.types.Overlays import Table, FramesRaster, overlay, frame_cascade
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Visual3D import Camera, View, Scene

from ATMANpy.types.Abstract import Points, Shots, read_in_data_tables, read_in_placeholder_tables

import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.vis_funcs as vs
import ATMANpy.auxil.constants as const
import ATMANpy.scripts.visualization_brest.atman_capabilities.utils as vis_ut

from ATMANpy.models.ParticleFilter import ParticleFilter
from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points

########################################
########### SAVE FOLDER ################
########################################

DESCRIPTOR = 'lines'
SAVE_FOLDER = 'atman_capabilities'
savePath = get_folder_struct('results_video')[SAVE_FOLDER]

#############################
#### READ IN CONFIGS ########
#############################
import ATMANpy.scripts.pipeline_brest_tightseg as ppl

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import collect_video_frames
import ATMANpy.scripts.visualization_brest.return_chip_topspin.utils as vis_ut

prm_mod = ppl.parameter_files.default
cnf_mod = ppl.config_files.default
    
avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

vid, sto = confs[0]['video_id'], confs[0]['store_name']
P = ConstFeature.from_store(vid, sto, featname = 'P').feature

###################################
######### READ IN DATA ############
###################################
dataTable = read_in_data_tables(confs)
nullTable = read_in_placeholder_tables(confs)

dataPnts = Points.from_datatable(dataTable, split_by_match = False)
nullPnts = Points.from_placeholder_table(nullTable)

allPnts = dataPnts.join(nullPnts).sort()

#####################################
#### ATMAN contraction animation ####
#####################################

from ATMANpy.types.Shapes3D import Point3D
z = 1000
oPt, cPt = Point3D((0, 0, 0)), Point3D((0, 0, z))
downCam = Camera.from_to(cPt, oPt)
text_color = const.BGR['leafy_green']
text_size = 2
left_pts = 325

titleStr = 'The Automated Tennis Match ANnotator'
titlePt = oPt.annotate(titleStr).left(left_pts).recolor(text_color).resize(text_size)
titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [titlePt])
titleFrm = titleView.snap()
titleFrm.show()

auxil_offset = 0
move1 = left_pts - 76
move2 = left_pts - 270
move3 = left_pts - 392
move4 = left_pts - 506
move5 = left_pts - 526

p1 = oPt.annotate('A').front(auxil_offset).left(move1).recolor(text_color).resize(text_size)
p2 = oPt.annotate('T').front(auxil_offset).left(move2).recolor(text_color).resize(text_size)
p3 = oPt.annotate('M').front(auxil_offset).left(move3).recolor(text_color).resize(text_size)
p4 = oPt.annotate('A').front(auxil_offset).left(move4).recolor(text_color).resize(text_size)
p5 = oPt.annotate('N').front(auxil_offset).left(move5).recolor(text_color).resize(text_size)

target1 = - 40
target2 = - 20
target3 = 0
target4 = 25
target5 = 47

p1_target = p1.right(target1 + move1)
p2_target = p2.right(target2 + move2)
p3_target = p3.right(target3 + move3)
p4_target = p4.right(target4 + move4)
p5_target = p5.right(target5 + move5)

titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [titlePt])
titleFrm = titleView.snap()
titleFrm.show()

letterView = View.make(downCam, Scene.make_empty(), annotatedPoints = [p1, p2, p3, p4, p5])
letterFrm = letterView.snap()
letterFrm.show()

# titleView = View.make(downCam, Scene.make_empty(), annotatedPoints = [p1_target, p2_target, p3_target, p4_target, p5_target])
# titleFrm = titleView.snap()
# titleFrm.show()

itpl_func = lambda l: [vs.interpolate_shape(p1, p1_target, l), 
                       vs.interpolate_shape(p2, p2_target, l),
                       vs.interpolate_shape(p3, p3_target, l),
                       vs.interpolate_shape(p4, p4_target, l),
                       vs.interpolate_shape(p5, p5_target, l)]

N = 50
pre_dim_factor = 0.5

pre_dimFrms = BGRFrames.from_frame(titleFrm, N).dim_progressively(fs_dim_factor = 1, ls_dim_factor = 0.5)
post_dimFrms = BGRFrames.from_frame(titleFrm, N).dim_progressively(fs_dim_factor = 0.5, ls_dim_factor = 0)
lambdas = np.linspace(0, 1, N, endpoint = True)

### deal with pre move dim frames
pre_dimFrms_lettered, _ = vs.animate_sequence([Scene.make_empty() for _ in range(N)], start_cam = downCam, stop_cam = downCam,
                            annotated_points = [p1, p2, p3, p4, p5], animate_over = pre_dimFrms)

### deal with post move dim frames
pt_frames = []
for i in range(N):
    l = lambdas[i]
    frm = post_dimFrms[i]

    annotated_points = itpl_func(l)
   
    ptView = View.make(downCam, Scene.make_empty(), annotatedPoints = annotated_points)
    ptFrm = ptView.snap(over_array = frm.array)
    pt_frames.append(ptFrm)

post_dimFrms_lettered = BGRFrames.from_single_frames(pt_frames)

jointFrms = BGRFrames.from_frames([pre_dimFrms_lettered, post_dimFrms_lettered])

DecapStream(jointFrms, 0, 100).show()

#####################
### OVERHEAD VIEW ###
#####################

## make an overhead view of a point, similar to SwingVision.
## little box with verbal shot info below a larger box of overhead frame

def get_shot_scenes(myPnt, ball_color = const.BGR['ball_yellow'], 
                    mark_size = 2, mark_color = const.BGR['bright_blue']):
    from ATMANpy.types.ShapesTennis3D import Mark

    bounce_scenes = []
    spline_scenes = []
    for myShot in myPnt:
        # splines
        shot_splines = myShot.feat_spline(which_endpoint = 'penultimate').values()
        n = len(shot_splines)

        shotScns = [Scene([spln.recolor(ball_color).resize(1.5)]) for spln in shot_splines if spln is not None]
        ### pad the initial None
        shotScns = [Scene.make_empty(), *shotScns] 
        ### pad the final frames, if splines are nan
        n_spln = len(shotScns)
        shotScns = [*shotScns, *[Scene.make_empty() for _ in range(n - n_spln)]]
        ### join to previous
        spline_scenes = [*spline_scenes, *shotScns]

        # bounce points
        all_frms = myShot.get_frames()
        bnc_frms = myShot.get_bounce_frames()
        bnc_x, bnc_y = myShot.get_first_bounce_position()
        
        try:
            prevScn = bounce_scenes[-1]
        except IndexError:
            prevScn = Scene.make_empty()

        if len(bnc_frms) > 0:
            bncPt = Point3D((bnc_x, bnc_y, 0))
            bncMark = Mark.make(bncPt, size = mark_size, color = mark_color)
            bnc_idx = all_frms.index(bnc_frms[0])
        else:
            bnc_idx = len(all_frms) ### this way, we'll never actually need the bncMark

        for i in range(n):
            if i >= bnc_idx:
                bounce_scenes.append(prevScn.put(bncMark))
            else:
                bounce_scenes.append(prevScn)
        
    return spline_scenes, bounce_scenes
    
myPnt = dataPnts[10]

myPnt = dataPnts[16]

spline_scenes, bounce_scenes = get_shot_scenes(myPnt)

len(bounce_scenes)
len(spline_scenes)

myFrms = myPnt.stream(confs = confs).decap().frames[:-1]

myCam = Camera.calibrated_brest_behind()
markFrms, _ = vs.animate_sequence(bounce_scenes, start_cam = myCam, stop_cam = myCam, animate_over = myFrms)
DecapStream(markFrms, 0, 100).show()

animFrms, _ = vs.animate_sequence(spline_scenes, start_cam = myCam, stop_cam = myCam, animate_over = markFrms,
                                cumulative_foreground_transparence = 0.5, cumulate_argument = True)
DecapStream(animFrms, 0, 100).show()

### top down view:

def animate_bounce_and_mark_scenes(myPnt, start_cam, stop_cam, 
                                    background = False, over = False, over_dim = 1,
                                    sub_region = True,
                                    **anim_kwargs):
    spline_scenes, bounce_scenes = get_shot_scenes(myPnt)

    if over:
        myFrms = myPnt.stream(confs = confs).decap().frames[:-1].rescale(3).dim_progressively(fs_dim_factor = 1, ls_dim_factor = over_dim)
    else:
        myFrms = None

    if background:
        courtScn = Scene.make_court()
    else:
        courtScn = Scene.make_empty()

    if sub_region:            
        from ATMANpy.types.Lines import CourtLines
        from ATMANpy.types.Shapes3D import Region3D
        subReg = CourtLines.make().get_court_region().scale(1.25)
        subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(const.BGR['smokey_grey'])
        courtScn = courtScn.put(subReg3D)

    markFrms, _ = vs.animate_sequence(bounce_scenes, start_cam = start_cam, stop_cam = stop_cam,
                                        background_scene = courtScn, 
                                        animate_over = myFrms, 
                                        **anim_kwargs)

    animFrms, _ = vs.animate_sequence(spline_scenes, start_cam = start_cam, stop_cam = stop_cam, 
                                        animate_over = markFrms,
                                        cumulative_foreground_transparence = 0.5, cumulate_argument = True,
                                        **anim_kwargs)

    return animFrms

c_x, c_y, c_z = -900, 900, 1500

startCam = Camera.calibrated_brest_behind().resize(3)
stopCam = Camera.from_to(Point3D((c_x, c_y, c_z)), Point3D((c_x, c_y, 0)))

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Shapes3D import Region3D
subReg = CourtLines.make().get_court_region().scale(1.25)
subReg3D = Region3D.from_region2D(subReg, z = -1).recolor(const.BGR['smokey_grey'])

testScn = Scene.make_court().put(subReg3D)
View.make(stopCam, testScn, resolution = 1).snap(over_array = myPnt.stream(confs = confs).decap().frames[0].rescale(3).array).show()

prmDict = {'background_resolution': 0.05, 'argument_resolution': 0.1}

animFrms = animate_bounce_and_mark_scenes(myPnt, start_cam = startCam, stop_cam = stopCam, background = True, 
                                          over = True, over_dim = 0.5, sub_region = True,
                                          start_later = 0, stop_earlier = 100,
                                          **prmDict)
DecapStream(animFrms, 0, 100).show()

animFrms[-1].show()


