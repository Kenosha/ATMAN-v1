# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:26:01 2023

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

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.extract_funcs as ex
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures
from ATMANpy.types.Frames import GreyFrame, BGRFrame, HSVFrame

from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region, Regions
from ATMANpy.types.Contours import Contour, Contours

from ATMANpy.funcs.player_funcs import get_service_ROIs, get_half_court_Lines
from ATMANpy.funcs.line_funcs import fit_frame_to_lines, get_court_schema_intersection_points

from ATMANpy.models.ParticleFilter import ParticleFilter

def load(vid, sto):
    HCF = ConstFeature.from_store(vid, sto, featname = 'H')
    # cntSF = StreamFeatures.from_store(vid, sto, featnameList = ['cand0Cnts', 'cand1Cnts', 'cand2Cnts'])
    cntSF = StreamFeature.from_store(vid, sto, featname = 'cand2Cnts') ### get only the high confidence contours!
    hitcntSF = StreamFeature.from_store(vid, sto, featname = 'hitCnt')
    hitflagSF = StreamFeature.from_store(vid, sto, featname = 'hitFlag')
    # chamfSF = StreamFeature.from_store(vid, sto, featname = 'chamferCnts')
    playBF = BoolFeature.from_store(vid, sto, featname = 'isRally')
    return HCF, cntSF, hitcntSF, hitflagSF, playBF
    
def save(chosePtSFs, vid, sto):
    chosePtSFs.rename('ball_choicePt').save(vid, sto)

def check_prop(vid, sto, name, bSFs, hitcntSFs, cap):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    from ATMANpy.types.Lines import Line
    
    import matplotlib
    matplotlib.use('Agg')

    SAVE_NAME = 'ball_directions_with_precalculated_models'
    
    base_graph_dir = ut.get_folder_struct('graphs')['pipeline_checks']
    store_dir = os.path.join(base_graph_dir, name, sto)
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    aspect_ratio = width / height
    fig_height = 36
    fig_width = fig_height * aspect_ratio

    ### establish how many random backgrounds we want to show
    grid_length = 10
    num_backgrounds = grid_length ** 2
    
    ### tighten the plot
    bleed = 0
    gridspec_kw = {'wspace': 0.05, 'hspace': 0.05}
    
    rng = np.random.default_rng()
    sf_idxs = rng.choice(range(bSFs.length()), size = num_backgrounds)
    
    fig, axes = plt.subplots(grid_length, grid_length, figsize = (fig_width, fig_height), gridspec_kw = gridspec_kw)
    
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for plot_idx, ax in enumerate(axes.ravel()):
        chosen_sf_idx = int(sf_idxs[plot_idx])
        chosen_hitSF = hitcntSFs[chosen_sf_idx].dropna()
        chosen_bSF = bSFs[chosen_sf_idx]

        sf_frames = chosen_bSF.get_frames()

        chosen_frame = int(rng.choice(sf_frames))
        try:
            next_frame = sf_frames[sf_frames.index(chosen_frame) + 1]
            d = next_frame - chosen_frame
        except IndexError:
            next_frame = None
            d = 25

        chosen_hit_contour = chosen_hitSF[chosen_frame]
        chosen_B_array = chosen_bSF[chosen_frame]

        bx, by = chosen_B_array[-2:]        
        bx, by = bx * d, by * d
        bPt = Point2D((bx, by))
        aPt = chosen_hit_contour.center()

        flightLn = Line(aPt, aPt + bPt)

        showFrm = GreyFrame.from_video(cap, chosen_frame).to_BGR()
        showFrm = showFrm.put(aPt, color = (255, 0, 0), thickness = 2)
        showFrm = showFrm.put(flightLn, color = (0, 255, 255), thickness = 2)

        showFrm = showFrm.put(f'Frame #: {chosen_frame}', color = (0, 0, 255))
        showFrm = np.flip(showFrm.array, axis = -1) 
        
        ax.imshow(showFrm)
        ax.set_axis_off()
    
    store_name = SAVE_NAME + '_' + vid + '.png'
    store_path = os.path.join(store_dir, store_name)
    
    fig.savefig(store_path)
    plt.close()



def main(conf, propagation_params, filter_init_params, filter_run_params, ball_choice_params, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    nwo, nch = conf['num_workers'], conf['chunksize']
    cap = conf['cap']
    width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    HCF, cntSF, hitcntSF, hitflagSF, playBF = load(vid, sto)
    # cntSF_low = cntSF['cand0Cnts'].rename('candCnts') ### low confidence contours
    # cntSF_mid = cntSF['cand1Cnts'].rename('candCnts') ### medium confidence contours
    # cntSF_high = cntSF['cand2Cnts'].rename('candCnts') ## high confidence contours
    cntSF_high = cntSF.rename('candCnts')

    ### Load Courtlines ###
    H = HCF.feature
    schemaLines = CourtLines.make().add_singles_netline()
    courtLines = sa.find_courtlines(H, schemaLines = schemaLines)
    base_length = courtLines['Bottom Baseline'].length() ### take the larger of the two (bot side) so that our maximum ratio is larger!
    side_length = courtLines['Bottom Singles Ad Line'].length() + courtLines['Top Singles Deuce Line'].length()
    
    ### Combine ball contours with hypothesized hit contours
    cntplusSF = cntSF_high.merge(hitcntSF, how = 'outer').apply(sa.merge_contour_into_contours)

    hitcntSFs = hitcntSF.conform(playBF.filter(playBF)).split()
    hitflagSFs = hitflagSF.conform(playBF.filter(playBF)).split()
    cntSFs_high = cntSF_high.conform(playBF.filter(playBF)).split()
    cntplusSFs = cntplusSF.conform(playBF.filter(playBF)).split()
    # chamfSFs = chamfSF.conform(playBF.filter(playBF)).split()
    
    ### Create model switching points at and around hit points
    hitSFs = hitflagSFs.dropna().thin().merge(hitcntSFs) ### thin out hitflags so we have one flag for each hit exactly.

    bSFs = hitSFs.apply_whole_mp(sf.create_propagation_models, nwo, courtLines = courtLines)

    ### Check propagation models
    if conf.get('visualize', True):
        check_prop(vid, sto, name, bSFs, hitcntSFs, cap)

    bSFs = bSFs.apply_whole_mp(sf.widen_propagation_models, nwo, **propagation_params)
    
    ### Simulate Particles ###
    # a) create SF with player chamfers and ball candidate contours
    mySFs = cntplusSFs.rename('candCnts').merge(bSFs).merge(hitSFs)

    # b) Initialize Particle Filter Model
    myFilter = ParticleFilter(**filter_init_params)
    
    # from ATMANpy.funcs.particle_funcs_vectorized import particles_to_points
    # for idx in range(10):

    #     # c) run particle filter
    #     filter_run_params = {'N': 1500, 'fps': 25, 'rectify': True}
    #     filter_run_params_additional = {'frame_width': width,
    #                                     'frame_height': height,
    #                                     'pixel_per_meter_horizontal': base_length / const.COURT_WIDTH,
    #                                     'pixel_per_meter_vertical': side_length / const.SIDE_LENGTH}
    #     partSF = myFilter.find_particles(mySFs[idx], **filter_run_params, **filter_run_params_additional, verbose = False)
    #     kitaSF = sf.kitagawa_smooth(partSF)

    #     #### RUN PARTICLE FILTER OLD TEST ###
    #     # mySFs2 = cntplusSFs.rename('candCnts').merge(chamfSFs)
    #     # partSF2 = myFilter.find_particles(mySFs2[idx], **filter_run_params, **filter_run_params_additional)
    #     # kitaSF2 = sf.kitagawa_smooth(partSF2)

    #     showSF = cntplusSFs[idx]
    #     showSF = partSF['particles']
    #     showSF = kitaSF
    #     showSF = showSF.apply(particles_to_points, particles_to_produce = 100)
    #     Stream.from_SF(showSF, cap).show(ms_between_frames = 50, frame_counter = True)


    filter_run_params_additional = {'frame_width': width,
                                        'frame_height': height,
                                        'pixel_per_meter_horizontal': base_length / const.COURT_WIDTH,
                                        'pixel_per_meter_vertical': side_length / const.SIDE_LENGTH}
    partSFs = mySFs.apply_whole_mp(myFilter.find_particles, nwo, **filter_run_params, **filter_run_params_additional) 
    
    ### Kitagawa Smoothing
    kitaSFs = partSFs.apply_whole_mp(sf.kitagawa_smooth, nwo)
    
    # cntSFs_low = cntSFs['cand0Cnts'].rename('candCnts') ### low confidence contours
    # cntSFs_mid = cntSFs['cand1Cnts'].rename('candCnts') ### medium confidence contours
    # cntSFs_high = cntSFs['cand2Cnts'].rename('candCnts') ## high confidence contours

    ### Choosing likely ball contour --> NOTE THAT WE ONLY USE THE CAND CONTOURS, NOT THE HIT CONTOURS
    jntSFs = kitaSFs.merge(cntSFs_high)
    choseSFs = jntSFs.apply(sa.choose_ball_contour, **ball_choice_params, pFilter = myFilter)
    chosePtSFs = choseSFs.apply(sa.maybe_center)

    save(chosePtSFs, vid, sto)

    # ms = 50
    # # seqNum = 14
    # for seqNum in range(1, 2):
    #     show_candidates = 0
    #     show_hit_contours = 0
    #     show_particles = 1
    #     show_kita = 0
    #     show_choice = 0
        
    #     if show_candidates:
    #         showSF = cntSFs_high[seqNum]
    #         Stream.from_SF(showSF, cap).show(ms_between_frames = ms, frame_counter = True)
        
    #     if show_hit_contours:
    #         showSF = hitcntSFs[seqNum]
    #         Stream.from_SF(showSF, cap).show(ms_between_frames = ms, frame_counter = True)
        
    #     if show_particles:
    #         showSF = partSFs[seqNum]['particles']
    #         showSF = showSF.apply(particles_to_points, particles_to_produce = 100)
    #         Stream.from_SF(showSF, cap).show(ms_between_frames = ms, frame_counter = True)
            
    #     if show_kita:
    #         showSF = kitaSFs[seqNum]['smooth_particles']
    #         showSF = showSF.apply(particles_to_points, particles_to_produce = 100)
    #         Stream.from_SF(showSF, cap).show(ms_between_frames = ms, frame_counter = True)
            
    #     if show_choice:
    #         Stream.from_SF(choseSFs[seqNum], cap).show(ms_between_frames = ms, frame_counter = True)
        
if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[2])
    
    filter_init_params = prm_mod.ball_inference_particles.filter_init_params
    filter_run_params = prm_mod.ball_inference_particles.filter_run_params
    ball_choice_params = prm_mod.ball_inference_particles.ball_choice_params
    propagation_params = prm_mod.ball_inference_particles.propagation_params

    name = ppl._name
    
    main(conf, propagation_params, filter_init_params, filter_run_params, ball_choice_params, name = name)
    
    
    
    
    
    
    

