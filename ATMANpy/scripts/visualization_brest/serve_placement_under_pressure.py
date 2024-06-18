
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.feature_funcs as ff
import ATMANpy.funcs.vis_funcs as vs

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.Abstract import Points, Point, read_in_all_tables

from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Player, Racquet
from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D, Circle3D, Circles3D, Lines3D

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Streams import DecapStream

from ATMANpy.auxil.court import courtLines


def main(confs, name = None):
    
    ### READ IN DATA
    allTable = read_in_all_tables(confs)
    allPnts = Points.from_datatable(allTable, split_by_match = False)


    deucePnts = allPnts.filter('is_serve_deuce').enforce_direction('Bot')
    # adPnts = allPnts.filter('is_serve_ad').enforce_direction('Bot')

    nameDict = {}
    pressDict = {}
    for somePnt in deucePnts:
        srvShot = somePnt.get_serve()
        if srvShot is None:
            continue
        if not srvShot.is_first_serve():
            continue

        server = srvShot.hitPlayer.get_name()
        serve_pressure = somePnt.get_joint_score_pressure(normalize = True)
        if server in nameDict:
            nameDict[server].append(srvShot)
            pressDict[server].append(serve_pressure)
        else:
            nameDict[server] = [srvShot]
            pressDict[server] = [serve_pressure]

    from ATMANpy.types.Abstract import Shots
    jointDict = {name: (Shots(nameDict[name]), np.array(pressDict[name])) for name in nameDict}

    def get_service_line_and_indicator(side, which):
        srvReg = courtLines.get_service_box(side = side, which = which)
        
        if side == 'Bot':
            srvLn = srvReg['Bottom']
        elif side == 'Top':
            srvLn = srvReg['Top']

        ### get indicator -> which is wide, which is down the middle
        srvLn.leftPt, srvLn.rightPt
        center_x, center_y = courtLines.get_center_point().coords

        if srvLn.leftPt.x == center_x:
            indicator = {'middle': srvLn.leftPt,
                         'wide': srvLn.rightPt}
        if srvLn.rightPt.x == center_x:
            indicator = {'middle': srvLn.rightPt,
                         'wide': srvLn.leftPt}

        return srvLn, indicator

    def find_intersection_with_service_line(myShot):
        from ATMANpy.types.Points import Point2D
        from ATMANpy.types.Lines import Line

        assert myShot.is_serve()
        side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
        which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
        srvLn, _ = get_service_line_and_indicator(side, which)

        hit_pos, bnc_pos = myShot.get_ball_hit_position(), myShot.get_first_bounce_position()
        if bnc_pos == (None, None):
            return (None, None)

        flightLn = Line(Point2D(tuple(hit_pos)), Point2D(tuple(bnc_pos)))

        interPt = flightLn.weak_intersect(srvLn)
        return interPt.coords
    
    def find_lambda_of_service_line(myShot):
        from ATMANpy.types.Points import Point2D
        from ATMANpy.types.Lines import Line

        assert myShot.is_serve()
        side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
        which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
        srvLn, indicator = get_service_line_and_indicator(side, which)

        hit_pos, bnc_pos = myShot.get_ball_hit_position(), myShot.get_first_bounce_position()
        if bnc_pos == (None, None):
            return None

        flightLn = Line(Point2D(tuple(hit_pos)), Point2D(tuple(bnc_pos)))

        interPt = flightLn.weak_intersect(srvLn)

        pt_x = interPt.x
        wide_x, middle_x = indicator['wide'].x, indicator['middle'].x
        middle_dist = middle_x - pt_x
        extent_to_which_the_ball_is_wide = middle_dist / (middle_x - wide_x)
        return extent_to_which_the_ball_is_wide

    def regress_lambda_on_pressure(lambdaArr_filt, pressArr_filt):    
        import statsmodels.api as sm
    
        X = np.stack([pressArr_filt, np.ones(len(pressArr_filt))]).T
        ols = sm.OLS(lambdaArr_filt, X)
        ols_result = ols.fit()
        return ols_result
    
    player = 'Joao Sousa'
    for player in jointDict:
        print(player)
        plShots, pressArr = jointDict[player]

        # hitArr, bncArr = shots.get_ball_hit_position()[:, :2], shots.get_first_bounce_position()
        
        # pressArr_filt = pressArr[~np.isnan(bncArr).all(axis = 1)]
        # hitArr_filt = hitArr[~np.isnan(bncArr).all(axis = 1)]
        # bncArr_filt = bncArr[~np.isnan(bncArr).all(axis = 1)]

        ### find intersection with service line
        interArr = np.array([find_intersection_with_service_line(myShot) for myShot in plShots], dtype = float)
        is_not_nan = (~np.isnan(interArr)).all(axis = 1)
        
        interArr_filt = interArr[is_not_nan]
        pressArr_filt = pressArr[is_not_nan]

        jointArr_filt = np.concatenate([pressArr_filt[:, np.newaxis], interArr_filt], axis = 1)

        ### metrics
        lambdaArr = np.array([find_lambda_of_service_line(myShot) for myShot in plShots], dtype = float)
        lambdaArr_filt = lambdaArr[is_not_nan]

        fitted_ols = regress_lambda_on_pressure(lambdaArr_filt, pressArr_filt)
        # print(fitted_ols.summary())

        ### bucketing
        pressure_thresholds = (0.25, 0.4)
        shot_buckets = [[] for _ in range(len(pressure_thresholds) + 1)]
        for shot_idx, pressure in enumerate(pressArr):
            for thresh_idx, thresh in enumerate(pressure_thresholds):
                if pressure < thresh:
                    shot_buckets[thresh_idx].append(plShots[shot_idx])
                    break
            if pressure > thresh:
                shot_buckets[thresh_idx + 1].append(plShots[shot_idx])
        shot_buckets = [Shots(shotList) for shotList in shot_buckets]

        ### calculating KDEs
        from ATMANpy.types.ShapesTennis3D import Court 
        lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
        deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

        kde_scenes = []
        line_scenes = []
        offset = 0
        for bucketShots in shot_buckets:
            dummy_scene = vs.kde_hillmap(bucketShots, deucelineReg, position_func = find_intersection_with_service_line, resolution = 5, normalize_density = False)
            dummy_scene.rastArrs[0] += [0, offset, 0]

            line_scene = Scene.make_empty().put(deucelineReg.back(offset))

            line_scenes.append(line_scene)
            kde_scenes.append(dummy_scene)
            offset += 20

        kde_scene, other_scenes = kde_scenes[0], kde_scenes[1:]
        for scene in other_scenes:
            kde_scene = kde_scene.combine(scene)

        line_scene, other_scenes = line_scenes[0], line_scenes[1:]
        for scene in other_scenes:
            line_scene = line_scene.combine(scene)
        
        from ATMANpy.auxil.court import back_view_asymmetric_cam, onto_service_box_cam
        myCam = onto_service_box_cam(which = 'Deuce', side = 'Top', dist = 150).tilt(0.15)
        
        # myScn = Scene.make_court()
        # View.make(myCam, myScn, resolution = 1).snap().show()

        # myScn = Scene.make_empty().put(deucelineReg)
        # View.make(myCam, myScn).snap().show()
        
        lineView = View.make(myCam, line_scene)
        kdeView = View.make(myCam, kde_scene)
        allView = lineView.combine(kdeView)

        allView.snap().show()



        ### visualizing high pressure shot trajectories
        shot_buckets_nonet = shot_buckets[2].filter('is_in_net', negate = True)
        mySF = shot_buckets_nonet.feat_spline_simul(color = const.BGR['ball_yellow'], which_endpoint = 'actual')
        shapes = list(mySF.values())
        shapes_vectorized = [f.vectorize() for f in shapes if f is not None]
        shapes_vectorized = shapes_vectorized[:25]

        myFrms, _ = vs.animate_sequence(shapes_vectorized, background_scene = Scene.make_court(), start_cam = myCam, stop_cam = myCam,
                                        cumulate_argument = True, cumulative_foreground_transparence = 0.5
                                        )
        
        from ATMANpy.types.Streams import DecapStream
        DecapStream(myFrms, 0, 25).show(ms_between_frames = 100)

    ### bucketing
    buckets = np.quantile(pressArr_filt, [0.25, 0.5, 0.75, 0.95])



    previous_bucket = 0
    bucketedArrs = []
    for bucket in buckets:
        bucket_filter = (jointArr_filt[:, 0] >= previous_bucket) & (jointArr_filt[:, 0] < bucket)
        bucketArr = jointArr_filt[bucket_filter]
        bucketedArrs.append(bucketArr)
        previous_bucket = bucket

    bucket_filter = jointArr_filt[:, 0] >= previous_bucket
    bucketArr = jointArr_filt[bucket_filter]
    bucketedArrs.append(bucketArr)

    ### showing
    from ATMANpy.auxil.court import back_view_asymmetric_cam, onto_service_box_cam
    myCam = onto_service_box_cam(which = 'Deuce', side = 'Top', dist = 150).tilt(0.15)
    

    myScn = Scene.make_court()
    View.make(myCam, myScn, resolution = 1).snap().show()

    from ATMANpy.types.ShapesTennis3D import Court 
    lineReg = Court.make()['Top Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
    deucelineReg, adlineReg = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

    myScn = Scene.make_empty().put(deucelineReg)
    View.make(myCam, myScn).snap().show()

    from ATMANpy.types.ShapesTennis3D import Mark
    myScn = Scene.make_empty()
    height = 0
    for bucketedArr in bucketedArrs:
        heightLn = deucelineReg.back(height)
        myScn = myScn.put(heightLn)

        pressure, coords2D = bucketedArr[:, 0], bucketedArr[:, 1:]
        # heightArr = np.full((len(pressure), 1), fill_value = height)
        heightArr = np.full((len(pressure), 1), fill_value = 0)
        coords2D_adj = coords2D + (0, height)
        coords3D = np.concatenate([coords2D_adj, heightArr], axis = 1)
        
        for coord in coords3D:
            myMark = Mark.make(Point3D(tuple(coord)), flat_along_dimension = 'y')
            myScn = myScn.put(myMark)

        height += 15

    View.make(myCam, myScn).snap().show()

    vs.epanechnikov(coords3D, heightLn.rasterize(resolution = 1))
    posArr_
    showArr = coords3D



    hitPnt_vec = Point2D(shots.get_ball_hit_position()[:, :2])
    bncPnt_vec = Point2D(shots.get_first_bounce_position())

    
    Line(hitPnt_vec, bncPnt_vec)

    shots[0].get_


if __name__ == '__main__':
    
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

    name = ppl._name

    main(confs, name = None)

    
    