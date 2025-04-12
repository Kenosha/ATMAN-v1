import numpy as np
import scipy
from ATMANpy.auxil.court import courtLines
from ATMANpy.types.Points import Point2D

from ATMANpy.types.Overlays import Legend, LegendRow
import ATMANpy.auxil.constants as const

from ATMANpy.types.Streams import DecapStream
from ATMANpy.types.MultiFrames import BGRFrames

def show(all_frames, ms_between_frames = 25):
    DecapStream(BGRFrames.from_frames(all_frames), 0, 100).show(ms_between_frames = ms_between_frames)

def get_service_line_and_indicator(side, which):
    srvReg = courtLines.get_service_box(side = side, which = which)
    
    if side == 'Bottom':
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

    hit_pos = myShot.get_ball_hit_position()
    if myShot.is_in_net():
        x, y, _ = myShot.get_ball_last_position()
    else:
        x, y = myShot.get_first_bounce_position()
    bnc_pos = x, y
    
    if bnc_pos == (None, None):
        return (None, None)

    flightLn = Line(Point2D(tuple(hit_pos)), Point2D(tuple(bnc_pos)))

    interPt = flightLn.weak_intersect(srvLn)
    return interPt.coords

def is_down_middle(myShot):
    from ATMANpy.types.Lines import Line
    x, _ = find_intersection_with_service_line(myShot)
    if x is None:
        return False

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, indDict = get_service_line_and_indicator(side, which)

    xm, _ = indDict['middle'].coords
    xw, _ = indDict['wide'].coords

    xc = 2/3 * xm + 1/3 * xw ### coordinate that delimits the inner third
    
    if xm < xw:
        x_clipped = np.clip(x, a_min = xm, a_max = xw)
        if xm <= x_clipped <= xc:
            return True
        else:
            return False
    else:
        x_clipped = np.clip(x, a_min = xw, a_max = xm)
        if xc <= x_clipped <= xm:
            return True
        else:
            return False

def is_out_wide(myShot):
    from ATMANpy.types.Lines import Line
    x, _ = find_intersection_with_service_line(myShot)
    if x is None:
        return False

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, indDict = get_service_line_and_indicator(side, which)

    xm, _ = indDict['middle'].coords
    xw, _ = indDict['wide'].coords

    xc = 1/3 * xm + 2/3 * xw ### coordinate that delimits the inner third
    
    if xm < xw:
        x_clipped = np.clip(x, a_min = xm, a_max = xw)
        if xc <= x_clipped <= xw:
            return True
        else:
            return False
    else:
        x_clipped = np.clip(x, a_min = xw, a_max = xm)
        if xw <= x_clipped <= xc:
            return True
        else:
            return False


def make_stat_string(loShots, hiShots, alpha = 0.05, lo_desc = 'Low Pressure', hi_desc = 'High Pressure'):
    ######## Tests for similarity of distribution #########
    low_x = np.array([find_intersection_with_service_line(aShot) for aShot in loShots], dtype = float)[:, 0]
    hi_x = np.array([find_intersection_with_service_line(aShot) for aShot in hiShots], dtype = float)[:, 0]

    ks_res = scipy.stats.kstest(low_x, hi_x)           ### kolmogorov-smirnov test (2-sample)
    ad_res = scipy.stats.anderson_ksamp([low_x, hi_x]) ### anderson darling test (2-sample)
    ks_p, ad_p = ks_res.pvalue, ad_res.pvalue

    if ks_p >= alpha and ad_p >= alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions\ncan not be shown to be statistically different at a = {alpha}.'
    elif ks_p < alpha and ad_p < alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions\nare statistically different at a = {alpha}.'
    elif ks_p < alpha and ad_p >= alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions\nare statistically different at a = {alpha}\nonly according to the Kolmogorov-Smirnov test.'
    elif ks_p >= alpha and ad_p < alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions\nare statistically different at a = {alpha}\nonly according to the Anderson-Darling test.'

    ksStr = f'Kolmogorov-Smirnov 2-Sample p-value: {round(ks_p, 3)}'
    adStr = f'Anderson-Darling 2-Sample p-value: {round(ad_p, 3)}'
    jntStr = '\n'.join([textStr, '', ksStr, adStr])

    return jntStr

deuceReg = courtLines.make().add_singles_netline().get_service_box(which = 'Deuce', side = 'Top')
deuceReg_enlarged = deuceReg.scale(1.25)

adReg = courtLines.make().add_singles_netline().get_service_box(which = 'Ad', side = 'Top')
adReg_enlarged = adReg.scale(1.25)



def is_player(myShot, player_name = 'Joao Sousa'):
    return myShot.hitPlayer.get_name() == player_name

def is_bounce_in_region(myShot, myReg):
    coords = myShot.get_first_bounce_position()
    if coords == (None, None):
        return False
    else:
        bouncePt = Point2D(coords)
        return bouncePt.within(myReg)
    
def is_bounce_in_ad(myShot, side = 'Top'):
    adReg = courtLines.make().add_singles_netline().get_service_box(which = 'Ad', side = side)
    adReg_enlarged = adReg.scale(1.25)
    return is_bounce_in_region(myShot, adReg_enlarged)

def is_bounce_in_deuce(myShot, side = 'Top'):
    deuceReg = courtLines.make().add_singles_netline().get_service_box(which = 'Deuce', side = side)
    deuceReg_enlarged = deuceReg.scale(1.25)
    return is_bounce_in_region(myShot, deuceReg_enlarged)

def make_legend_frame(player_name, handedness, side, player_spaces = 15, 
                      index_color = const.BGR['bright_grey'], data_color = const.BGR['grass_green']):
    from ATMANpy.types.Overlays import Table

    if handedness == 'Right':
        hand_spaces = 16
    elif handedness == 'Left':
        hand_spaces = 16

    if side == 'Deuce':
        side_spaces = 15
    elif side == 'Ad':
        side_spaces = 16

    table_data = {'Player': {'': f'{player_name:>{player_spaces}}'}, 
                  'Handedness': {'': f'{handedness:>{hand_spaces}}'},
                  'Serving': {'': f'{side:>{side_spaces}}'}}

    myTab = Table(  table_data, dimensions = (350, 200), 
                #   title = 'Legend', title_horizontal_offset = 3.25, title_vertical_offset = 0.25, title_color = color,
                    header_row_height = 0.5, 
                    data_text_color = data_color, index_text_color = index_color,
                    skeleton_shrinkage = 0.2, index_offset = 0.25, automatic_vertical_offset_text_line_height = 0.25)
    
    tabFrm = myTab.render()
    tabFrm = tabFrm.framed(2, index_color)
    return tabFrm

def collect_video_frames(shots, confs, widen_by = 5):
    frms_list = []
    ### Frames object
    for shot in shots:
        idx = shot.get_match_index()

        cap = confs[idx]['cap']
        shotStm = shot.stream(cap).widen(widen_by)
        shotDtm = shotStm.decap()

        shotFrms = shotDtm.frames

        frms_list.append(shotFrms)

    return frms_list


def split_kde(serveShots_sample, side, p_thresh_lo = 0.04, p_thresh_hi = 0.08, **kde_kwargs):
    from ATMANpy.types.ShapesTennis3D import Court
    from ATMANpy.types.Shapes3D import Region3D, Line3D, Point3D
    from ATMANpy.types.Visual3D import Scene
    from ATMANpy.types.Lines import CourtLines
    import ATMANpy.funcs.vis_funcs as vs

    srvLn, indDict = CourtLines.make().add_singles_netline().get_service_line_and_indicator('Bottom', side)

    lineReg = Court.make()['Bottom Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
    adlineReg, deucelineReg  = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

    if side == 'Deuce':
        showReg = deucelineReg
    else:
        showReg = adlineReg

    ### MAKE KDES WITH SPLIT CATEGORIES
    lowShots = serveShots_sample.filter('get_total_match_score_pressure', is_below = p_thresh_lo)
    midShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = p_thresh_lo, is_below = p_thresh_hi)
    hiShots = serveShots_sample.filter('get_total_match_score_pressure', is_above = p_thresh_hi)

    low_kde_scene = vs.kde_hillmap(lowShots, one_color = const.BGR['grass_green'], showReg = showReg, **kde_kwargs)
    mid_kde_scene = vs.kde_hillmap(midShots, one_color = const.BGR['soft_yellow'], showReg = showReg, **kde_kwargs)
    high_kde_scene = vs.kde_hillmap(hiShots, one_color = const.BGR['intense_red'], showReg = showReg, **kde_kwargs)

    mid_kde_scene.rastArrs[0] -= [0, 60, 0]
    high_kde_scene.rastArrs[0] -= [0, 120, 0]

    splitScn = low_kde_scene.combine(mid_kde_scene).combine(high_kde_scene)

    ### SERVICE BOX GRAPH
    width = courtLines.see_dimensions()['otherline_width']
    x_ax, y_ax = srvLn, courtLines['Bottom Center Line']
    x_ax_3D = Region3D.from_line3D(Line3D.from_line2D(x_ax, z = 0), width = width, dim = 'y')
    y_ax_3D = Region3D.from_line3D(Line3D.from_line2D(y_ax, z = 0), width = width, dim = 'x')
    xyScn = Scene([x_ax_3D, y_ax_3D])

    origPt = x_ax.weak_intersect(y_ax)
    origPt_3D = Point3D.from_point2D(origPt, z = 0)

    if side == 'Deuce':
        loPt = origPt_3D.left(25).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
        midPt = origPt_3D.front(70).left(35).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
        hiPt = origPt_3D.front(150).left(37).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

        tPt = origPt_3D.right(0).back(10).annotate('Down the Middle')    
        wPt = origPt_3D.right(100).back(10).annotate('Out Wide')

    elif side == 'Ad':
        loPt = origPt_3D.right(15).annotate(f'Low Pressure \n (Point Impact <4%) \n n = {len(lowShots)}').recolor(const.BGR['grass_green'])
        midPt = origPt_3D.front(70).right(20).annotate(f'Medium Pressure \n (Point Impact 4-8%) \n n = {len(midShots)}').recolor(const.BGR['soft_yellow'])
        hiPt = origPt_3D.front(150).right(25).annotate(f'High Pressure \n (Point Impact >8%) \n n = {len(hiShots)}').recolor(const.BGR['intense_red'])

        tPt = origPt_3D.left(15).back(10).annotate('Down the Middle')    
        wPt = origPt_3D.left(110).back(10).annotate('Out Wide')

    diagPt = Point3D((x_ax.rightPt.x, y_ax.leftPt.y, 0))
    jntStr = make_stat_string(lowShots, hiShots)
    statPt = diagPt.annotate(jntStr).front(110).right(40)

    annPnts = [loPt, midPt, hiPt, tPt, wPt, statPt]

    return splitScn, xyScn, annPnts

def joint_kde(serveShots_sample, side, **kde_kwargs):
    from ATMANpy.types.ShapesTennis3D import Court
    from ATMANpy.types.Shapes3D import Region3D, Line3D, Point3D
    from ATMANpy.types.Visual3D import Scene
    from ATMANpy.types.Lines import CourtLines
    import ATMANpy.funcs.vis_funcs as vs

    lineReg = Court.make()['Bottom Service Line'] ### in case of changing away from top or deuce, these might have to be changed!
    adlineReg, deucelineReg  = lineReg.split(how = 'vertical') ### in case of changing away from top or deuce, these might have to be changed!

    if side == 'Deuce':
        showReg = deucelineReg
    else:
        showReg = adlineReg

    ### JOINT KDEs    
    joint_kde_null_scene = vs.kde_hillmap(serveShots_sample, max_height = -0.5, one_color = const.BGR['strong_blue'], showReg = showReg,
                                          **kde_kwargs)
    joint_kde_scene = vs.kde_hillmap(serveShots_sample, one_color = const.BGR['strong_blue'], showReg = showReg,
                                     **kde_kwargs)

    return joint_kde_null_scene, joint_kde_scene

def morph_split_kdes(splitScn0, xyScn0, annPnts0, legFrm0,
                     splitScn1, xyScn1, annPnts1, legFrm1,
                     myCam, 
                     num_frames = 50, tlPt_leg = Point2D((0, 0)),
                     **prmDict):
    from ATMANpy.types.Visual3D import View
    from ATMANpy.types.Overlays import overlay
    import ATMANpy.funcs.vis_funcs as vs

    ### interpolate the annotated points and legends
    emFrm0 = View.make(myCam, xyScn0, annotatedPoints = annPnts0, resolution = prmDict['background_resolution']).snap()
    emFrm1 = View.make(myCam, xyScn1, annotatedPoints = annPnts1[:-1], resolution = prmDict['background_resolution']).snap()

    emFrm0 = overlay(emFrm0, legFrm0, tlPt_leg)
    emFrm1 = overlay(emFrm1, legFrm1, tlPt_leg)

    emFrms_itpl = vs.interpolate_frames(emFrm0, emFrm1, num_frames)
    # DecapStream(emFrms_itpl, 0, 100).show()

    ### animate over these the interpolated split scenes
    itplScns = vs.interpolate_dummy_scenes(splitScn0, splitScn1, 50)

    itplFrms, _ = vs.animate_sequence(itplScns, start_cam = myCam, stop_cam = myCam, 
                                      animate_over = emFrms_itpl, **prmDict)
    
    return itplFrms
