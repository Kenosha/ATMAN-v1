
import numpy as np
import matplotlib.pyplot as plt
import os

import statsmodels.api as sm

import ATMANpy.auxil.constants as const

from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D

from ATMANpy.scripts.visualization_brest.serve_placement_under_pressure.utils import is_out_wide, is_down_middle

#######################################
######### COURT REGIONS ###############
#######################################
courtLines = CourtLines.make().add_singles_netline()
botReg = courtLines.get_half_region(side = 'Bottom', width = 'Singles')
topReg = courtLines.get_half_region(side = 'Top', width = 'Singles')

##########################################
########### POINT APPLICATORS ############
##########################################

def is_speed_above_threshold(myPnt, threshold = 140):
    return myPnt.get_serve().get_initial_speed() > threshold

def is_speed_below_threshold(myPnt, threshold = 140):
    return myPnt.get_serve().get_initial_speed() <= threshold

def is_too_many_frames(myPnt, threshold = 35):
    return myPnt.get_serve().get_num_frames() > threshold

def is_no_bounce(myPnt):
    x, y = myPnt.get_serve().get_first_bounce_position()
    return x is None and y is None

def is_not_pathological(myPnt, threshold_speed = 140, threshold_frames = 35):
    speed_bool = is_speed_above_threshold(myPnt, threshold = threshold_speed)
    frames_bool = not is_too_many_frames(myPnt, threshold = threshold_frames)
    bounce_bool = not is_no_bounce(myPnt)
    return speed_bool and frames_bool and bounce_bool

def is_pathological(myPnt, threshold_speed = 140, threshold_frames = 35):
    return not is_not_pathological(myPnt, threshold_speed = threshold_speed, threshold_frames = threshold_frames)

def get_serve_speed(myPnt):
    return myPnt.get_serve().get_initial_speed()

def get_serve_spin(myPnt):
        return myPnt.get_serve().get_initial_spin()

def get_return_speed(myPnt):
    return myPnt.get_return().get_initial_speed()

def get_return_spin(myPnt):
    return myPnt.get_return().get_initial_spin()

def get_return_pos(myPnt):
    return myPnt.get_serve().get_ball_next_position()

def get_return_flight_time(myPnt, fps = 25):
    return myPnt.get_return().get_num_frames() / fps

def get_distance_from_flight_line(myPnt):
    srvShot = myPnt.get_serve()
    retPnt = Point2D(srvShot.get_player_rec_position())
    fsPnt = Point2D(srvShot.get_ball_hit_position()[:2])
    lsPnt = Point2D(srvShot.get_ball_last_position()[:2])
    flightLn = Line(fsPnt, lsPnt)
    return flightLn.distance_to(retPnt) ### perpendicular distance to the flight line of the returner

def is_return_fore(myPnt):
    return myPnt.get_return().is_fore()

def is_return_back(myPnt):
    return myPnt.get_return().is_back()

def is_return_bounce_not_none(myPnt):
    x, y = myPnt.get_return().get_first_bounce_position()
    return x is not None and y is not None

def is_return_bounce_in(myPnt):
    return myPnt.get_return().is_fair(None)

def is_serve_from_fore(myPnt):
    if myPnt.is_serve_deuce():
        side = 'deuce'
    elif myPnt.is_serve_ad():
        side = 'ad'

    hand = myPnt.get_serve().hitPlayer.get_hand()

    if (hand == 'right' and side == 'deuce') or (hand == 'left' and side == 'ad'):
        return True
    elif (hand == 'right' and side == 'ad') or (hand == 'left' and side == 'deuce'):
        return False
    
def is_serve_out_wide(myPnt):
    return is_out_wide(myPnt.get_serve())
def is_serve_down_middle(myPnt):
    return is_down_middle(myPnt.get_serve())

def get_ret_dist_to_bsl(myPnt):
    retShot = myPnt.get_return()

    botLn = courtLines['Bottom Baseline']
    bot_y = botLn.leftPt.y
    if retShot is None:
        return None
    
    ret_y = retShot.get_first_bounce_position()[1]
    if ret_y is None:
        return None

    dist = bot_y - ret_y
    dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
    return dist_in_meters

def get_ret_dist_to_bh(myPnt):
    retShot = myPnt.get_return()
    botLn = botReg['Bottom']

    server_hand = retShot.recPlayer.get_hand()

    if server_hand == 'right':
        bhPnt = botLn.leftPt
    elif server_hand == 'left':
        bhPnt = botLn.rightPt
    else:
        return None

    bh_x = bhPnt.x

    if retShot is None:
        return None
    
    ret_x = retShot.get_first_bounce_position()[0]
    if ret_x is None:
        return None

    if server_hand == 'right':
        dist = ret_x - bh_x
    elif server_hand == 'left':
        dist = bh_x - ret_x
    
    dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
    return dist_in_meters

def get_ret_dist_to_opt(myPnt, meters_behind_baseline = 2):
    ### calculate the distance of the returning player to the optimal position to be in for the shot after the return.
    retShot = myPnt.get_return()
    retLn = retShot.get_flight_line()

    botLn = botReg['Bottom']
    topLn = topReg['Top']
    cntPt = courtLines.get_center_point()

    ### calculate where the server will meet the return. assumption: he meets it on the baseline
    intPt = retLn.weak_intersect(botLn)

    ### the optimal point is opposite the intersection of the return's ball flight with the baseline on the server's side
    mirrorLn = Line(intPt, cntPt)
    
    ### the optimal point is meters_behind_baseline -> the returner is not yet in offense, on average.
    pt_up = meters_behind_baseline * const.SCALE * const.METER_TO_FOOT
    recLn = topLn.up(pt_up)
    optPt = mirrorLn.weak_intersect(recLn)

    ### now, check where the returner actually is, at the time of the return
    retPt = Point2D(retShot.get_player_hit_position())
    dist = retPt.distance_to(optPt) 

    dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
    return dist_in_meters

# def get_ret_dist_to_opt_adaptive(myPnt, meters_behind_bounce = 2, meters_behind_baseline = 2):
#     ### calculate the distance of the returning player to the optimal position to be in for the shot after the return.
#     retShot = myPnt.get_return()
#     a, b = retShot.get_ball_hit_position(), retShot.get_first_bounce_position()
#     aPt, bPt = Point2D(a[:2]), Point2D(b)
#     retLn = Line(aPt, bPt) ### here, the flight line ends at bounce

#     pt_up = meters_behind_baseline * const.SCALE * const.METER_TO_FOOT

#     hitLn = Line(bPt.down(pt_up).left(10), bPt.down(pt_up).right(10)) ### this is the height at which the ball is assumed to be met



#     botLn = botReg['Bottom']
#     topLn = topReg['Top']
#     cntPt = courtLines.get_center_point()

#     ### calculate where the server will meet the return. assumption: he meets it on the baseline
#     intPt = retLn.weak_intersect(botLn)

#     ### the optimal point is opposite the intersection of the return's ball flight with the baseline on the server's side
#     mirrorLn = Line(intPt, cntPt)
    
#     ### the optimal point is meters_behind_baseline -> the returner is not yet in offense, on average.
#     pt_up = meters_behind_baseline * const.SCALE * const.METER_TO_FOOT
#     recLn = topLn.up(pt_up)
#     optPt = mirrorLn.weak_intersect(recLn)

#     ### now, check where the returner actually is, at the time of the return
#     retPt = Point2D(retShot.get_player_hit_position())
#     dist = retPt.distance_to(optPt) 

#     dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
#     return dist_in_meters

def get_ret_dist_to_opt_with_points(rastArr_2D, ret_ball_pos, ret_player_pos, meters_behind_baseline = 2):
    ### get_ret_dist_to_opt, but with whole array.

    meters_behind_baseline = 2
    pt_up = meters_behind_baseline * const.SCALE * const.METER_TO_FOOT
    bot_y = botReg['Bottom'].leftPt.y
    top_y = topReg['Top'].up(pt_up).leftPt.y
    cnt_x, cnt_y = courtLines.get_center_point().coords

    ### intersection with bot line
    flightVec = rastArr_2D - ret_ball_pos ### essentially, reset origin to ret_ball_pos
    dVec = flightVec[:, 1] / flightVec[:, 0]  ### then, all we need is the gradient of the flight vector
    bot_x_unadj = (bot_y - ret_ball_pos[1]) / dVec ### in this coordinate system (with origin at ret_ball_pos), bot_x = ...
    bot_x = bot_x_unadj + ret_ball_pos[0] ### transform back
    botArr = np.stack([bot_x, np.full_like(bot_x, fill_value = bot_y)], axis = 1)

    ### intersection with top line (through center)
    mirrorVec = botArr - (cnt_x, cnt_y) ### reset origin to center of court
    dVec = mirrorVec[:, 1] / mirrorVec[:, 0]
    top_x_unadj = (top_y - cnt_y) / dVec
    top_x = top_x_unadj + cnt_x
    topArr = np.stack([top_x, np.full_like(top_x, fill_value = top_y)], axis = 1)

    dist = ((ret_player_pos - topArr) ** 2).sum(axis = 1) ** 0.5
    dist_in_meters = dist / const.SCALE * const.FOOT_TO_METER
    return dist_in_meters

def return_is_in_net(myPnt):
    return myPnt.get_return().is_in_net()

###############################
###### POINTS FUNCTIONS #######
###############################

def get_winners_from_score(retPnts, allPnts):
    winner_codes = []
    for myPnt in retPnts:
        match_idx, point_idx = myPnt.get_match_index(), myPnt.get_point_index()
        nextPnt = allPnts.get(match_idx, point_idx + 1)

        winner = myPnt.get_winner_from_score(nextPnt, verbose = True)

        if winner is None:
            winner_codes.append(0)
        elif winner == myPnt.get_server().get_name():
            winner_codes.append(1)
        elif winner == myPnt.get_returner().get_name():
            winner_codes.append(-1)

    return winner_codes

def get_psm_var_dict(retPnts):
    
    courtLines = CourtLines.make()
    x, y = courtLines['Top Baseline'].center().coords ### note that direction has been enforced -> Bot to Top
    x, y = x / const.SCALE * const.FOOT_TO_METER, y / const.SCALE * const.FOOT_TO_METER

    ### Dependent Variable: chipped or topspin?
    def is_chipped(myPnt):
        return myPnt.get_return().is_slice_visual()
    chipArr = retPnts.apply(is_chipped)

    ### Independent Variables:
    # a) Serve Speed
    
    speedArr = retPnts.apply(get_serve_speed)
    # speedArr = np.log(speedArr)

    # b) Location of Return
    locArr = retPnts.apply(get_return_pos) / const.SCALE * const.FOOT_TO_METER

    # b1) Width of the return -> relative to center line
    widthArr = np.abs(locArr[:, 0] - x) ### absolute, because we control for side by looking at forehand/backhand
    # b2) Depth of return
    depthArr = -(locArr[:, 1] - y) ### negative -> behind baseline, positive, in front -> so we turn it around!!
    # b3) Height of Return
    heightArr = locArr[:, 2]

    # c) Distance of Returner from Flight Line
    distArr = retPnts.apply(get_distance_from_flight_line) / const.SCALE * const.FOOT_TO_METER

    # d) Forehand/Backhand
    backArr = retPnts.apply(is_return_back)

    # e) spin
    spinArr = retPnts.apply(get_serve_spin)
    spinArr = spinArr.clip(min = -1000, max = 5000) ### take it easy on the backspin... no such serve

    varDict = { 'is_chipped': chipArr,
                'serve_speed': speedArr,
                'serve_spin': spinArr,
                'serve_width': widthArr,
                'serve_depth': depthArr,
                'serve_height': heightArr,
                'serve_dist': distArr,
                'ret_is_back': backArr}

    return varDict

def calculate_attribution(myFrm, chip_col = 'is_chipped', outcome_col = 'ret_wins', 
                          bounce_col = 'must_have_bounced_in', weight_col = None):
    
    if weight_col is None:
        myFrm = myFrm[[chip_col, outcome_col, bounce_col]]
        myFrm.columns = ['is_chipped', 'outcome', 'bounce']
        myFrm['weight'] = 1
    else:
        myFrm = myFrm[[chip_col, outcome_col, bounce_col, weight_col]]
        myFrm.columns = ['is_chipped', 'outcome', 'bounce', 'weight']
    
    myFrm['weight_outcome'] = myFrm['outcome'] * myFrm['weight']
    myFrm['weight_bounce'] = myFrm['bounce'] * myFrm['weight']

    grp = myFrm.groupby('is_chipped')
    cond_grp = myFrm[myFrm['bounce']].groupby('is_chipped')

    total_avgs = grp['weight_outcome'].sum() / grp['weight'].sum()
    make_avgs = grp['weight_bounce'].sum() / grp['weight'].sum()
    cond_avgs = cond_grp['weight_outcome'].sum() / cond_grp['weight'].sum()

    total_del = total_avgs[False] - total_avgs[True]
    make_del = make_avgs[False] - make_avgs[True]
    cond_del = cond_avgs[False] - cond_avgs[True]

    comb_contrib = make_del * cond_del
    make_contrib = make_del * cond_avgs[True]
    cond_contrib = cond_del * make_avgs[True]
    total_contrib = comb_contrib + make_contrib + cond_contrib

    assert round(total_del, 3) == round(total_contrib, 3)
    
    comb_share, make_share, cond_share = comb_contrib / total_contrib, make_contrib / total_contrib, cond_contrib / total_contrib

    adj_make_share = make_share / (make_share + cond_share)
    adj_cond_share = cond_share / (make_share + cond_share)

    print('Total Averages:') 
    print(f'    {chip_col} = False: {total_avgs[False]:.3f}')
    print(f'    {chip_col} = True : {total_avgs[True]:.3f}')

    print('Make Averages:')
    print(f'    {chip_col} = False: {make_avgs[False]:.3f}')
    print(f'    {chip_col} = True : {make_avgs[True]:.3f}')

    print('Conditional Averages:')
    print(f'    {chip_col} = False: {cond_avgs[False]:.3f}')
    print(f'    {chip_col} = True : {cond_avgs[True]:.3f}')

    print('Delta Attribution: ')
    print(f'    Make Share: {make_share:.3f}')
    print(f'    Cond Share: {cond_share:.3f}')
    print(f'    Comb Share: {comb_share:.3f}')

    print('Adjusted Delta Attribution')
    print(f'    Make Share: {adj_make_share:.3f}')
    print(f'    Cond Share: {adj_cond_share:.3f}')

    chipDict = {'total': total_avgs[True], 'make': make_avgs[True], 'cond': cond_avgs[True]}
    topsDict = {'total': total_avgs[False], 'make': make_avgs[False], 'cond': cond_avgs[False]}
    attrDict = {'d_make': make_share, 'd_cond': cond_share, 'd_comb': comb_share, 
                'd_make_adj': adj_make_share, 'd_cond_adj': adj_cond_share}

    return chipDict, topsDict, attrDict


def rectify_shots(myShots):
    myShots_lh_receiver = myShots.filter(lambda shot: shot.recPlayer.get_hand() == 'left').flip(across_middle = True, across_net = False)
    myShots_rh_receiver = myShots.filter(lambda shot: shot.recPlayer.get_hand() == 'right')
    myShots_rectified = myShots_lh_receiver.join(myShots_rh_receiver).sort()
    return myShots_rectified

def get_X_y_w(regFrm, exog, endog, weight_col = None, add_constant = True):
    regFrm = regFrm[regFrm.winner_code != 0]

    if weight_col is None:
        regFrm_sub = regFrm[[*exog, endog]]
    else:
        regFrm_sub = regFrm[[*exog, endog, weight_col]]

    nan_filter = ~regFrm_sub.isna().any(axis = 1)
    regFrm_valid = regFrm_sub[nan_filter]

    xFrm, yFrm = regFrm_valid[exog].astype(float), regFrm_valid[endog].astype(float)
    if add_constant:
        xFrm['const'] = 1.0
    
    if weight_col is None:
        wFrm = None
    else:
        wFrm = regFrm_valid[weight_col].astype(float)

    return xFrm, yFrm, wFrm

def save_model_summary_as_png(fitted_model, save_path, save_name):
    summary_text = fitted_model.summary().as_csv()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.01, 0.05, summary_text, fontfamily='monospace', fontsize=14)
    ax.axis('off')
    plt.tight_layout()

    full_path = os.path.join(save_path, save_name + '.png')
    plt.savefig(full_path, dpi=300, bbox_inches='tight')

def make_and_save_model(regressor_frame, regressors, save_path, save_name,
                        endog_col = 'ret_wins', weight_col = None, verbose = True):
    X, y, w = get_X_y_w(regressor_frame, regressors, endog_col, add_constant = True, weight_col = weight_col)
    glmReg_win = sm.GLM(y, X, family = sm.families.Binomial(), freq_weights = w).fit()
    summary = glmReg_win.summary()

    if verbose: 
        print(summary)
    
    save_model_summary_as_png(glmReg_win, save_path = save_path, save_name = save_name)

    return glmReg_win

def retrieve_X_y(fitted_model, regressors):
    import pandas as pd
    X = pd.DataFrame(fitted_model.model.exog, columns = [*regressors, 'const'])
    y = pd.Series(fitted_model.model.endog, name = 'ret_win')
    return X, y

def get_relative_raster_array(absolute_raster_array, origin_point):
    rel_rastArr = np.copy(absolute_raster_array)
    rel_rastArr[:, 0] = rel_rastArr[:, 0] - origin_point.x
    rel_rastArr[:, 1] = origin_point.y - rel_rastArr[:, 1] 
    rel_rastArr = rel_rastArr[:, :2]
    rel_rastArr = rel_rastArr / const.SCALE * const.FOOT_TO_METER
    return rel_rastArr

def get_color_and_alpha_array_from_predictions(glm_model, regFrm, relative_to = None,
                                               zero_color = const.BGR['intense_red'], one_color = const.BGR['leafy_green'],
                                               force_zero_to_middle = True, alpha_exponent = 1):
    pred_winSer = glm_model.predict(regFrm)
    pred_winArr = pred_winSer.values[:, np.newaxis]

    if relative_to is not None:
        pred_winArr = pred_winArr - relative_to

    minimum, maximum = pred_winArr.min(), pred_winArr.max()
    
    if force_zero_to_middle:
        pred_winArr_renorm = np.where(pred_winArr < 0, - pred_winArr / minimum, pred_winArr / maximum)
        pred_winArr_renorm = (pred_winArr_renorm + 1) / 2 ### from [-1, 1] to [0, 1]
    else:
        pred_winArr_renorm = (pred_winArr - minimum) / (maximum - minimum)

    colArr = pred_winArr_renorm * one_color + (1 - pred_winArr_renorm) * zero_color
    alArr = np.abs(pred_winArr_renorm - 0.5) * 2
    alArr = alArr ** alpha_exponent

    return colArr, alArr

def make_mode_lines(mode_angles, line_len = 100, x_ax = Point2D((1, 0, 0)), y_ax = Point2D((0, 1, 0)), orig = courtLines.get_center_point()):
    from ATMANpy.types.Shapes3D import Line3D, Lines3D

    mode_lines = []
    for ang in mode_angles:
        rad = ang * const.ANGLE_TO_RAD
        mode_x, mode_y = np.cos(rad) * line_len, np.sin(rad) * line_len
        mode_vec = x_ax * mode_x + y_ax * mode_y
        modeLn = Line3D(orig, mode_vec + orig)
        mode_lines.append(modeLn)

    return Lines3D(mode_lines)