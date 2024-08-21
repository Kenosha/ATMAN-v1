
import numpy as np

import ATMANpy.auxil.constants as const

from ATMANpy.types.Lines import CourtLines, Line
from ATMANpy.types.Points import Point2D

#######################################
######### COURT REGIONS ###############
#######################################
courtLines = CourtLines.make().add_singles_netline()
botReg = courtLines.get_half_region(side = 'Bottom', width = 'Singles')

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

def get_distance_from_flight_line(myPnt):
    srvShot = myPnt.get_serve()
    retPnt = Point2D(srvShot.get_player_rec_position())
    fsPnt = Point2D(srvShot.get_ball_hit_position()[:2])
    lsPnt = Point2D(srvShot.get_ball_last_position()[:2])
    flightLn = Line(fsPnt, lsPnt)
    return flightLn.distance_to(retPnt) ### perpendicular distance to the flight line of the returner

def is_return_fore(myPnt):
    return myPnt.get_return().is_fore()

def is_return_bounce_not_none(myPnt):
    x, y = myPnt.get_return().get_first_bounce_position()
    return x is not None and y is not None

def is_return_bounce_in(myPnt):
    return myPnt.get_return().is_fair(None)

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
    srvShot = myPnt.get_serve()
    retShot = myPnt.get_return()
    botLn = botReg['Bottom']

    if srvShot.hitPlayer.get_hand() == 'right':
        bhPnt = botLn.leftPt
    elif retShot.recPlayer.get_hand() == 'left':
        bhPnt = botLn.rightPt
    else:
        return None

    bh_x = bhPnt.x

    if retShot is None:
        return None
    
    ret_x = retShot.get_first_bounce_position()[0]
    if ret_x is None:
        return None

    dist = ret_x - bh_x
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
    foreArr = retPnts.apply(is_return_fore)

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
                'ret_is_fore': foreArr}

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

def rectify_shots(myShots):
    myShots_lh_receiver = myShots.filter(lambda shot: shot.recPlayer.get_hand() == 'left').flip(across_middle = True, across_net = False)
    myShots_rh_receiver = myShots.filter(lambda shot: shot.recPlayer.get_hand() == 'right')
    myShots_rectified = myShots_lh_receiver.join(myShots_rh_receiver).sort()
    return myShots_rectified

def get_X_y_w(regFrm, exog, endog, weight_col = 'weight_topspin', add_constant = True):
    regFrm = regFrm[regFrm.winner_code != 0]
    regFrm_sub = regFrm[[*exog, endog, weight_col]]

    nan_filter = ~regFrm_sub.isna().any(axis = 1)
    regFrm_valid = regFrm_sub[nan_filter]

    xFrm, yFrm = regFrm_valid[exog], regFrm_valid[endog]
    if add_constant:
        xFrm['const'] = 1
    
    wFrm = regFrm_valid[weight_col]

    return xFrm.astype(float), yFrm.astype(float), wFrm.astype(float)

def get_relative_raster_array(absolute_raster_array, origin_point):
    rel_rastArr = np.copy(absolute_raster_array)
    rel_rastArr[:, 0] = rel_rastArr[:, 0] - origin_point.x
    rel_rastArr[:, 1] = origin_point.y - rel_rastArr[:, 1] 
    rel_rastArr = rel_rastArr[:, :2]
    rel_rastArr = rel_rastArr / const.SCALE * const.FOOT_TO_METER
    return rel_rastArr

def get_color_and_alpha_array_from_predictions(glm_model, regressor_array, relative_to = None,
                                               zero_color = const.BGR['intense_red'], one_color = const.BGR['leafy_green'],
                                               force_zero_to_middle = True, alpha_exponent = 1):
    pred_winArr = glm_model.predict(regressor_array)
    pred_winArr = pred_winArr[:, np.newaxis]

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

