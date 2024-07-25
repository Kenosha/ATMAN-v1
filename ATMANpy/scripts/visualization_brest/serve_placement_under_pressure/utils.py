import numpy as np
import scipy
from ATMANpy.auxil.court import courtLines
from ATMANpy.types.Points import Point2D

from ATMANpy.types.Overlays import Legend
import ATMANpy.auxil.constants as const

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

def make_stat_string(loShots, hiShots, alpha = 0.05, lo_desc = 'Low Pressure', hi_desc = 'High Pressure'):
    ######## Tests for similarity of distribution #########
    low_x = np.array([find_intersection_with_service_line(aShot) for aShot in loShots], dtype = float)[:, 0]
    hi_x = np.array([find_intersection_with_service_line(aShot) for aShot in hiShots], dtype = float)[:, 0]

    ks_res = scipy.stats.kstest(low_x, hi_x)           ### kolmogorov-smirnov test (2-sample)
    ad_res = scipy.stats.anderson_ksamp([low_x, hi_x]) ### anderson darling test (2-sample)
    ks_p, ad_p = ks_res.pvalue, ad_res.pvalue

    if ks_p >= alpha and ad_p >= alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions \n can not be shown to be statistically different at a = {alpha}.'
    elif ks_p < alpha and ad_p < alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions \n are statistically different at a = {alpha}.'
    elif ks_p < alpha and ad_p >= alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions \n are statistically different at a = {alpha} \n only according to the Kolmogorov-Smirnov test.'
    elif ks_p >= alpha and ad_p < alpha:
        textStr = f'{hi_desc} and {lo_desc} distributions \n are statistically different at a = {alpha} \n only according to the Anderson-Darling test.'

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
    
def is_bounce_in_ad(myShot):
    return is_bounce_in_region(myShot, adReg_enlarged)

def is_bounce_in_deuce(myShot):
    return is_bounce_in_region(myShot, deuceReg_enlarged)

def make_legend_frame(player_name, handedness):
    ## legend parameters
    text_size = 1
    thickness = 1
    offset = 0.5
    color = const.BGR['bright_grey']

    myLeg = Legend.make_empty(resolution = 100, header_offset = 3.25, legend_dimensions = (350, 200), header_thickness = 2)

    l1 = {'text_string_shape': 'Player:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
        'text_string': player_name,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}        
    l2 = {'text_string_shape': 'Handedness:', 'text_color_shape': color, 'text_size_shape': text_size, 'text_thickness_shape': thickness, 'text_offset_shape': offset,
            'text_string': handedness,  'text_color': color, 'text_size': text_size, 'text_thickness': thickness, 'text_offset': 3 * offset}
    
    myLeg = myLeg.add(l1).add(l2)

    legFrm = myLeg.render()

    return legFrm