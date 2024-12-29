"""

This module contains functions that take as FIRST argument a Shot, and then other arguments potentially.
They return a boolean always.

This way, we can use them with the .filter method of Shots, to search for specific Shots.

"""

import numpy as np

from ATMANpy.auxil.court import courtLines

def is_down_middle(myShot):
    from ATMANpy.types.Lines import Line
    x, _ = myShot.get_intersection_with_service_line()
    if x is None:
        return False

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, indDict = courtLines.get_service_line_and_indicator(side, which)

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
    x, _ = myShot.get_intersection_with_service_line()
    if x is None:
        return False

    assert myShot.is_serve()
    side = 'Top' if myShot.get_hit_side() == 'Bot' else 'Bottom'
    which = 'Ad' if myShot.is_serve_ad() else 'Deuce'
    srvLn, indDict = courtLines.get_service_line_and_indicator(side, which)

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
     