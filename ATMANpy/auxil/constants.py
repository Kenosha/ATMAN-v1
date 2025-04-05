# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 22:36:11 2023

@author: gianl
"""

import numpy as np

### UNIT CONVERSIONS ###
FOOT_TO_METER = 0.3048
METER_TO_FOOT = 1 / FOOT_TO_METER

KM_TO_M = 1000
M_TO_KM = 1 / KM_TO_M

HOUR_TO_MIN = 60
MIN_TO_HOUR = 1 / HOUR_TO_MIN
MIN_TO_SEC = 60
SEC_TO_MIN = 1 / MIN_TO_SEC

KPH_TO_MPS = KM_TO_M / (HOUR_TO_MIN * MIN_TO_SEC)
MPS_TO_KPH = (HOUR_TO_MIN * MIN_TO_SEC) / KM_TO_M

RAD_TO_ANGLE = 180 / np.pi
ANGLE_TO_RAD = np.pi / 180

### SCALE OF MODEL COURT ###
SCALE = 8
MARGIN = 150

### DIMENSIONS IN FEET OF MODEL COURT ###

## court dimensions in feet
COURT_WIDTH = 36
SIDE_LENGTH = 39
DOUBLES_ALLEY_WIDTH = 4.5
SINGLES_WIDTH = COURT_WIDTH - 2 * DOUBLES_ALLEY_WIDTH
SERVICE_BOX_LENGTH = 21
SERVICE_BOX_WIDTH = SINGLES_WIDTH / 2
NET_BEYOND_LINE = 3
NET_BEYOND_POST = 3.5
NET_HEIGHT_MIDDLE = 3
BASELINE_WIDTH = 0.5 # 1/3: the minimum baseline width is a third of a foot
OTHERLINE_WIDTH = 0.25 # 1/6: the minimum width of other lines is a sixth of a foot

### COLORS ####

### these colors look good on black according to the internet ###
RGB = {'ball_yellow': (223, 255, 79),
       'soft_yellow': (254, 232, 81),
       'fair_yellow': (249, 223, 48),
       'taxi_yellow': (240, 233, 149),
       
       'minty_green': (53, 185, 151),
       'grass_green': (145, 193, 62),
       'leafy_green': (34, 175, 75),
       'bluey_green': (8, 185, 165),
       
       'strong_blue': (15, 137, 202),
       'bright_blue': (29, 162, 220),
       'marine_blue': (26, 87, 143),
       
       'salmon_pink': (240, 85, 97),
       'strong_pink': (238, 34, 74),
       
       'rose_orange': (238, 87, 61),
       'sign_orange': (143, 87, 26),
    
       'light_brown': (203, 119, 35),

       'intense_red': (238, 37, 36),

       'bright_grey': (238, 238, 238),
       'smokey_grey': (150, 150, 150),
       'darker_grey': (30, 30, 30)
       }

BGR = {name: (val[2], val[1], val[0]) for name, val in RGB.items()}


MISSING_SENTINEL_VALUE = 9999 ### if an integer is missing, we add this value. 
                                # can't be np.nan, because np.nan is not an integer.
  