# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 18:19:19 2022

@author: gianl
"""

import numpy as np

######## GET SCHEMA ########
from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()
schemaRegion = schemaLines.get_court_region()
schemaFrm = fit_frame_to_lines(schemaLines)
    

### GET STREAM FEATURES ###
### VIDEO ######
from ATMANpy.funcs.util_funcs import get_video   
VIDEO_ID = 'pqxAmUWIoKc'
cap = get_video(VIDEO_ID)
################

### STORENAME ###
STORE_NAME = '10to100_ok2'
#################

from ATMANpy.funcs.feature_funcs import load_stored_features
stream_features = load_stored_features(VIDEO_ID, STORE_NAME)

#%% COLLECT MORE SHAPES IF NEEDED 

collect_more_shapes = False
if collect_more_shapes:
    ## collect a few good player shapes, so they can aid in calibration
    
    from ATMANpy.types.Frames import GreyFrame
    
    shapes_needed = 5
    shapes_top = []
    shapes_bot = []
    
    while len(shapes_top) < shapes_needed:
        randframe = np.random.choice(list(stream_features.keys()))
        gFrm = GreyFrame.from_video(cap, randframe)
        featDict = stream_features[randframe]
    
        playerStr = 'playerTop'        
        player = featDict[playerStr]
        H = featDict['H']
        
        gFrm.to_BGR().put(player).show()
        
        key = input(f'{playerStr} shape: looks good? press y/n: ')
        if key == 'y':
            height = input(f'{playerStr}: How Tall?')
            
            shapes_top.append((player, H, height))
        
    
    while len(shapes_bot) < shapes_needed:
        randframe = np.random.choice(list(stream_features.keys()))
        gFrm = GreyFrame.from_video(cap, randframe)
        featDict = stream_features[randframe]
    
        playerStr = 'playerBot'        
        player = featDict[playerStr]
        H = featDict['H']
        
        gFrm.to_BGR().put(player).show()
        
        key = input(f'{playerStr} shape: looks good? press y/n: ')
        if key == 'y':
            height = input(f'{playerStr}: How Tall?')
            
            shapes_bot.append((player, H, height))
    
    
    player_coords_world = []
    player_coords_image = []
    for player, H, height_meter in shapes_top + shapes_bot:
        
        plReg = player.enclosing_region()
    
        top = plReg['Top'].center() ### this is the point we will actually use as pixel representation
        bottom = plReg['Bottom'].center() ### this point we need to transform with H, to understand where on the schema we are.
    
        bottom_world = bottom.apply_homography(np.linalg.inv(H))
        
        SCALE_FACTOR = 8
        height_feet = float(height_meter) * 3.281 * SCALE_FACTOR
        
        coords_world = [*bottom_world.coords, height_feet]
        coords_image = list(top.coords)
        
        player_coords_world.append(coords_world)
        player_coords_image.append(coords_image)
else:
    player_coords_world = None
    player_coords_image = None

#%% ##### CAMERA CALIBRATION ######

index = 93935

from ATMANpy.types.Frames import GreyFrame
gFrm = GreyFrame.from_video(cap, index)

H = stream_features[index]['H']

from ATMANpy.funcs.court_funcs import calibrate_camera
P = calibrate_camera(gFrm, schemaLines, H, 
                     add_coords_world = player_coords_world,
                     add_coords_image = player_coords_image)










