# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:44:04 2022



@author: gianl
"""

######## GET SCHEMA ########
from ATMANpy.types.Lines import CourtLines
from ATMANpy.funcs.court_funcs import SCHEMA_SIZE, SCHEMA_MARGIN
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make(size = SCHEMA_SIZE, margin = SCHEMA_MARGIN).add_singles_netline()

###### GET LINE FUNC #######
from ATMANpy.funcs.court_funcs import find_courtlines

# from ATMANpy.types.PutDispatcher import put

def draw(featList, add_text = None):
    
    THICKNESS = 2
    
    PLAYER_COLOR = (0, 0, 255)
    BALL_COLOR = (0, 255, 255)
    LINE_COLOR = (255, 0, 0)
    TEXT_COLOR = (0, 255, 0)
    OTHER_COLOR = (255, 255, 0)
    
    FRAME_NUM_LOC = (1800, 50)
    OTHER_TXT_LOC = (50, 50)
    TEXT_DISTANCE = 30
    
    FRM_NUM_ALIASES = ['frmNum', 'frameNum', 'frmNumber', 'frameNumber']
    
    NONPLAY_BRIGHTNESS_FACTOR = 0.4
    
    def drawFunc(frmDict, featDict, frameNum = None):
        
        cFrm = frmDict['cFrm']
        sendFrm = cFrm
        
        string_space_counter = 0
        
        #### ADD ADDITIONAL TEXT ###
        if not add_text is None:
            sendFrm = sendFrm.put(add_text, thickness = THICKNESS, color = TEXT_COLOR,
                                  where = OTHER_TXT_LOC)
                
            string_space_counter += 1
            
        
        #### ADD ALL FEATURES #####
        for featStr in featList:
            
            ### FRAME NUMBER ###
            if featStr in FRM_NUM_ALIASES:
                sendFrm = sendFrm.put(str(int(frameNum)), thickness = THICKNESS, color = TEXT_COLOR, 
                                      where = FRAME_NUM_LOC)
                continue
            
            ### CHECK IF FEATURE EXISTS ###
            try:
                feat = featDict[featStr]
            except KeyError:
                if featStr == 'courtLines' and 'H' in featDict: ### we can make courtlines!
                    feat = find_courtlines(featDict['H'], schemaLines)
                else:
                    print(f'Feature {featStr} not found, will not be drawn.')
                    continue
               
            ### DEAL WITH BOOLS ###
            if isinstance(feat, bool):
                if featStr == 'isPlay':
                    if not feat: ### if the frame is not play...
                        sendFrm = sendFrm.dim(NONPLAY_BRIGHTNESS_FACTOR) ### we make the frame darker!
                continue ### have to continue since bools are also integers...
                
            ### DEAL WITH INT AND STR ###
            if isinstance(feat, int):
                ### turn an integer to a string
                ### note that this will trigger the isinstance(feat, str) below!
                feat = f'{featStr}: {feat}' 
            
            if isinstance(feat, str):
                
                ### figure out the location of text: begings in OTHER_TEXT_LOC, then goes down.
                text_coords = OTHER_TXT_LOC[0], OTHER_TXT_LOC[1] + TEXT_DISTANCE * string_space_counter
            
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = TEXT_COLOR,
                                      where = text_coords)
                
                string_space_counter += 1
                continue
            
            ### DEAL WITH FEATURE ###
            if featStr.startswith('player'):
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = PLAYER_COLOR)
            elif featStr == 'ball':
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = BALL_COLOR)
            elif featStr == 'courtLines':
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = LINE_COLOR)
            else:
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = OTHER_COLOR)
                
        return sendFrm
    
    return drawFunc
