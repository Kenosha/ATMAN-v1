# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:44:04 2022



@author: gianl
"""

######## GET SCHEMA ########
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines, Region
from ATMANpy.funcs.line_funcs import fit_frame_to_lines
schemaLines = CourtLines.make().add_singles_netline()

###### GET LINE FUNC #######
from ATMANpy.funcs.court_funcs import find_courtlines

# from ATMANpy.types.PutDispatcher import put

def draw(featList, add_text = None, textRegion = None):
    
    THICKNESS = 2
    
    PLAYER_COLOR = (0, 0, 255)
    BALL_COLOR = (0, 255, 255)
    LINE_COLOR = (255, 0, 0)
    TEXT_COLOR = (0, 255, 0)
    OTHER_COLOR = (255, 0, 255)
    
    SECONDARY_COLOR = (255, 255, 0)
    INTERPOLATED_BALL_COLOR = (0, 180, 255)
    
    FRAME_NUM_LOC = (1800, 50)
    
    FRM_NUM_ALIASES = ['frmNum', 'frameNum', 'frmNumber', 'frameNumber']
    
    NONPLAY_BRIGHTNESS_FACTOR = 0.4
    
    def drawFunc(frmDict, featDict, frameNum = None, textRegion = textRegion):
        
        cFrm = frmDict['cFrm']
        sendFrm = cFrm
        
        ### handle as text
        textFeatures = [featStr for featStr in featList if type(featDict[featStr]) in [int, str]]
        otherFeatures = [featStr for featStr in featList if not featStr in textFeatures]
        
        #### HANDLE ALL TEXT ###
        textLines = []
        if textRegion is None:
            textRegion = Region.from_points(Point2D((50, 50)), Point2D((200, 200)))
            
        if not add_text is None:
            textLines.append(add_text)
        
        for featStr in textFeatures:
            feat = featDict[featStr]
            if isinstance(feat, int):
                feat = f'{featStr}: {feat}'
            textLines.append(feat)
        
        if not len(textLines) == 0:
            text_distance = int(textRegion['Left'].length() / len(textLines))
            
            for i, text in enumerate(textLines):
                text_coords = textRegion.p1.down(text_distance * i).coords
                sendFrm = sendFrm.put(text, thickness = THICKNESS, color = TEXT_COLOR,
                                      where = text_coords)
        
        #### ADD ALL OTHER FEATURES #####
        for featStr in otherFeatures:
            
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
                
            if feat is None:
                continue
               
            ### DEAL WITH BOOLS ###
            if isinstance(feat, bool):
                if featStr == 'isPlay':
                    if not feat: ### if the frame is not play...
                        sendFrm = sendFrm.dim(NONPLAY_BRIGHTNESS_FACTOR) ### we make the frame darker!
                continue ### have to continue since bools are also integers...
                
            ### DEAL WITH INT AND STR ###
            if isinstance(feat, int):
                dont_add_text = ['size']
                
                putBool = True
                for string in dont_add_text:
                    if string in featStr.lower():
                        putBool = False
    
                if putBool:
                    ### turn an integer to a string
                    ### note that this will trigger the isinstance(feat, str) below!
                    feat = f'{featStr}: {feat}' 
            
            ### DEAL WITH FEATURE ###
            if 'secondary' in featStr:
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = SECONDARY_COLOR)
            elif featStr.startswith('player'):
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = PLAYER_COLOR)
            elif featStr == 'ball':
                  
                try:
                    ball_color = featDict['ballColor']
                except:
                    try:
                        detectBool = featDict['ballDetected'] 
                    except:
                        ball_color = BALL_COLOR
                    else:
                        ball_color = BALL_COLOR if detectBool else INTERPOLATED_BALL_COLOR
                    
                try:
                    sizeInt = featDict['ballSize']
                except:
                    radius = None
                    thickness = THICKNESS
                else:
                    radius = sizeInt
                    thickness = sizeInt
                  
                sendFrm = sendFrm.put(feat, thickness = thickness, color = ball_color, radius = radius)
                    
            elif featStr == 'courtLines':
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = LINE_COLOR)

            else:
                if 'good' in featStr:
                    color = (0, 255, 0)
                elif 'bad' in featStr:
                    color = (0, 0, 255)
                else:
                    color = OTHER_COLOR
                sendFrm = sendFrm.put(feat, thickness = THICKNESS, color = color)
                
        return sendFrm
    
    return drawFunc




def split_and_check(boolArr, left_prop = 1, overall_prop = 2, 
                    alt_boolArr = None,
                    curr_x_offset = 0, curr_y_offset = 0,
                    alt_x_offset = 0, alt_y_offset = 0):
    ### this functions finds a large (not the largest, but good enough) rectangle
    ### that has all boolean values set to one.
    ### in this way, we can find an empty area in a frame, by feeding this function
    ### an array with True if the value is 0 or (0, 0, 0)
    ### for different outcomes, change left_prop and overall_prop to other integers, left_prop < overall_prop.
    
    from ATMANpy.funcs.util_funcs import split_array_in_two
    
    rows, cols = boolArr.shape
    
    if boolArr.sum() == rows * cols:
        p1, p2 = (curr_x_offset, curr_y_offset), (curr_x_offset + rows, curr_y_offset + cols)
        p1 = p1[1], p1[0] ### switch, because in frames, x and y are switched!
        p2 = p2[1], p2[0] ### switch, because in frames, x and y are switched!
        return Region.from_points(Point2D(p1), Point2D(p2))
    else:
        if rows > cols: ### split into upper and lower
            axis = 0
            add_x_offset = left_prop * rows // overall_prop
            add_y_offset = 0
        else: ### split into left and right
            axis = 1
            add_x_offset = 0
            add_y_offset = left_prop * cols // overall_prop
    
    a, b = split_array_in_two(boolArr, left_prop = left_prop, overall_prop = overall_prop, axis = axis)

    if alt_boolArr is None:
        if a.sum() >= b.sum():
            return split_and_check(a, alt_boolArr = b, 
                                   left_prop = left_prop, overall_prop = overall_prop, 
                                   curr_x_offset = curr_x_offset, 
                                   curr_y_offset = curr_y_offset,
                                   alt_x_offset = curr_x_offset + add_x_offset, 
                                   alt_y_offset = curr_y_offset + add_y_offset)
        elif b.sum() >= a.sum():
            return split_and_check(b, alt_boolArr = a, 
                                   left_prop = left_prop, overall_prop = overall_prop, 
                                   curr_x_offset = curr_x_offset + add_x_offset, 
                                   curr_y_offset = curr_y_offset + add_y_offset,
                                   alt_x_offset = curr_x_offset, 
                                   alt_y_offset = curr_y_offset)
    elif a.sum() >= b.sum() >= alt_boolArr.sum():
        return split_and_check(a, alt_boolArr = b, 
                               left_prop = left_prop, overall_prop = overall_prop, 
                               curr_x_offset = curr_x_offset, 
                               curr_y_offset = curr_y_offset,
                               alt_x_offset = curr_x_offset + add_x_offset, 
                               alt_y_offset = curr_y_offset + add_y_offset)
    elif a.sum() >= alt_boolArr.sum() >= b.sum():
        return split_and_check(a, alt_boolArr = alt_boolArr, 
                               left_prop = left_prop, overall_prop = overall_prop, 
                               curr_x_offset = curr_x_offset, 
                               curr_y_offset = curr_y_offset,
                               alt_x_offset = alt_x_offset, 
                               alt_y_offset = alt_y_offset)
    elif b.sum() >= a.sum() >= alt_boolArr.sum():
        return split_and_check(b, alt_boolArr = a, 
                               left_prop = left_prop, overall_prop = overall_prop, 
                               curr_x_offset = curr_x_offset + add_x_offset, 
                               curr_y_offset = curr_y_offset + add_y_offset,
                               alt_x_offset = curr_x_offset, 
                               alt_y_offset = curr_y_offset)
    elif b.sum() >= alt_boolArr.sum() >= a.sum():
        return split_and_check(b, alt_boolArr = alt_boolArr, 
                               left_prop = left_prop, overall_prop = overall_prop, 
                               curr_x_offset = curr_x_offset + add_x_offset, 
                               curr_y_offset = curr_y_offset + add_y_offset,
                               alt_x_offset = alt_x_offset, 
                               alt_y_offset = alt_y_offset)
    elif alt_boolArr.sum() >= a.sum() >= b.sum():
        return split_and_check(alt_boolArr, alt_boolArr = a, 
                               left_prop = left_prop, overall_prop = overall_prop, 
                               curr_x_offset = alt_x_offset, 
                               curr_y_offset = alt_y_offset,
                               alt_x_offset = curr_x_offset, 
                               alt_y_offset = curr_y_offset)
    elif alt_boolArr.sum() >= b.sum() >= a.sum():
        return split_and_check(alt_boolArr, alt_boolArr = b, 
                               left_prop = left_prop, overall_prop = overall_prop, 
                               curr_x_offset = alt_x_offset, 
                               curr_y_offset = alt_y_offset,
                               alt_x_offset = curr_x_offset + add_x_offset, 
                               alt_y_offset = curr_y_offset + add_y_offset)
   





