
import os
import time
import shlex

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### GET SCHEMA ###
from ATMANpy.types.Lines import CourtLines, NetLines

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.sf_funcs as sf
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.types.Streams import Stream, CachedStream, NCStream, NCDecapStream, DecapStream, StreamList, MulStream, RandStream, StillStream
from ATMANpy.types.Features import StreamFeature, StreamFeatures, BoolFeature, BoolFeatures, SFList, ConstFeature, ConstFeatures

from ATMANpy.types.Visual3D import Scene, Camera, View

from ATMANpy.types.Shapes3D import Point3D, Ball3D, Region3D, Line3D

from ATMANpy.types.Frames import MaskFrame, GreyFrame
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, Region


RESCALE_FACTOR = 2.8 ### at 2.8, our letters come out to roughly 32 pixel height, which is the perfect height apparently for tesseract.
BACK_PIXEL_ROW = 5
THRESHOLD_VALUE_BACKGROUND = 255 // 2
THRESHOLD_VALUE_FOREGROUND = 100
X1, X2 = 10, 200#140 -> for just the point score, no games
Y1, Y2 = 305, 350
EROSION_KERNEL = (3, 3)
VERBOSE = False
SUPER_VERBOSE = False

LEN_X, LEN_Y = (X2 - X1), (Y2 - Y1)

def preprocess_frame(someFrm, 
                    background_value = 0, 
                    erosion_kernel = EROSION_KERNEL,
                    rescale_factor = RESCALE_FACTOR,
                    threshold_value = THRESHOLD_VALUE_FOREGROUND):
    workFrm = someFrm
    workFrm = workFrm.rescale(rescale_factor)
    workFrm = workFrm.to_grey()
    
    ### Erosion -> since we want to erode white features on dark background, we flip to white text on dark background...
    if background_value == 255:
        workFrm = ~workFrm

    if erosion_kernel is not None:
        workFrm = workFrm.erode(erosion_kernel)

    ### ... and then flip back
    workFrm = ~workFrm

    if threshold_value is None:
        return workFrm
    else:
        return workFrm > threshold_value

def get_upper_lower_string(inFrm, config_string = None, psm = 6):
    import pytesseract as tess
    if config_string is None:
        config_string = ''

    ### split in upper row and lower row
    upper_array, lower_array = np.split(inFrm.array, 2, axis = 0)
    
    ### letters_tennis whitelists the alphabet, - and .
    upper_string = tess.image_to_string(upper_array, config = f'--psm {psm} {config_string}')
    lower_string = tess.image_to_string(lower_array, config = f'--psm {psm} {config_string}')
    
    upper_string, lower_string = upper_string.replace(' ', ''), lower_string.replace(' ', '') ### remove spaces
    upper_string, lower_string = upper_string.replace('\n', ' '), lower_string.replace('\n', ' ') # replace linebreak with spaces

    return upper_string, lower_string

def get_ocr_config(config_name, conf):
    config_path = conf.get('ocr_configs', {}).get(config_name, config_name)
    return '--psm 6 ' + shlex.quote(os.fspath(config_path))

def load(vid, sto):
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_warmup_removed') ### TODO: once we remove warmups earlier in the pipeline, this will not be necessary!
    return segBF

def save(vid, sto, outSF):
    topnameSF, botnameSF = outSF['sb_name_upper_Flag'], outSF['sb_name_lower_Flag'],
    topscoreSF, botscoreSF = outSF['sb_score_upper_Flag'],  outSF['sb_score_lower_Flag']
    topgamesSF, botgamesSF = outSF['sb_games_upper_Flag'],outSF['sb_games_lower_Flag']
    topnameSF.rename('sb_name_upper_Flag').save(vid, sto)
    botnameSF.rename('sb_name_lower_Flag').save(vid, sto)
    topscoreSF.rename('sb_score_upper_Flag').save(vid, sto)
    botscoreSF.rename('sb_score_lower_Flag').save(vid, sto)
    topgamesSF.rename('sb_games_upper_Flag').save(vid, sto)
    botgamesSF.rename('sb_games_lower_Flag').save(vid, sto)

def main(conf, name = None):
    import pytesseract as tess
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']

    segBF = load(vid, sto)
    segBFs = segBF.filter(segBF).split()

    segStms = StreamList.from_SFList(segBFs, cap)

    p1, p2 = Point2D((X1, Y1)), Point2D((X2, Y2))
    scoreReg = Region.from_points(p1, p2)
    
    q1, q2 = Point2D((0, BACK_PIXEL_ROW)), Point2D((LEN_X, BACK_PIXEL_ROW))
    myLn = Line(q1, q2)

    outSFs = []
    for segStm in segStms:
        fs, ls = segStm.start_frame, segStm.stop_frame
        myFrm = segStm.get(fs + 5)
        if SUPER_VERBOSE: 
            myFrm.put(scoreReg).show()

        cropFrm = myFrm.crop(scoreReg)
        if SUPER_VERBOSE: 
            cropFrm.show()

        greyFrm = cropFrm.to_grey()
        if SUPER_VERBOSE: 
            greyFrm.show()

        threshMask = greyFrm > THRESHOLD_VALUE_BACKGROUND
        if SUPER_VERBOSE: 
            threshMask.to_grey().to_BGR().put(myLn).show()

        ### find the longest run of consecutive background values -> this is the name box, which contains only letters
        ### all segments to the right contain only numbers.
        background_values = threshMask.array[BACK_PIXEL_ROW]

        background_segments = ut.split_into_contiguous(background_values, offset = 0)
        segment_lengths = [len(seg) for seg in background_segments]
        segment_values = [np.unique(seg)[0] for seg in background_segments]
        name_box_idx = np.argmax(segment_lengths)

        cum_segment_lengths = np.cumsum(segment_lengths)
        if name_box_idx == 0:
            xl1 = 0
        else:
            xl1 = cum_segment_lengths[name_box_idx - 1]
        xl2, xl3, xl4 = cum_segment_lengths[name_box_idx], cum_segment_lengths[name_box_idx + 1], cum_segment_lengths[name_box_idx + 2]

        l1, l2 = Point2D((xl1, 0)), Point2D((xl2, LEN_Y))
        letterReg = Region.from_points(l1, l2)
        l3, l4 = Point2D((xl2, 0)), Point2D((xl3, LEN_Y))
        numberReg = Region.from_points(l3, l4)
        l5, l6 = Point2D((xl3, 0)), Point2D((xl4, LEN_Y))
        gamesReg = Region.from_points(l5, l6)

        if SUPER_VERBOSE: 
            cropFrm.put(letterReg).put(numberReg, color = const.BGR['strong_pink']).put(gamesReg, color = const.BGR['ball_yellow'], thickness = 2).show()

        letterFrm, numberFrm, gamesFrm = cropFrm.crop(letterReg), cropFrm.crop(numberReg), cropFrm.crop(gamesReg)

        background_value_letters, background_value_numbers, background_value_games = segment_values[name_box_idx], segment_values[name_box_idx + 1], segment_values[name_box_idx + 2]

        # print('______')
        
        # for erosion_kernel in (5, 5), (3, 3):
        #     for threshold in [None, 100, 110, 120, 130, 140]:
                
        #         letterFrm_processed = preprocess_frame(letterFrm, background_value = background_value_letters, 
        #                                                erosion_kernel = erosion_kernel, threshold_value = threshold)
        #         numberFrm_processed = preprocess_frame(numberFrm, background_value = background_value_numbers, 
        #                                                erosion_kernel = erosion_kernel, threshold_value = threshold)
        
        #         upper_letters, lower_letters = get_upper_lower_string(letterFrm_processed, config_string = 'letters_tennis', psm = 6)
        #         upper_numbers, lower_numbers = get_upper_lower_string(numberFrm_processed, config_string = 'digits_tennis', psm = 6)

        #         print(f'threshold: {str(threshold)}, erosion: {erosion_kernel} ---- upper: {upper_letters}/{upper_numbers}, lower: {lower_letters}/{lower_numbers}')
            
        letterFrm_processed = preprocess_frame(letterFrm, background_value = background_value_letters)
        numberFrm_processed = preprocess_frame(numberFrm, background_value = background_value_numbers)
        gamesFrm_processed = preprocess_frame(gamesFrm, background_value = background_value_games, 
                                              erosion_kernel = (3, 3), rescale_factor = round(RESCALE_FACTOR * 0.75, 1),
                                              threshold_value = THRESHOLD_VALUE_FOREGROUND)
        
        upper_letter_array, lower_letter_array = np.split(letterFrm_processed.array, 2, axis = 0)
        upper_number_array, lower_number_array = np.split(numberFrm_processed.array, 2, axis = 0)
        upper_games_array, lower_games_array = np.split(gamesFrm_processed.array, 2, axis = 0)

        ### letters_tennis whitelists the alphabet, - and .
        upper_letters = tess.image_to_string(upper_letter_array, config = get_ocr_config('letters_tennis', conf))
        lower_letters = tess.image_to_string(lower_letter_array, config = get_ocr_config('letters_tennis', conf))
        
        ### digits_tennis whitelists chars 0123456789AD, AD for advantage
        upper_numbers = tess.image_to_string(upper_number_array, config = get_ocr_config('digits_tennis', conf))
        lower_numbers = tess.image_to_string(lower_number_array, config = get_ocr_config('digits_tennis', conf))
        
        ### games_tennis whitelists chars 01234567
        upper_games = tess.image_to_string(upper_games_array, config = get_ocr_config('games_tennis', conf))
        lower_games = tess.image_to_string(lower_games_array, config = get_ocr_config('games_tennis', conf))

        ### kill the line break
        upper_letters, lower_letters = upper_letters.rstrip('\n'), lower_letters.rstrip('\n')
        upper_numbers, lower_numbers = upper_numbers.rstrip('\n'), lower_numbers.rstrip('\n')
        upper_games, lower_games = upper_games.rstrip('\n'), lower_games.rstrip('\n')

        upper_games, lower_games = '-'.join(upper_games), '-'.join(lower_games)

        featDict = {'sb_name_upper_Flag': upper_letters, 'sb_name_lower_Flag': lower_letters, 
                    'sb_score_upper_Flag': upper_numbers, 'sb_score_lower_Flag': lower_numbers,
                    'sb_games_upper_Flag': upper_games, 'sb_games_lower_Flag': lower_games}
        
        if VERBOSE: 
            print(featDict)

        streamDict = {frame: featDict for frame in range(fs, ls + 1)}
        outSF = StreamFeatures(streamDict)
        outSFs.append(outSF)

    outSFs = SFList(outSFs)
    outSF = outSFs.concatenate()

    print('sb_score_upper_Flag uniques: ', outSF['sb_score_upper_Flag'].unique())
    print('sb_score_lower_Flag uniques: ', outSF['sb_score_lower_Flag'].unique())

    print('sb_games_upper_Flag uniques: ', outSF['sb_games_upper_Flag'].unique())
    print('sb_games_lower_Flag uniques: ', outSF['sb_games_lower_Flag'].unique())

    save(vid, sto, outSF)                                       

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[12])

    name = ppl._name
    main(conf, name = name)

