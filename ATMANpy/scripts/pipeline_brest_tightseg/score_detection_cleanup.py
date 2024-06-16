
import os
import time

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

def load(vid, sto):
    gameSF = StreamFeatures.from_store(vid, sto, featnameList = ['sb_games_upper_Flag', 'sb_games_lower_Flag']) ### TODO: once we remove warmups earlier in the pipeline, this will not be necessary!
    segBF = StreamFeature.from_store(vid, sto, featname = 'isRally_warmup_removed')
    return gameSF, segBF

def save(vid, sto, outSF):
    topgamesSF, botgamesSF = outSF['sb_games_upper_cleaned_Flag'], outSF['sb_games_lower_cleaned_Flag']
    topgamesSF.rename('sb_games_upper_cleaned_Flag').save(vid, sto)
    botgamesSF.rename('sb_games_lower_cleaned_Flag').save(vid, sto)

def main(conf, name = None):
    vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    cap = conf['cap']

    gameSF, segBF = load(vid, sto)
    gameSFs = gameSF.conform(segBF).filter(segBF).split()

    allDict = {}
    prev_upper_string, prev_lower_string = '0', '0'
    for rallySF in gameSFs:
        print('_______')
        curr_upper_string = rallySF['sb_games_upper_Flag'].unique()[0]
        curr_lower_string = rallySF['sb_games_lower_Flag'].unique()[0]

        if curr_upper_string is None:
            curr_upper_string = '0'
        if curr_lower_string is None:
            curr_lower_string = '0'

        try:
            curr_upper_ints, curr_lower_ints = [int(x) for x in curr_upper_string.split('-')], [int(x) for x in curr_lower_string.split('-')]
            prev_upper_ints, prev_lower_ints = [int(x) for x in prev_upper_string.split('-')], [int(x) for x in prev_lower_string.split('-')]
        except:
            print('fix!')
            print(curr_upper_string, prev_upper_string)
            showStm = Stream.from_SF(rallySF, cap)
            show_frame = showStm.get_framelist()[0]
            showStm.get(show_frame).show()
            
            uppers_corrected = input(f'Correct upper scoreline: ')
            if uppers_corrected == 'q':
                uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
            else:
                lowers_corrected = input(f'Correct lower scoreline: ')
            print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')
                
            featDict = {f: {'sb_games_upper_cleaned_Flag': uppers_corrected,
                            'sb_games_lower_cleaned_Flag': lowers_corrected} for f in rallySF.get_frames()}
            allDict.update(featDict)
            prev_upper_string, prev_lower_string = uppers_corrected, lowers_corrected
            continue

        if len(curr_upper_ints) != len(curr_lower_ints):
            print('fix!')
            print(curr_upper_ints, curr_lower_ints)
            showStm = Stream.from_SF(rallySF, cap)
            show_frame = showStm.get_framelist()[0]
            showStm.get(show_frame).show()

            uppers_corrected = input(f'Correct upper scoreline: ')
            if uppers_corrected == 'q':
                uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
            else:
                lowers_corrected = input(f'Correct lower scoreline: ')
            print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')

        else:
            prev_games_upper, prev_games_lower = prev_upper_ints[-1], prev_lower_ints[-1]
            curr_games_upper, curr_games_lower = curr_upper_ints[-1], curr_lower_ints[-1]

            if len(prev_upper_ints) == len(curr_upper_ints) and len(prev_lower_ints) == len(curr_lower_ints): ### Same Set!

                if prev_upper_ints[:-1] == curr_upper_ints[:-1] and prev_lower_ints[:-1] == curr_lower_ints[:-1]: ### finished sets remain the same!
                    if curr_games_upper not in [prev_games_upper, prev_games_upper + 1] or curr_games_lower not in [prev_games_lower, prev_games_lower + 1]:
                        print('fix!')
                        print(curr_upper_ints, curr_lower_ints)
                        showStm = Stream.from_SF(rallySF, cap)
                        show_frame = showStm.get_framelist()[0]
                        showStm.get(show_frame).show()
            
                        uppers_corrected = input(f'Correct upper scoreline: ')
                        if uppers_corrected == 'q':
                            uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
                        else:
                            lowers_corrected = input(f'Correct lower scoreline: ')
                        print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')

                    else:
                        print('All good!')
                        print(curr_upper_ints, curr_lower_ints)

                        uppers_corrected = curr_upper_string
                        lowers_corrected = curr_lower_string
                else:
                    print('fix!')
                    print(curr_upper_ints, curr_lower_ints)
                    showStm = Stream.from_SF(rallySF, cap)
                    show_frame = showStm.get_framelist()[0]
                    showStm.get(show_frame).show()
                    
                    uppers_corrected = input(f'Correct upper scoreline: ')
                    if uppers_corrected == 'q':
                        uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
                    else:
                        lowers_corrected = input(f'Correct lower scoreline: ')
                    print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')


            elif len(prev_upper_ints) + 1 == len(curr_upper_ints) and len(prev_lower_ints) + 1 == len(curr_lower_ints): ### Next Set begins
                # ### Check the finished sets final score.
                # lower_wins = prev_games_upper == curr_upper_ints[-2] and prev_games_lower + 1 == curr_lower_ints[-2]
                # upper_wins = prev_games_upper + 1 == curr_upper_ints[-2] and prev_games_lower == curr_lower_ints[-2]
                # if lower_wins or upper_wins:
                #     upper_set_score = '-'.join(curr_upper_ints[:-1])
                #     lower_set_score = '-'.join(curr_lower_ints[:-1])
                # else:
                #     upper_set_score = input('Correct Upper Set Final Score: ')
                #     lower_set_score = input('Correct Lower Set Final Score: ')

                if prev_upper_ints == curr_upper_ints[:-1] and prev_lower_ints == curr_lower_ints[:-1]: ### finished sets remain the same! 

                    if curr_games_upper == curr_games_lower == 0:
                        print('All good')
                        print(curr_upper_ints, curr_lower_ints)

                        uppers_corrected = curr_upper_string
                        lowers_corrected = curr_lower_string

                        # if upper_set_score == '' and lower_set_score == '':
                        #     uppers_corrected = str(curr_games_upper)
                        #     lowers_corrected = str(curr_games_lower)
                        # else:
                        #     uppers_corrected = upper_set_score + '-' + str(curr_games_upper)
                        #     lowers_corrected = lower_set_score + '-' + str(curr_games_lower)

                    else:
                        print('fix!')
                        print(curr_upper_ints, curr_lower_ints)
                        showStm = Stream.from_SF(rallySF, cap)
                        show_frame = showStm.get_framelist()[0]
                        showStm.get(show_frame).show()
                        
                        uppers_corrected = input(f'Correct upper scoreline: ')
                        if uppers_corrected == 'q':
                            uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
                        else:
                            lowers_corrected = input(f'Correct lower scoreline: ')
                        print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')
                
                else:
                    print('fix!')
                    print(curr_upper_ints, curr_lower_ints)
                    showStm = Stream.from_SF(rallySF, cap)
                    show_frame = showStm.get_framelist()[0]
                    showStm.get(show_frame).show()
                    
                    uppers_corrected = input(f'Correct upper scoreline: ')
                    if uppers_corrected == 'q':
                        uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
                    else:
                        lowers_corrected = input(f'Correct lower scoreline: ')
                    print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')


                    # if upper_set_score == '' and lower_set_score == '':
                    #     uppers_corrected = input(f'correct upper games: ?')
                    #     lowers_corrected = input(f'correct upper games: ?')
                    # else:
                    #     uppers_corrected = upper_set_score + '-' + input(f'correct upper games: {upper_set_score}-?')
                    #     lowers_corrected = lower_set_score + '-' + input(f'correct upper games: {lower_set_score}-?')

            else:
                print('fix!')
                print(curr_upper_ints, curr_lower_ints)
                showStm = Stream.from_SF(rallySF, cap)
                show_frame = showStm.get_framelist()[0]
                showStm.get(show_frame).show()
            
                uppers_corrected = input(f'Correct upper scoreline: ')
                if uppers_corrected == 'q':
                    uppers_corrected, lowers_corrected = prev_upper_string, prev_lower_string
                else:
                    lowers_corrected = input(f'Correct lower scoreline: ')
                print(f'  --> fixed to: {uppers_corrected}, {lowers_corrected}')


        featDict = {f: {'sb_games_upper_cleaned_Flag': uppers_corrected,
                        'sb_games_lower_cleaned_Flag': lowers_corrected} for f in rallySF.get_frames()}
        allDict.update(featDict)
        prev_upper_string, prev_lower_string = uppers_corrected, lowers_corrected

        
    outSF = StreamFeatures(allDict)
    save(vid, sto, outSF)

if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    
    prm_mod = ppl.parameter_files.default
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]
    conf = cnf_mod.get_conf(avail_vids[11])

    name = ppl._name

    main(conf, name = name)