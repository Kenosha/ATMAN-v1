
import numpy as np
import pandas as pd
from functools import partial


def get_rally_probabilities(hitSer, next_hit_should_come_soon = 20, next_hit_should_have_come = 50, 
                            decay_mild = 0.9, decay_strong = 0.5,
                            initial_rally_probability = 0):

    rally_probability = initial_rally_probability
    time_since_last_potential_stroke = 0
    
    rally_probabilities = []
    for row, hit_probability in hitSer.items():
        if hit_probability > rally_probability:
            rally_probability = hit_probability
            time_since_last_potential_stroke = 0
        else:
            if time_since_last_potential_stroke < next_hit_should_come_soon:
                pass  ### leave rally probability unchanged!
            elif time_since_last_potential_stroke < next_hit_should_have_come:
                rally_probability *= decay_mild
            else:
                rally_probability *= decay_strong
            
            time_since_last_potential_stroke += 1 
        
        rally_probabilities.append(rally_probability)
    
    return pd.Series(rally_probabilities, index = hitSer.index, name = 'rally_p')
    
def make_smoothing_function(kernel, mode = 'same'):
    norm_kernel = kernel / kernel.sum()
    return partial(np.convolve, v = norm_kernel, mode = mode)

def make_video_hit_frame(videoFrm):
    ### get video
    tophitSer = 1 - videoFrm['top_oth']
    bothitSer = 1 - videoFrm['bot_oth']
    tophitSer.name = 'top_hit_p'
    bothitSer.name = 'bot_hit_p'
    return pd.concat([tophitSer, bothitSer], axis = 1)

def make_video_hit_series(videoFrm, name = 'video_hit_p'):
    vidhitFrm = make_video_hit_frame(videoFrm)
    vidhitSer = vidhitFrm.max(axis = 1)
    vidhitSer.name = name
    
    return vidhitSer

def make_audio_hit_series(audioSer, name = 'audio_hit_p'):
    ### get audio
    audiohitSer = audioSer
    audiohitSer.name = name
    return audiohitSer
        
def weight_cobb_douglas(leftSer, rightSer, left_lambda, name = 'joint_hit_p'):
    outSer = (leftSer ** left_lambda).mul(rightSer ** (1 - left_lambda), axis = 'index')
    outSer.name = name
    return outSer
