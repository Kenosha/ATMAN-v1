"""

This module contains functions that take as FIRST argument a Point, and then other arguments potentially.
They return a boolean always.

This way, we can use them with the .filter method of points, to search for specific Points.

"""

import ATMANpy.funcs.filter_shots_funcs as filt_shot

def is_decisive(myPnt):
    return myPnt.get_winner_from_score() is not None
def is_let_or_first_serve_fault(myPnt):
    return myPnt.get_winner_from_score() is None

def is_server(myPnt, name):
    return myPnt.get_srv_player_name() == name
def is_returner(myPnt, name):
    return myPnt.get_rec_player_name() == name

def is_serve_out_wide(myPnt):
    return filt_shot.is_out_wide(myPnt.get_serve())
def is_serve_down_middle(myPnt):
    return filt_shot.is_down_middle(myPnt.get_serve())

def is_longer_than(myPnt, threshold):
    return len(myPnt) > threshold 
def is_shorter_than(myPnt, threshold):
    return len(myPnt) < threshold 
def is_exactly_this_long(myPnt, length):
    return len(myPnt) == length