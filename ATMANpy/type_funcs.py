# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import cv2 as cv
import numpy as np
from typing import Optional

def typecheck(myClass):
    for name, field_type in myClass.__annotations__.items():
        initialized_attribute = getattr(myClass, name)
        if not isinstance(initialized_attribute, field_type):
            curr_type = type(initialized_attribute)
            raise TypeError(f"Field Name {name} has to be {field_type}, not {curr_type}!")
    
# def arraycheck(myClass, channel_number: int, color_channel_length = int | None):
def arraycheck(myClass, 
               member_type: str = 'uint8',
               channel_number: int = 3, color_channel_length: Optional[int] = None):
    if myClass.array.dtype != member_type:
        raise TypeError(f'{member_type} integers expected!')
    
    if len(myClass.array.shape) > channel_number:
        raise TypeError(f'{channel_number}-dimensional array expected!')
    elif len(myClass.array.shape) == channel_number:
        if color_channel_length is not None:
            ### enforce BGR channel to be of length 3
            if myClass.array.shape[-1] != color_channel_length:
                raise TypeError(f"Color Channel has to be {color_channel_length}-dimensional!")
    else:
        ### if the array is of smaller dimension than the target, 
        ### we can generally fix it inside the specific post_init
        pass

def pointcheck(myClass):
    x, y = myClass.coords
    if not (isinstance(x, int) and isinstance(y, int)):
        raise TypeError("Point coordinates have to be integers!")
    ### prohibit negative points.
    if not (x >= 0 and y >= 0):
        raise TypeError("Negative coordinates not allowed!")
        