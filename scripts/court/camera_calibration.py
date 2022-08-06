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


##### CAMERA CALIBRATION ######

index = 93935 ## arbitrary feature


# a) get necessary court geometry items
from ATMANpy.types.Frames import GreyFrame
gFrm = GreyFrame.from_video(cap, index)

H = stream_features[index]['H']

from ATMANpy.funcs.court_funcs import find_courtlines, find_netlines
courtLines = find_courtlines(H, schemaLines)
netLines = find_netlines(gFrm, courtLines)

# b) gather points in world and on image
court_points_3D = schemaLines.get_3D_corner_points()
net_points_3D = schemaLines.get_3D_net_points()
court_points_3D = [p for p in court_points_3D]
net_points_3D = [p for p in net_points_3D]

court_points_2D = courtLines.get_court_region().to_points()
court_points_2D = [np.array(p.coords, dtype = np.float32) for p in court_points_2D]

net_l = netLines['Left Net']
net_r = netLines['Right Net']
net_points_2D = [net_l.leftPt, net_l.rightPt, net_r.rightPt]
net_points_2D = [np.array(p.coords, dtype = np.float32) for p in net_points_2D]

objpoints = np.array([*court_points_3D, *net_points_3D], dtype = np.float32)
imgpoints =  np.array([*court_points_2D, *net_points_2D], dtype = np.float32)

# c) create matrix A

objpoint = objpoints[0]
imgpoint = imgpoints[0]
def make_A_entry(objpoint, imgpoint):
    u, v = imgpoint
    objpoint_homog = np.append(objpoint, 1)
    
    u_entry = np.hstack([objpoint_homog, np.zeros(4), - u * objpoint_homog])
    v_entry = np.hstack([np.zeros(4), objpoint_homog, - v * objpoint_homog]) 
    
    entry = np.vstack([u_entry, v_entry])

    return entry

def make_A(objpoints, imgpoints):
    entryList = []
    for objP, imgP in zip(objpoints, imgpoints):
        entry = make_A_entry(objP, imgP)
        entryList.append(entry)
    return np.vstack(entryList)    
    
A = make_A(objpoints, imgpoints)

# d) Find Projection Matrix

vals, vecs = np.linalg.eig(A.T @ A)
smallest_value_index = vals.argmin()

p = vecs[smallest_value_index]
P = p.reshape(3, 4)

















