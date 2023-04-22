# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:56:24 2023

@author: gianl
"""

import pandas as pd

from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Shapes3D import Point3D, Line3D
from ATMANpy.types.Features import StreamFeatures

from ATMANpy.funcs.HMM.HPost import narrow_down_instantaneous_events
from ATMANpy.funcs.util_funcs import split_into_contiguous, widen_list
import ATMANpy.funcs.spin_funcs as sf


groundArr, vertArr, trajDict = sf.load_trajectories('TrajectoriesRPM_small', dtype = 'float32')

### initialise search spaces with proper arguments
import inspect
kwargList = inspect.getargspec(sf.initialize).args
kwargs = {k: v for k, v in trajDict.items() if k in kwargList}
v_init_ground_space, v_init_vert_space, spin_rate_space = sf.initialize(dtype = 'float32', 
                                                                        squeeze = True,  
                                                                        **kwargs)

import ATMANpy.auxil.constants as const

#%%

@dataclass(frozen = True)
class Trajectory:
    trajFeats: StreamFeatures
    
    def __iter__(self):
        return iter(self.trajFeats)
    
    def __getitem__(self, item):
        return self.trajFeats.__getitem__(item)
    
    def get_coord_frame(self, keep_before = 1): ### keep x frames before the beginning of flight 

        eventDict = self.trajFeats['event']
    
        traj_frames = [frame for frame, event in eventDict.items() if event[0] == 'F'] ## init with flight frames
        for i in range(1, keep_before + 1):
            add_frame = min(traj_frames) - i
            if add_frame in eventDict:
                traj_frames = [add_frame, *traj_frames]
        
        ballSer = pd.Series(self.trajFeats['ball'])[traj_frames] ### get series of balls of interest
        coordSer = ballSer.map(lambda myBall: myBall.get_point().coords) ### get coordinates
        coordFrm = coordSer.apply(pd.Series) ### how the hell does this work?
        coordFrm.columns = ['x', 'y', 'z']
        return coordFrm
    
    def get_planar_ball_flight(self, keep_before = 1):
        return sf.calculate_planar_ball_flight(self.get_coord_frame(keep_before = keep_before))
    
    def get_flight_direction(self, keep_before = 1):
        coordFrm = self.get_coord_frame(keep_before = keep_before)
        Dx, Dy = (coordFrm.iloc[-1] - coordFrm.iloc[0])[['x', 'y']]
        return Point3D((Dx, Dy, 0)).normalize()
    
    def get_pre_flight_position(self, keep_before = 1):
        coordFrm = self.get_coord_frame(keep_before = keep_before)
        return Point3D(coordFrm.iloc[0])
    
    def get_height(self, keep_before = 1):
        coordFrm = self.get_coord_frame(keep_before = keep_before)
        return coordFrm.iloc[0]['z']
    
    def get_flight_frames(self, keep_before = 1):
        coordFrm = self.get_coord_frame(keep_before = keep_before)
        return list(coordFrm.index)
    
    def fit_trajectory(self, theory_ground_array, theory_vert_array,
                       v_init_ground_space, v_init_vert_space, spin_rate_space,
                       keep_before = 1, sum_of_squares = False):
        ### transform to trajectory to 2D coordinates
        pos_ground_measured, pos_vert_measured = self.get_planar_ball_flight(keep_before = keep_before)
        
        ### compare against theoretical trajectory
        v_ground_idx, v_vert_idx, rpm_idx = sf.get_optimal_trajectory_indices(theory_ground_array, theory_vert_array, 
                                                                              pos_ground_measured, pos_vert_measured,
                                                                              sum_of_squares = sum_of_squares)
        
        ### get initial conditions
        v_init_ground = v_init_ground_space[v_ground_idx]
        v_init_vert = v_init_vert_space[v_vert_idx]
        spin_rpm = spin_rate_space[rpm_idx]
        
        ### get theoretical positions
        pos_ground_theory = theory_ground_array[v_ground_idx, v_vert_idx, rpm_idx, :len(pos_ground_measured)]
        pos_vert_theory = theory_vert_array[v_ground_idx, v_vert_idx, rpm_idx, :len(pos_vert_measured)]
        
        ### use positions to enhance the stream features with a 'ball_fit' feature.
        ball_fit_dict = {}
        flight_frames = self.get_flight_frames(keep_before = keep_before)
        start_position = self.get_pre_flight_position(keep_before = keep_before)
        ground_direction = self.get_flight_direction(keep_before = keep_before) * const.SCALE * const.METER_TO_FOOT 
        vert_direction = Point3D((0, 0, 1)) * const.SCALE * const.METER_TO_FOOT 
        for frame in self.trajFeats:
            try:
                flight_idx = flight_frames.index(frame) ### check if frame is in flight frames
            except ValueError:
                ball_fit_dict[frame] = None
            else:
                ### "deplanarize", using the ground direction and vert_direction unit vectors
                x, z = pos_ground_theory[flight_idx], pos_vert_theory[flight_idx]
                ball_fit_dict[frame] = start_position + ground_direction * x + vert_direction * z
            
        new_trajFeats = self.trajFeats.add('ball_fit', ball_fit_dict)
        
        return TrajectoryFitted(new_trajFeats, v_init_ground, v_init_vert, spin_rpm)
        
    def spline(self):
        flight_frames = self.get_flight_frames(keep_before = 1)
        splineDict = self.trajFeats.spline_3D(pointFeature = 'ball', frameList = flight_frames)
        return splineDict
    
@dataclass(frozen = True)
class TrajectoryFitted(Trajectory):
    ### a trajectory, with initial conditions describing its flight approximately
    ### a separate object, because fitting the trajectory (and creating initial conditions)
    ### takes a bit too long algorithmically,
    ### so we use this class to instantiate objects with cached initial conditions
    v_init_ground: float
    v_init_vert: float
    spin_rpm: float
     
    def spline(self):
        flight_frames = self.get_flight_frames(keep_before = 1)
        splineDict = self.trajFeats.spline_3D(pointFeature = 'ball', frameList = flight_frames)
        splineDict_fit = self.trajFeats.spline_3D(pointFeature = 'ball_fit', frameList = flight_frames)
        return splineDict, splineDict_fit
            
        
    
@dataclass(frozen = True)
class Shot:
    shotTraj: Trajectory
    bounceList: List[Trajectory] ### shots can bounce zero times, once, twice, ...
    
    @staticmethod
    ### go from shots to trajectories
    def from_features(myFeat):
        ### go from shots to trajectories
        events_non_null = {frame: event for frame, event in myFeat['event'].items() if event != 'None'}

        ### 1) Get the initial shot trajectory
        trajList = []
        current_traj_feature = {}
        registered_hit = None
        is_bouncing = False
        
        for frame, event in events_non_null.items():
            if registered_hit is None: ### no hit or bounce registered yet -> at the beginning of shot trajectory.
                if event[0] in ['S', 'H']: ### hit 
                    registered_hit = event
                    current_traj_feature[frame] = myFeat[frame]
                else:   
                    continue
            else:
                if registered_hit == event: ### the original hit event -> still at the beginning of shot trajectory.
                    current_traj_feature[frame] = myFeat[frame]
                elif event[0] in ['B', 'N']: ### a bounce or net event!
                    if is_bouncing: ### already bouncing in previous frame!
                        current_traj_feature = myFeat[frame]
                    else: ### first bouncing frame here!
                        trajList.append(StreamFeatures(current_traj_feature))
                        current_traj_feature = {frame: myFeat[frame]}
                    is_bouncing = True
                        
                else: ### in flight!
                    current_traj_feature[frame] = myFeat[frame]
                    is_bouncing = False
                    
        trajList.append(StreamFeatures(current_traj_feature))   
        
        shotTraj, bounceList = Trajectory(trajList[0]), [Trajectory(traj) for traj in trajList[1:]]
        
        return Shot(shotTraj, bounceList)
                    
    
@dataclass(frozen = True)
class Shots:
    ### a general collection of shots.
    shotList: List[Shot]
    
    @classmethod
    def from_features(cls, myFeat):
        ### go from point to shots
        events_non_null = {frame: event for frame, event in myFeat['event'].items() if event != 'None'}

        shotList = []
        current_shot_feature = {}
        registered_hit = None
        for frame, event in events_non_null.items():
            if registered_hit is None: ### no hit registered yet -> at the beginning of a shot.
                if event[0] in ['S', 'H']: ### hit event
                    registered_hit = event
                    current_shot_feature[frame] = myFeat[frame]
                else:
                    continue
            else:
                if registered_hit == event: ### the original hit event -> still at the beginning of a shot.
                    current_shot_feature[frame] = myFeat[frame]
                elif event[0] in ['S', 'H']: ### a different hit event!
                    shotList.append(StreamFeatures(current_shot_feature))
                    current_shot_feature = {frame: myFeat[frame]}
                    registered_hit = event
                else:
                    current_shot_feature[frame] = myFeat[frame]
                    
        shotList.append(StreamFeatures(current_shot_feature))    
        
        ### Check
        shotFrames = [f for shotFeat in shotList for f in shotFeat.get_frames()]
        nonnullFrames = list(events_non_null.keys())
        not_in_nonnull = [f for f in shotFrames if f not in nonnullFrames]
        not_in_shot = [f for f in nonnullFrames if f not in shotFrames]
        assert not_in_nonnull == [] and not_in_shot == [], f'Superfluous: {not_in_nonnull}. Missing: {not_in_shot}'
        
        return cls([Shot.from_features(shotFeat) for shotFeat in shotList])
            
    def __getitem__(self, item):
        return self.shotList[item]
    
   
@dataclass(frozen = True)
class Point(Shots):
    ### every point is a collection of shots in sequence!
    pass
    
@dataclass(frozen = True)
class Points:
    pointList: List[Point]
    
    @staticmethod
    def from_stored_features(video_id, store_name, 
                             event_name = 'grammatic_event', ball_name = '3D_Ball',
                                 narrow_instantaneous_events = True):
        ### go from points to point
        myFeats = StreamFeatures.from_store(video_id, store_name, featnameList = [event_name, ball_name])
        myFeats = myFeats.rename_feature(event_name, 'event').rename_feature(ball_name, 'ball')
        
        ### turn into list of contiguous stream features
        featList = myFeats.split_joint_features()
        
        ### narrow instantaneous events
        if narrow_instantaneous_events:
            narrowList = []
            for myFeat in featList:
                narrowDict = narrow_down_instantaneous_events(myFeat['event'])
                narrowFeat = myFeat.drop('event').add('event', narrowDict)
                narrowList.append(narrowFeat)
            featList = narrowList
        
        return Points([Point.from_features(myFeat) for myFeat in featList])
    
    def __getitem__(self, item):
        return self.pointList[item]
    


#%% Testing

video_id = 'pqxAmUWIoKc'
store_name = '10to100_ok2'

event_name = 'grammatic_event'
ball_name = '3D_Ball_smart_inference'

myPts = Points.from_stored_features(video_id, store_name, 
                                      event_name = event_name, ball_name = ball_name)
myPt = myPts[0]
myShot = myPt[0]
myTraj = myShot.shotTraj


pos_ground_measured, pos_vert_measured = myTraj.get_planar_ball_flight()

#%% 
### read theoretical trajectories

myTrajFit = myTraj.fit_trajectory(groundArr, vertArr, 
                                  v_init_ground_space, v_init_vert_space, spin_rate_space)

myTrajFit.trajFeats['ball']
myTrajFit.trajFeats['ball_fit']

myTraj.spline()
myTrajFit.spline()






