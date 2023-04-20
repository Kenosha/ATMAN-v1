# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:56:24 2023

@author: gianl
"""

from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Shapes3D import Point3D
from ATMANpy.types.Features import StreamFeatures

from ATMANpy.funcs.HMM.HPost import narrow_down_instantaneous_events
from ATMANpy.funcs.util_funcs import split_into_contiguous, widen_list


@dataclass(frozen = True)
class Trajectory:
    trajFeats: StreamFeatures
    # startEvent: str
    # endEvent: str
    # pointList: List[Point3D]
    
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
class Point:
    shotList: List[Shot]
    
    @staticmethod
    def from_features(myFeat):
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
        
        return Point([Shot.from_features(shotFeat) for shotFeat in shotList])
        
        # ## first: find frames where 
        # flight_frames = [frame for frame in events if events[frame] in ['FLIGHT_UP', 'FLIGHT_DOWN']]
        # contiguous_flight_frames = split_into_contiguous(flight_frames)
        # widened_flight_frames = [widen_list(myList, by = 1)[:-1] for myList in contiguous_flight_frames] ### these include also the frame with the event right before flight!

        # ### shot frames
        # def is_shot(fList): ### is a trajectory a shot or a bounce?
        #     return events[fList[0]] in ['S', 'H'] # recall that by widening at the left, the first frame is the event preceding flight!
        
        # current_shot_traj = 
        # for fList in widened_flight_frames:
            
        
        
        # shot_frames = [fs for fs in widened_flight_frames if events[fs[0]] in ['S', 'H']]
        # bounce_frames = [fs for fs in widened_flight_frames if fs not in shot_frames]        

        

    
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
    


#%% Testing

video_id = 'pqxAmUWIoKc'
store_name = '10to100_ok2'

event_name = 'grammatic_event'
ball_name = '3D_Ball_smart_inference'

myFeats = StreamFeatures.from_store(video_id, store_name, featnameList = [event_name, ball_name])
myFeats = myFeats.rename_feature(event_name, 'event').rename_feature(ball_name, 'ball')
featList = myFeats.split_joint_features()

myFeat = featList[0]
myPt = Point.from_features(myFeat)
myPt.shotList[0]

Points.from_stored_features(video_id, store_name, 
                            event_name = event_name, ball_name = ball_name)


 