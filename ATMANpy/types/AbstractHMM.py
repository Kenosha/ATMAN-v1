# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:56:24 2023

@author: gianl
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.ShapesTennis3D import Racquet

from ATMANpy.types.Shapes3D import Point3D, Line3D
from ATMANpy.types.Features import StreamFeature, StreamFeatures

from ATMANpy.funcs.HMM.HPost import narrow_down_instantaneous_events
from ATMANpy.funcs.util_funcs import split_into_contiguous, widen_list, prepend_index_level
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
class PlayerMovement:
    playerSF: StreamFeature
    
    @staticmethod
    def from_features(myFeat, which):
        ### myFeat -> a contiguous StreamFeatures containing 'top' and 'bot' feature
        ### which -> top or bottom
        return PlayerMovement(myFeat[which])
    
    def __getitem__(self, item):
        return self.playerSF.__getitem__(item)
    
    def get_name(self):
        return self.playerSF.name


@dataclass(frozen = True)
class Trajectory:
    trajSF: StreamFeatures ### with keys: ball, event
    
    def __iter__(self):
        return iter(self.trajSF)
    
    def __getitem__(self, item):
        return self.trajSF.__getitem__(item)
    
    def __repr__(self):
        initial_event = self.get_initial_event()
        frames = self.trajSF.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        
        return f'Trajectory({initial_event}, frames: {min_frame} to {max_frame})'
    
    @staticmethod
    def from_features(trajSF):
        assert 'ball' in trajSF.get_features(), f'ball feature not found in trajectory init with {trajSF}'
        assert 'event' in trajSF.get_features(), f'event feature not found in trajectory init with {trajSF}'
        return Trajectory(trajSF.keep(['ball', 'event']))
    
    ### Trajectory Functions
    # -> last serve, hit or bounce frame, all flight frames!
    def get_trajectory_dataframe(self, keep_before = 1): ### keep x frames before the beginning of flight 

        eventDict = self.trajSF['event']
        
        last_initial_frame = self.get_initial_frames()[-1]
        traj_frames = [frame for frame, event in eventDict.items() if event[0] == 'F'] ## init with flight frames
        for i in range(keep_before):
            add_frame = last_initial_frame - i
            if add_frame in eventDict:
                traj_frames = [add_frame, *traj_frames]
        
        ballSer = self.trajSF['ball'].to_series()[traj_frames] ### get series of balls of interest
        ballSer = ballSer.dropna() ### drop None ball events --> usually occur at end of point!
        coordSer = ballSer.map(lambda myBall: myBall.get_point().coords) ### get coordinates
        coordFrm = coordSer.apply(pd.Series) ### how the hell does this work?
        coordFrm.columns = ['x', 'y', 'z']
        return coordFrm
    
    def get_trajectory_frames(self, keep_before = 1):
        coordFrm = self.get_trajectory_dataframe(keep_before = keep_before)
        return list(coordFrm.index)
    
    def get_trajectory_planar(self, keep_before = 1):
        return sf.calculate_planar_ball_flight(self.get_trajectory_dataframe(keep_before = keep_before))
    
    def get_trajectory_direction(self, keep_before = 1):
        coordFrm = self.get_trajectory_dataframe(keep_before = keep_before)
        Dx, Dy = (coordFrm.iloc[-1] - coordFrm.iloc[0])[['x', 'y']]
        return Point3D((Dx, Dy, 0)).normalize()
    
    ### Initial Functions 
    # -> Serve, Hit, or Bounce frames!
    def get_initial_event(self):
        eventDict = self.trajSF['event']
        eventList = [event for event in eventDict.values() if not event.startswith('F')]
        eventList = list(set(eventList))
        assert len(eventList) == 1
        return eventList[0]
    
    def get_initial_frames(self):
        eventDict = self.trajSF['event']
        return [frame for frame, event in eventDict.items() if not event.startswith('F')]
    
    def get_initial_position(self, how = 'mean'):
        ### how: mean, first, last -> which of the bounces do we want?
        ballDict = self.trajSF['ball']
        hit_or_bounce_points = [ballDict[frame].get_point() for frame in self.get_initial_frames()]
        
        if how == 'mean':
            ### sum takes one or two arguments! if two are given, the second is the start value!
            return sum(hit_or_bounce_points, Point3D((0, 0, 0))) / len(hit_or_bounce_points)
        elif how == 'first':
            return hit_or_bounce_points[0]
        elif how == 'last':
            return hit_or_bounce_points[-1]
    
    ### Flight Functions
    # -> Flight frames
    def get_flight_event(self):
        eventDict = self.trajSF['event']
        eventList = [event for event in eventDict.values() if event.startswith('F')]
        eventList = list(set(eventList))
        assert len(eventList) == 1
        return eventList[0]
    
    def get_flight_frames(self):
        eventDict = self.trajSF['event']
        return [frame for frame, event in eventDict.items() if event.startswith('F')]
    
    ### Other Functions
    def get_frames(self):
        return self.trajSF.get_frames()
    
    def get_frame_event_series(self):
        initial_frames = self.get_initial_frames()
        return self.trajSF['event'].to_series()[initial_frames]
    
    def get_frame_event_pairs(self, how = 'last'):
        idx = 0 if how == 'first' else -1
        return self.get_initial_frames()[idx], self.get_initial_event()
    
    def get_frame_event_dict(self):
        hit_event = self.get_initial_event()
        hit_frames = self.get_initial_frames()
        return {'event': hit_event, 'start': min(hit_frames), 'end': max(hit_frames)}    
    
    def compare(self, otherTraj, reverse = False, left_name = 'left', right_name = 'right'):
        # if reverse: left is right, right is left
        self_which = left_name if not reverse else right_name
        other_which = right_name if not reverse else left_name
        
        ### compare hit with hit
        selfDict = self.get_frame_event_dict()
        ### turn into series
        selfSer = pd.Series(selfDict)
        selfSer.index.name = 'datum'
        selfSer = prepend_index_level(selfSer, 'which', self_which)

        if otherTraj is not None:
            otherDict = otherTraj.get_frame_event_dict()
            ### turn into series
            otherSer = pd.Series(otherDict)
            otherSer.index.name = 'datum'
            otherSer = prepend_index_level(otherSer, 'which', other_which)
        
            return pd.concat([selfSer, otherSer], axis = 0)
        else:
            return selfSer
    
    def fit_trajectory(self, theory_ground_array, theory_vert_array,
                       v_init_ground_space, v_init_vert_space, spin_rate_space,
                       keep_before = 1, sum_of_squares = False):
        ### transform to trajectory to 2D coordinates
        pos_ground_measured, pos_vert_measured = self.get_trajectory_planar(keep_before = keep_before)
        
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
        
        ### calculate planar direction unit vectors and start position in 3D
        unit_rescale = const.SCALE * const.METER_TO_FOOT 
        start_position = self.get_initial_position(how = 'last')
        vert_direction = Point3D((0, 0, 1))
        ground_direction = self.get_trajectory_direction(keep_before = keep_before) 
        if ground_direction is None: ### at the end of a point -> no direction because no interpolation!
            return TrajectoryFitted(self.trajSF, None, None, None)
        vert_direction, ground_direction = vert_direction * unit_rescale, ground_direction * unit_rescale
        
        ### calculate 3D position from planar direction unit vectors
        flight_frames = self.get_trajectory_frames(keep_before = keep_before)
        ball_fit_dict = {}
        for frame in self.trajSF:
            try:
                flight_idx = flight_frames.index(frame) ### check if frame is in flight frames
            except ValueError:
                ball_fit_dict[frame] = None
            else:
                ### "deplanarize", using the ground direction and vert_direction unit vectors
                x, z = pos_ground_theory[flight_idx], pos_vert_theory[flight_idx]
                ball_fit_dict[frame] = start_position + ground_direction * x + vert_direction * z
            
        ball_fit_SF = StreamFeature(ball_fit_dict, name = 'ball_fit')
        new_trajSF = self.trajSF.merge(ball_fit_SF)
        
        return TrajectoryFitted(new_trajSF, v_init_ground, v_init_vert, spin_rpm)
        
    ### Feature Getters ###
    def feat_splines(self):
        flight_frames = self.get_trajectory_frames(keep_before = 1)
        splineDict = self.trajSF.spline_3D(pointFeature = 'ball', frameList = flight_frames)
        return StreamFeature(splineDict, 'ball_spline')
    
@dataclass(frozen = True)
class TrajectoryFitted(Trajectory):
    ### a trajectory, with initial conditions describing its flight approximately
    ### a separate object, because fitting the trajectory (and creating initial conditions)
    ### takes a bit too long algorithmically,
    ### so we use this class to instantiate objects with cached initial conditions
    v_init_ground: float
    v_init_vert: float
    spin_rpm: float
    
    ### Feature Getters ###
    def feat_fit_splines(self):
        flight_frames = self.get_trajectory_frames(keep_before = 1)
        splineDict = self.trajSF.spline_3D(pointFeature = 'ball_fit', frameList = flight_frames)
        return StreamFeature(splineDict, 'ball_fit_spline')
    
    
@dataclass(frozen = True)
class Shot:
    shotTraj: Trajectory
    bounceList: List[Trajectory] ### shots can bounce zero times, once, twice, ...
    hitPlayer: PlayerMovement ### the hitting player
    recPlayer: PlayerMovement ### the receiving player

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
                elif event[0] in ['B', 'N', 'C']: ### a bounce, net or cord event!
                    if is_bouncing: ### already bouncing in previous frame!
                        current_traj_feature[frame] = myFeat[frame]
                    else: ### first bouncing frame here!
                        trajList.append(StreamFeatures(current_traj_feature))
                        current_traj_feature = {frame: myFeat[frame]}
                    is_bouncing = True
                        
                else: ### in flight!
                    current_traj_feature[frame] = myFeat[frame]
                    is_bouncing = False
                    
        trajList.append(StreamFeatures(current_traj_feature))   
        
        ### instantiate trajectories
        shotTraj, bounceList = Trajectory.from_features(trajList[0]), [Trajectory.from_features(traj) for traj in trajList[1:]]
        
        ### instantiate players
        far_or_near = registered_hit[1] ### F -> far; N -> near (correspond to top and bot resp.)
        hitStr = 'top' if far_or_near == 'F' else 'bot'
        recStr = 'bot' if far_or_near == 'F' else 'top'
        hitPlayer = PlayerMovement.from_features(myFeat, hitStr)
        recPlayer = PlayerMovement.from_features(myFeat, recStr)
        
        return Shot(shotTraj, bounceList, hitPlayer, recPlayer)

    def __repr__(self):
        frames = self.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        events = self.get_events()
        event_strings = '->'.join(events)
        hitStr, recStr = self.hitPlayer.get_name(), self.recPlayer.get_name()
        return f'Shot({event_strings}, hitting: {hitStr}, receiving: {recStr}, frames: {min_frame}...{max_frame})'
        
    def fit_trajectory(self, theory_ground_array, theory_vert_array,
                       v_init_ground_space, v_init_vert_space, spin_rate_space,
                       keep_before = 1, sum_of_squares = False, fit_bounces = True):
        shotTrajFit = self.shotTraj.fit_trajectory(theory_ground_array, theory_vert_array, 
                                                   v_init_ground_space, v_init_vert_space, spin_rate_space,
                                                   keep_before = keep_before, sum_of_squares = sum_of_squares)
        if fit_bounces:
            bounceList = [bounceTraj.fit_trajectory(theory_ground_array, theory_vert_array, 
                                                       v_init_ground_space, v_init_vert_space, spin_rate_space,
                                                       keep_before = keep_before, sum_of_squares = sum_of_squares) for bounceTraj in self.bounceList]
        else:
            bounceList = self.bounceList
        return Shot(shotTrajFit, bounceList, self.hitPlayer, self.recPlayer)
    
    def get_hit_event(self):
        return self.shotTraj.get_initial_event()
    
    def get_bounce_event(self):
        if len(self.bounceList) == 0:
            return None
        else:
            return self.bounceList[0].get_initial_event()
        
    def get_bounce_events(self):
        return [bounceTraj.get_initial_event() for bounceTraj in self.bounceList]
    
    def get_events(self):
        return [self.get_hit_event(), *self.get_bounce_events()]
    
    def get_ball_hit_position(self, how = 'mean'):
        return self.shotTraj.get_initial_position(how = 'mean')
    
    def get_hit_height(self):
        return self.get_ball_hit_position().z
    
    def get_bounce_positions(self):
        return [bounceTraj.get_initial_position() for bounceTraj in self.bounceList]
    
    def get_player_hit_position(self, how = 'mean'):
        hit_frames = self.shotTraj.get_initial_frames()
        hitting_player_points = [self.hitPlayer[frame].get_foot_point() for frame in hit_frames]
        
        if how == 'mean':
            ### sum takes one or two arguments! if two are given, the second is the start value!
            return sum(hitting_player_points, Point3D((0, 0, 0))) / len(hitting_player_points)
        elif how == 'first':
            return hitting_player_points[0]
        elif how == 'last':
            return hitting_player_points[-1]
    
    def get_hit_which(self):
        return 'bot' if self.get_hit_event()[1] == 'N' else 'top'
    
    def get_rec_which(self):
        return 'bot' if self.get_hit_event()[1] == 'F' else 'top' 
        
    def get_num_bounces(self):
        return len(self.bounceList)
    
    def get_frames(self):
        shot_frames = self.shotTraj.get_frames()
        bounce_frames = [frame for bounceTraj in self.bounceList for frame in bounceTraj.get_frames()]
        return [*shot_frames, *bounce_frames]
    
    def get_frame_event_pairs(self, how = 'last'):
        return [myTraj.get_frame_event_pairs(how = how) for myTraj in [self.shotTraj, *self.bounceList]]
    
    def get_frame_event_series(self):
        shotSer = self.shotTraj.get_frame_event_series()
        bounceList = [bounceTraj.get_frame_event_series() for bounceTraj in self.bounceList]
        return pd.concat([shotSer, *bounceList], axis = 0)
    
    def compare(self, otherShot, reverse = False, left_name = None, right_name = None):
        compFrm_rows = []
        
        if otherShot is None:
            hitSer = self.shotTraj.compare(None, reverse = reverse, left_name = left_name, right_name = right_name)
            compFrm_rows.append(hitSer)
            for bounceTraj in self.bounceList:
                bounceSer = bounceTraj.compare(None, reverse = reverse, left_name = left_name, right_name = right_name)
                compFrm_rows.append(bounceSer)
        else:
            ### compare hit with hit
            hitSer = self.shotTraj.compare(otherShot.shotTraj, reverse = reverse, left_name = left_name, right_name = right_name)
            compFrm_rows.append(hitSer)
            
            ### Compare bounces --> here, we need matching along closeness of frames!
            # idea: compute a match matrix 
                # -> i dimension: self.trajectories, j dimension: other.trajectories
                # value -> 'closeness' of other trajectory
            matchArr = np.zeros((len(self.bounceList), len(otherShot.bounceList)))
            
            def closeness_number(start1, stop1, start2, stop2):
                return (abs(start1 - start2) + abs(stop1 - stop2)) / 2
            
            for i, selfTraj in enumerate(self.bounceList):
                self_frames = selfTraj.get_initial_frames()
                self_min_frame, self_max_frame = min(self_frames), max(self_frames)
                
                for j, otherTraj in enumerate(otherShot.bounceList):
                    other_frames = otherTraj.get_initial_frames()
                    other_min_frame, other_max_frame = min(other_frames), max(other_frames)
                    
                    matchArr[i, j] = closeness_number(self_min_frame, self_max_frame, 
                                                 other_min_frame, other_max_frame)
            
            ### now, get minimum match for each 
            self_matches = []
            other_matches = []
            ACCEPTABLE_CLOSENESS = 10   
            try:
                best_match = np.argmin(matchArr, axis = 1)
            except ValueError: ### happens if there is no shot to argmin over:
                pass
            else:
                for selfIdx, otherIdx in enumerate(best_match): ### closest for each traj in self
                    if matchArr[selfIdx, otherIdx] <= ACCEPTABLE_CLOSENESS:
                        matchSer = self.bounceList[selfIdx].compare(otherShot.bounceList[otherIdx], reverse = reverse, left_name = left_name, right_name = right_name)
                        compFrm_rows.append(matchSer)
                        self_matches.append(selfIdx)
                        other_matches.append(otherIdx)
            
            ### now, deal with unmatched
            for i in range(len(self.bounceList)):
                if i not in self_matches:
                    unmatchSer = self.bounceList[i].compare(None, reverse = reverse, left_name = left_name, right_name = right_name)
                    compFrm_rows.append(unmatchSer)
                    
            for i in range(len(otherShot.bounceList)):
                if i not in other_matches:
                    unmatchSer = otherShot.bounceList[i].compare(None, reverse = ~reverse, ### inverted, because otherShot is "self" 
                                                                 left_name = left_name, right_name = right_name) 
                    compFrm_rows.append(unmatchSer)
                    
        ### sort out the output frame
        outFrm = pd.concat(compFrm_rows, axis = 1).T
        outFrm.columns = outFrm.columns.swaplevel()
        outFrm.index.name = 'Traj'
        outFrm = outFrm.sort_index(axis = 1, level = 0,
                                   key = lambda idx: idx.map({'event': 0, 'start': 1, 'end': 2}))
        return outFrm 
        
    ### FEATURE GETTERS ###
    # a) Objects
    def feat_hit_player(self):
        return self.hitPlayer.playerSF
    
    def feat_rec_player(self):
        return self.recPlayer.playerSF
    
    def feat_bot_player(self):
        if self.get_hit_which() == 'bot':
            return self.feat_hit_player()
        else:
            return self.feat_rec_player()
    
    def feat_top_player(self):
        if self.get_hit_which() == 'top':
            return self.feat_hit_player()
        else:
            return self.feat_rec_player()
        
    def feat_ball(self):
        ballSF = self.shotTraj.trajSF['ball']
        for bounceTraj in self.bounceList:
            ballSF = ballSF.extend(bounceTraj.trajSF['ball'])
        return ballSF
    
    def feat_hit_racquet(self, prevShot):
        ### prevShot -> needed to understand the direction of the incoming ball.
        top_or_bot = self.get_hit_which()
        
        hit_frames = self.shotTraj.get_initial_frames()
        flight_frames = [f for f in self.get_frames() if f not in hit_frames]
        last_hit_frame = max(hit_frames)
        
        ### figure out ready position (for after the hit has been made)
        READY_POSITION_WHEN = 3 ### how many frames after hit is the ready position reached?
        READY_POSITION_DIST = 10
        READY_POSITION_VERT = 0.5
        READY_POSITION_HORI = 0.5 if top_or_bot == 'top' else -0.5 
        readyPlayer = self.hitPlayer[last_hit_frame + READY_POSITION_WHEN]
        readyRacquet = Racquet.make_at_player(readyPlayer, 
                                              dir_hori = READY_POSITION_HORI,
                                              dir_vert = READY_POSITION_VERT,
                                              distance_from_player = READY_POSITION_DIST)
        
        sfDict = {}
        
        ### Figure out Hit Frames -> different for serve and normal hit
        if self.is_serve():
            ####### SERVE #########
            serveBall = self.shotTraj['ball'][last_hit_frame]
            nextBall = self.shotTraj[last_hit_frame + 1]['ball']
            servePlayer = self.hitPlayer[last_hit_frame]
            hitRacquet = Racquet.make_at_contact(servePlayer, serveBall,
                                                 aBall_before = None, aBall_after = nextBall)
            SERVE_MOTION_WHEN = 3 ### x frames before serve is hit, motion starts!
            
            for frame in hit_frames:
                if last_hit_frame - frame > SERVE_MOTION_WHEN:
                    sfDict[frame] = readyRacquet
                else:
                    
                    ### interpolate between the ready position and the serve contact racquet
                    lambda_bef_aft = 1 - (last_hit_frame - frame) / SERVE_MOTION_WHEN
                    motionRacquet = Racquet.make_between(readyRacquet, hitRacquet, lambda_bef_aft = lambda_bef_aft)
                    sfDict[frame] = motionRacquet
        else:
            ###### NORMAL HIT #####
            ### figure out previous and next ball, to calculate the inclination of the racquet.
            if prevShot is None:
                prevBall = None
            else:
                if prevShot.bounceList == []:
                    frm, prevBall = prevShot.shotTraj['ball'].last()
                else:
                    frm, prevBall = prevShot.shotTraj['ball'].last()
            nextBall = self.shotTraj[last_hit_frame + 1]['ball']
            
            ### start populating feature dictionary

            for frame in hit_frames:
                hitRacquet = Racquet.make_at_contact(self.hitPlayer[frame], self.shotTraj['ball'][frame],
                                                    aBall_before = prevBall, aBall_after = nextBall)
                sfDict[frame] = hitRacquet
        
        ### figure out flight frames
        for i, frame in enumerate(flight_frames):
            if i < READY_POSITION_WHEN:
                lambda_bef_aft = i / READY_POSITION_WHEN
                betweenRacquet = Racquet.make_between(hitRacquet, readyRacquet, lambda_bef_aft)
                sfDict[frame] = betweenRacquet
            else:
                sfDict[frame] = Racquet.make_at_player(self.hitPlayer[frame], 
                                                       dir_hori = READY_POSITION_HORI,
                                                       dir_vert = READY_POSITION_VERT,
                                                       distance_from_player = READY_POSITION_DIST)
            
        return StreamFeature(sfDict, name = 'Racquet')
    
    def feat_rec_racquet(self, nextShot):
        ### nextShot -> needed to understand the striking motion beginning before the actual shot event.
        top_or_bot = self.get_rec_which()
        
        ### ready position parameters
        READY_POSITION_DIST = 10
        READY_POSITION_VERT = 0.5
        READY_POSITION_HORI = 0.5 if top_or_bot == 'top' else -0.5 
        
        if nextShot is None: ### in this case, there is no takeback, no striking motion to prepare
            sfDict = {}
            for frame in self.get_frames():
                readyRacquet = Racquet.make_at_player(self.recPlayer[frame], 
                                                      distance_from_player = READY_POSITION_DIST,
                                                      dir_hori = READY_POSITION_HORI,
                                                      dir_vert = READY_POSITION_VERT)
                sfDict[frame] = readyRacquet
            
        else: ### in this case, we need to prepare the takeback, and the striking motion
            hit_frames = self.shotTraj.get_initial_frames()
            last_hit_frame = max(hit_frames)
            
            next_hit_frames = nextShot.shotTraj.get_initial_frames()
            next_first_hit_frame = min(next_hit_frames) ### want the first hit frame!
            next_last_hit_frame = max(next_hit_frames)
            
            ### figure out take-back position (in preparation of the next hit by the receiving player)
            TAKEBACK_POSITION_WHEN = 3 ### how many frames after hit is the ready position reached?
            TAKEBACK_POSITION_DIST = 10
            TAKEBACK_POSITION_VERT = 0.5
            prep_right = (nextShot.is_fore() and top_or_bot == 'bot') or (nextShot.is_back() and top_or_bot == 'top')
            TAKEBACK_POSITION_HORI = 0 if prep_right else 1
            takebackPlayer = self.recPlayer[last_hit_frame + TAKEBACK_POSITION_WHEN]
            takebackRacquet = Racquet.make_at_player(takebackPlayer, 
                                                     dir_hori = TAKEBACK_POSITION_HORI,
                                                     dir_vert = TAKEBACK_POSITION_VERT,
                                                     distance_from_player = TAKEBACK_POSITION_DIST)
            
            ### figure out where ball is struck (in preparation of the next hit by the receiving player)
            hitPlayer = nextShot.hitPlayer[next_first_hit_frame]
            hitBall = nextShot.shotTraj['ball'][next_first_hit_frame]
            hitBall_after = nextShot.shotTraj[next_last_hit_frame + 1]['ball']
            frm, hitBall_before = self.shotTraj['ball'].last() 
            
            hitRacquet = Racquet.make_at_contact(hitPlayer, hitBall,
                                                aBall_before = hitBall_before, aBall_after = hitBall_after)
            
            ### start populating feature dictionary
            REACTION_FRAMES = 2 ### how many frames needed to react before takeback
            BEGIN_STRIKE_WHEN = 3 ### how many frames before the hit the racquet begins swinging forward
            
            sfDict = {}
            for i, frame in enumerate(self.get_frames()):
                if i < REACTION_FRAMES: ### in ready position, before player has reacted
                
                    readyRacquet = Racquet.make_at_player(self.recPlayer[frame], 
                                           distance_from_player = READY_POSITION_DIST,
                                           dir_hori = READY_POSITION_HORI,
                                           dir_vert = READY_POSITION_VERT)
                    sfDict[frame] = readyRacquet
                    
                elif i < REACTION_FRAMES + TAKEBACK_POSITION_WHEN: ### taking back, in motion
                    lambda_bef_aft = i / (REACTION_FRAMES + TAKEBACK_POSITION_WHEN)
                    takingbackRacquet = Racquet.make_between(readyRacquet, takebackRacquet, lambda_bef_aft = lambda_bef_aft)
                    sfDict[frame] = takingbackRacquet
                
                elif len(self.get_frames()) - i < BEGIN_STRIKE_WHEN: ### striking, in motion
                    lambda_bef_aft = 1 - (len(self.get_frames()) - i) / BEGIN_STRIKE_WHEN
                    strikingRacquet = Racquet.make_between(takingbackRacquet, hitRacquet, lambda_bef_aft = lambda_bef_aft)
                    sfDict[frame] = strikingRacquet
                
                else: ### static, in take back position
                    takingbackRacquet = Racquet.make_at_player(self.recPlayer[frame], 
                                                               dir_hori = TAKEBACK_POSITION_HORI,
                                                               dir_vert = TAKEBACK_POSITION_VERT,
                                                               distance_from_player = TAKEBACK_POSITION_DIST)
                    sfDict[frame] = takingbackRacquet
                    
        return StreamFeature(sfDict, name = 'Racquet')
                
    # b) Splines
    def feat_ball_splines(self):
        splineSF = self.shotTraj.feat_splines()
        bouncesplineSFs = [bounceTraj.feat_splines() for bounceTraj in self.bounceList]
        for bSF in bouncesplineSFs:
            ### need the connecting spline also! otherwise, there is a one frame gap...
            # frame_bef, lastLine3D = splineSF.last()
            # frame_aft, firstLine3D = bSF.first()
            # # assert frame_bef + 1 == frame_aft - 1, f'Connecting spline failed between {splineSF} and {bSF}'
            # connecting_spline = Line3D(lastLine3D.rightPt, firstLine3D.leftPt)
            # splineSF = splineSF.add(frame_bef + 1, connecting_spline)
            splineSF = splineSF.extend(bSF)
            
    def feat_ball_fit_splines(self):
        ### has to be a TrajectoryFit
        splineSF = self.shotTraj.feat_fit_splines()
        bouncesplineSFs = [bounceTraj.feat_fit_splines() for bounceTraj in self.bounceList] ### not fitted!!!
        for bSF in bouncesplineSFs:
        #     ### need the connecting spline also! otherwise, there is a one frame gap...
        #     frame_bef, lastLine3D = splineSF.last()
        #     frame_aft, firstLine3D = bSF.first()
        #     assert frame_bef + 1 == frame_aft - 1, f'Connecting spline failed between {splineSF} and {bSF}'
        #     connecting_spline = Line3D(lastLine3D.rightPt, firstLine3D.leftPt)
        #     splineSF = splineSF.add(frame_bef + 1, connecting_spline)
            splineSF = splineSF.extend(bSF)
        return splineSF
    
    ### TODO: allow splining of player movement
    
    ### FILTERS ###
    def is_fore(self):
        ### TODO: Handedness of player!
        playerPt = self.get_player_hit_position()
        ballPt = self.get_ball_hit_position()
        
        if self.get_hit_which() == 'bot':
            return ballPt.is_right(playerPt)
        else:
            return ballPt.is_left(playerPt)
    
    def is_back(self):
        return not self.is_fore()
    
    def is_volley(self, prevShot):
        ### prevShot: None or Shot
        return prevShot.get_num_bounces() == 0 
    
    def is_ground(self, prevShot):
        ### prevShot: None or Shot
        return not self.is_volley(prevShot)
        
    def is_serve(self):
        return self.get_hit_event()[0] == 'S'
    
    def is_hit(self):
        return not self.is_serve()
        
    def is_double_bounce(self):
        return self.get_num_bounces() > 1
    
    def is_winner(self, nextShot):
        ### nextShot: None or Shot
        if nextShot is None:
            return self.is_fair(nextShot)
        else:
            return self.is_fair(nextShot) & self.is_double_bounce()
    
    def is_cross(self):
        ### if the x dimension is more than half the singles court width
        CROSS_THRESHOLD = const.SINGLES_WIDTH / 2
        trajVec = self.get_ball_hit_position() - self.get_bounce_positions()[0]
        return abs(trajVec.y) > CROSS_THRESHOLD
    
    def is_dtl(self):
        return not self.is_cross()
    
    def is_overhead(self):
        OVERHEAD_THRESHOLD = 2.25 * const.METER_TO_FOOT * const.SCALE ### 2.25 meter 
        return self.is_hit() & self.get_hit_height() > OVERHEAD_THRESHOLD
        
    def is_fair(self, nextShot):
        ### nextShot: None or Shot
        # Note: nextShot only needed if we record no bounce: 
        # if there is no nextShot, it's fault, otherwise, it's fair.
        bounce_event = self.get_bounce_event()
        if bounce_event is None:
            if nextShot is None:
                return False
            else:
                return True
        else:
            if bounce_event[1] == 'I': ### IN
                return True
            else:
                return False
    
    def is_fault(self, nextShot):
        ### nextShot: None or Shot
        # Note: nextShot only needed if we record no bounce: 
        # if there is no nextShot, it's fault, otherwise, it's fair.
        return not self.is_fair(nextShot)
    
    
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
    
    def __len__(self):
        return len(self.shotList)
     
    def fit_trajectory(self, theory_ground_array, theory_vert_array,
                       v_init_ground_space, v_init_vert_space, spin_rate_space,
                       keep_before = 1, sum_of_squares = False):
        return self.__class__([myShot.fit_trajectory(theory_ground_array, theory_vert_array,
                                                     v_init_ground_space, v_init_vert_space, spin_rate_space,
                                                     keep_before = 1, sum_of_squares = False) for myShot in self.shotList]) 
    
    ### FEATURE GETTERS ###
    
    def feat_hit_racquet(self):
        pass ### need to handle this method separately, but not so easy as with continuous Point!
    
   
@dataclass(frozen = True)
class Point(Shots):
    ### every point is a collection of shots in sequence!
    
    def __repr__(self):
        serveShot = self.shotList[0]
        server, receiver = serveShot.hitPlayer.get_name(), serveShot.recPlayer.get_name()
        events = self.get_events()
        frames = self.get_frames()
        event_string = '->'.join(events)
        min_frame, max_frame = frames[0], frames[-1]
        return f'Point({event_string}, serving: {server}, receiving: {receiver} from {min_frame} to {max_frame}.)'
    
    def get_events(self):
        return [event for shot in self.shotList for event in shot.get_events()]
        
    def get_frames(self):
        return [frame for shot in self.shotList for frame in shot.get_frames()]
    
    def get_winner(self):
        lastShot = self.shotList[-1]
        
        if lastShot.is_fair(None): ### None -> there is no nextShot
            return lastShot.hitPlayer.get_name()
        else:
            return lastShot.recPlayer.get_name()
        
    def get_frame_event_pairs(self, how = 'last'):
        return [frm_evt for myShot in self for frm_evt in myShot.get_frame_event_pairs(how = how)]
    
    def get_frame_event_series(self):
        return pd.concat([shot.get_frame_event_series() for shot in self.shotList], axis = 0)
    
    def compare(self, otherPt, left_name = None, right_name = None):
        shotframeList = []
        
        if len(self) >= len(otherPt):
            for idx in range(len(self)):
                selfShot = self[idx]
                try:
                    otherShot = otherPt[idx]
                except:
                    otherShot = None
                
                shotFrm = selfShot.compare(otherShot, reverse = False, left_name = left_name, right_name = right_name)
                shotFrm = prepend_index_level(shotFrm, 'Shot', idx)
                shotframeList.append(shotFrm)
                
        else:
            for idx in range(len(otherPt)):
                otherShot = otherPt[idx]
                try:
                    selfShot = self[idx]
                except:
                    selfShot = None
                
                shotFrm = otherShot.compare(selfShot, reverse = True, left_name = left_name, right_name = right_name)
                shotFrm = prepend_index_level(shotFrm, 'Shot', idx)
                shotframeList.append(shotFrm)
                
        outFrm = pd.concat(shotframeList, ignore_index = False)
        outFrm = outFrm.sort_index(axis = 1, level = 0,
                                   key = lambda idx: idx.map({'event': 0, 'start': 1, 'end': 2}))
        return outFrm
    
    def feat_hit_racquet(self):
        someSFs = []
        
        for i, myShot in enumerate(self.shotList):
            try:
                prevShot = self.shotList[i - 1]
            except:
                prevShot = None
            someSFs.append(myShot.feat_hit_racquet(prevShot))
    
        ### Join StreamFeature objects.
        outSF, otherSFs = someSFs[0], someSFs[1:]
        for otherSF in otherSFs:
            outSF = outSF.extend(otherSF)
        return outSF
    
    def feat_rec_racquet(self):
        someSFs = []
        
        for i, myShot in enumerate(self.shotList):
            try:
                nextShot = self.shotList[i + 1]
            except:
                nextShot = None
            someSFs.append(myShot.feat_rec_racquet(nextShot))
    
        ### Join StreamFeature objects.
        outSF, otherSFs = someSFs[0], someSFs[1:]
        for otherSF in otherSFs:
            outSF = outSF.extend(otherSF)
        return outSF
    
    def feat_top_racquet(self):
        someSFs = []
        
        for i, myShot in enumerate(self.shotList):
            if myShot.get_hit_which() == 'top':
                try:
                    prevShot = self.shotList[i - 1]
                except:
                    prevShot = None
                someSFs.append(myShot.feat_hit_racquet(prevShot))
            else:
                try:
                    nextShot = self.shotList[i + 1]
                except:
                    nextShot = None 
                someSFs.append(myShot.feat_rec_racquet(nextShot))
        
        ### Join StreamFeature objects.
        outSF, otherSFs = someSFs[0], someSFs[1:]
        for otherSF in otherSFs:
            outSF = outSF.extend(otherSF)
        return outSF
    
    def feat_bot_racquet(self):
        someSFs = []
        
        for i, myShot in enumerate(self.shotList):
            if myShot.get_hit_which() == 'bot':
                try:
                    prevShot = self.shotList[i - 1]
                except:
                    prevShot = None
                someSFs.append(myShot.feat_hit_racquet(prevShot))
            else:
                try:
                    nextShot = self.shotList[i + 1]
                except:
                    nextShot = None 
                someSFs.append(myShot.feat_rec_racquet(nextShot))
        
        ### Join StreamFeature objects.
        outSF, otherSFs = someSFs[0], someSFs[1:]
        for otherSF in otherSFs:
            outSF = outSF.extend(otherSF)
        return outSF
        
                
    
@dataclass(frozen = True)
class Points:
    pointList: List[Point]
    name: str = 'Points'
    
    @staticmethod
    def from_stored_features(video_id, store_name, name = "Points",
                             event_name = 'grammatic_event', ball_name = '3D_Ball', 
                             top_player_name = '3D_topPlayer', bot_player_name = '3D_botPlayer',
                             narrow_instantaneous_events = True):
        ### go from points to point
        featnameList = [event_name, ball_name, top_player_name, bot_player_name]
        myFeats = StreamFeatures.from_store(video_id, store_name, featnameList = featnameList)
        myFeats = myFeats.rename_feature(event_name, 'event').rename_feature(ball_name, 'ball')
        myFeats = myFeats.rename_feature(top_player_name, 'top').rename_feature(bot_player_name, 'bot')
        
        ### turn into list of contiguous stream features
        featList = myFeats.split()
        
        ### narrow instantaneous events
        if narrow_instantaneous_events:
            narrowList = []
            for myFeat in featList:
                narrowDict = narrow_down_instantaneous_events(myFeat['event'])
                eventSF = StreamFeature(narrowDict, 'event')
                narrowFeat = myFeat.drop('event').merge(eventSF)
                narrowList.append(narrowFeat)
            featList = narrowList
            
        return Points([Point.from_features(myFeat) for myFeat in featList], name = name)
    
    def __getitem__(self, item):
        return self.pointList[item]
    
    def __len__(self):
        return len(self.pointList)
    
    def compare(self, otherPts, left_name = None, right_name = None):
        assert len(self) == len(otherPts), 'Points have to be equal length to compare!'
        if left_name is None:
            left_name = self.name
        if right_name is None:
            right_name = otherPts.name
        
        frameList = []
        for idx in range(len(self)):
            compFrm = self[idx].compare(otherPts[idx], left_name = self.name, right_name = otherPts.name)      
            compFrm = prepend_index_level(compFrm, 'Point', idx)
            frameList.append(compFrm)
        return pd.concat(frameList)
        
    
#### CREATE FEATURE GETTERS FOR POINT, SHOTS
def __make_method(myClass, shotMethodStr): ### helper to automatically make all "feat_" methods that exist in Shot for Shots also.
    def method(self):
        someSFs = [getattr(myShot, shotMethodStr)() for myShot in self.shotList]
        outSF, otherSFs = someSFs[0], someSFs[1:]
        for otherSF in otherSFs:
            outSF = outSF.extend(otherSF)
        return outSF
    method.__name__ = shotMethodStr
    setattr(myClass, shotMethodStr, method)
    
for methodStr in dir(Shot):
    if methodStr.startswith('feat_'):
        if not methodStr in dir(Shots): ### do not overwrite
            __make_method(Shots, methodStr)
        if not methodStr in dir(Point): ### do not overwrite
            __make_method(Point, methodStr)
    

# #%% Testing

# video_id = 'pqxAmUWIoKc'
# store_name = '10to100_ok2'

# event_name = 'grammatic_event'
# ball_name = '3D_Ball_smart_inference'
# top_player_name = '3D_topPlayer'
# bot_player_name = '3D_botPlayer'

# myPts = Points.from_stored_features(video_id, store_name, 
#                                     event_name = event_name, ball_name = ball_name,
#                                     top_player_name = top_player_name, bot_player_name = bot_player_name)
# myPt = myPts[0]
# myShot = myPt[0]
# myTraj = myShot.shotTraj


# pos_ground_measured, pos_vert_measured = myTraj.get_trajectory_planar()

# #%% 
# ### read theoretical trajectories

# myTrajFit = myTraj.fit_trajectory(groundArr, vertArr, 
#                                   v_init_ground_space, v_init_vert_space, spin_rate_space)

# myTrajFit.trajSF['ball']
# myTrajFit.trajSF['ball_fit']

# myTraj.spline()
# myTrajFit.spline()


# myShot.get_ball_hit_position()
# myShot.get_player_hit_position()
# myShot.get_bounce_positions()
