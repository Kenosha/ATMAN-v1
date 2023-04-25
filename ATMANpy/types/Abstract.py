# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:56:24 2023

@author: gianl
"""

import pandas as pd

from dataclasses import dataclass
from typing import Optional, List

from ATMANpy.types.Shapes3D import Point3D, Line3D
from ATMANpy.types.Features import StreamFeature, StreamFeatures

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
class PlayerMovement:
    playerFeatures: StreamFeatures
    
    @staticmethod
    def from_features(myFeat, which):
        ### myFeat -> a contiguous StreamFeatures containing 'top' and 'bot' feature
        ### which -> top or bottom
        return myFeat[which]
        


@dataclass(frozen = True)
class Trajectory:
    trajFeats: StreamFeatures ### with keys: ball, event
    
    def __iter__(self):
        return iter(self.trajFeats)
    
    def __getitem__(self, item):
        return self.trajFeats.__getitem__(item)
    
    def __repr__(self):
        initial_event = self.get_initial_event()
        frames = self.trajFeats.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        
        return f'Trajectory({initial_event}, frames: {min_frame} to {max_frame})'
    
    @staticmethod
    def from_features(trajFeats):
        assert 'ball' in trajFeats.get_features(), f'ball feature not found in trajectory init with {trajFeats}'
        assert 'event' in trajFeats.get_features(), f'event feature not found in trajectory init with {trajFeats}'
        return Trajectory(trajFeats.keep(['ball', 'event']))
    
    ### Trajectory Functions
    # -> last serve, hit or bounce frame, all flight frames!
    def get_trajectory_dataframe(self, keep_before = 1): ### keep x frames before the beginning of flight 

        eventDict = self.trajFeats['event']
    
        traj_frames = [frame for frame, event in eventDict.items() if event[0] == 'F'] ## init with flight frames
        for i in range(1, keep_before + 1):
            add_frame = min(traj_frames) - i
            if add_frame in eventDict:
                traj_frames = [add_frame, *traj_frames]
        
        ballSer = self.trajFeats['ball'].to_series()[traj_frames] ### get series of balls of interest
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
        eventDict = self.trajFeats['event']
        eventList = [event for event in eventDict.values() if not event.startswith('F')]
        eventList = list(set(eventList))
        assert len(eventList) == 1
        return eventList[0]
    
    def get_initial_frames(self):
        eventDict = self.trajFeats['event']
        return [frame for frame, event in eventDict.items() if not event.startswith('F')]
    
    def get_initial_position(self, how = 'mean'):
        ### how: mean, first, last -> which of the bounces do we want?
        ballDict = self.trajFeats['ball']
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
        eventDict = self.trajFeats['event']
        eventList = [event for event in eventDict.values() if event.startswith('F')]
        eventList = list(set(eventList))
        assert len(eventList) == 1
        return eventList[0]
    
    def get_flight_frames(self):
        eventDict = self.trajFeats['event']
        return [frame for frame, event in eventDict.items() if event.startswith('F')]
    
    ### Other Functions
    def get_frames(self):
        return self.trajFeats.get_frames()
    
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
            return TrajectoryFitted(self.trajFeats, None, None, None)
        vert_direction, ground_direction = vert_direction * unit_rescale, ground_direction * unit_rescale
        
        ### calculate 3D position from planar direction unit vectors
        flight_frames = self.get_trajectory_frames(keep_before = keep_before)
        ball_fit_dict = {}
        for frame in self.trajFeats:
            try:
                flight_idx = flight_frames.index(frame) ### check if frame is in flight frames
            except ValueError:
                ball_fit_dict[frame] = None
            else:
                ### "deplanarize", using the ground direction and vert_direction unit vectors
                x, z = pos_ground_theory[flight_idx], pos_vert_theory[flight_idx]
                ball_fit_dict[frame] = start_position + ground_direction * x + vert_direction * z
            
        ball_fit_SF = StreamFeature(ball_fit_dict, name = 'ball_fit')
        new_trajFeats = self.trajFeats.merge(ball_fit_SF)
        
        return TrajectoryFitted(new_trajFeats, v_init_ground, v_init_vert, spin_rpm)
        
    def get_splines(self):
        flight_frames = self.get_trajectory_frames(keep_before = 1)
        splineDict = self.trajFeats.spline_3D(pointFeature = 'ball', frameList = flight_frames)
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
    
    def get_fit_splines(self):
        flight_frames = self.get_trajectory_frames(keep_before = 1)
        splineDict = self.trajFeats.spline_3D(pointFeature = 'ball_fit', frameList = flight_frames)
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
                elif event[0] in ['B', 'N']: ### a bounce or net event!
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
    
    def get_splines(self, which = 'ball'):
        print(self)
        ### TODO: allow splining of player movement
        if which == 'ball':
            splineSF = self.shotTraj.get_splines()
            bouncesplineSFs = [bounceTraj.get_splines() for bounceTraj in self.bounceList]
            for bSF in bouncesplineSFs:
                ### need the connecting spline also! otherwise, there is a one frame gap...
                # frame_bef, lastLine3D = splineSF.last()
                # frame_aft, firstLine3D = bSF.first()
                # # assert frame_bef + 1 == frame_aft - 1, f'Connecting spline failed between {splineSF} and {bSF}'
                # connecting_spline = Line3D(lastLine3D.rightPt, firstLine3D.leftPt)
                # splineSF = splineSF.add(frame_bef + 1, connecting_spline)
                splineSF = splineSF.extend(bSF)
        elif which == 'ball_fit': ### has to be a TrajectoryFit
            splineSF = self.shotTraj.get_fit_splines()
            bouncesplineSFs = [bounceTraj.get_fit_splines() for bounceTraj in self.bounceList] ### not fitted!!!
            for bSF in bouncesplineSFs:
            #     ### need the connecting spline also! otherwise, there is a one frame gap...
            #     frame_bef, lastLine3D = splineSF.last()
            #     frame_aft, firstLine3D = bSF.first()
            #     assert frame_bef + 1 == frame_aft - 1, f'Connecting spline failed between {splineSF} and {bSF}'
            #     connecting_spline = Line3D(lastLine3D.rightPt, firstLine3D.leftPt)
            #     splineSF = splineSF.add(frame_bef + 1, connecting_spline)
                splineSF = splineSF.extend(bSF)
        return splineSF
    
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
    
     
    def fit_trajectory(self, theory_ground_array, theory_vert_array,
                       v_init_ground_space, v_init_vert_space, spin_rate_space,
                       keep_before = 1, sum_of_squares = False):
        return self.__class__([myShot.fit_trajectory(theory_ground_array, theory_vert_array,
                                                     v_init_ground_space, v_init_vert_space, spin_rate_space,
                                                     keep_before = 1, sum_of_squares = False) for myShot in self.shotList]) 
    
    def get_splines(self, which = 'ball'):
        splineSFs = [myShot.get_splines(which = which) for myShot in self.shotList]
        outSF, otherSFs = splineSFs[0], splineSFs[1:]
        for otherSF in otherSFs:
            outSF = outSF.extend(otherSF)
        return outSF
        
   
@dataclass(frozen = True)
class Point(Shots):
    ### every point is a collection of shots in sequence!
    
    def get_frames(self):
        return [frame for shot in self.shotList for frame in shot.get_frames()]
    
@dataclass(frozen = True)
class Points:
    pointList: List[Point]
    
    @staticmethod
    def from_stored_features(video_id, store_name, 
                             event_name = 'grammatic_event', ball_name = '3D_Ball', 
                             top_player_name = '3D_topPlayer', bot_player_name = '3D_botPlayer',
                             narrow_instantaneous_events = True):
        ### go from points to point
        featnameList = [event_name, ball_name, top_player_name, bot_player_name]
        myFeats = StreamFeatures.from_store(video_id, store_name, featnameList = featnameList)
        myFeats = myFeats.rename_feature(event_name, 'event').rename_feature(ball_name, 'ball')
        myFeats = myFeats.rename_feature(top_player_name, 'top').rename_feature(bot_player_name, 'bot')
        
        ### turn into list of contiguous stream features
        featList = myFeats.split_joint_features()
        
        ### narrow instantaneous events
        if narrow_instantaneous_events:
            narrowList = []
            for myFeat in featList:
                narrowDict = narrow_down_instantaneous_events(myFeat['event'])
                eventSF = StreamFeature(narrowDict, 'event')
                narrowFeat = myFeat.drop('event').merge(eventSF)
                narrowList.append(narrowFeat)
            featList = narrowList
        
        return Points([Point.from_features(myFeat) for myFeat in featList])
    
    def __getitem__(self, item):
        return self.pointList[item]
    

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

# myTrajFit.trajFeats['ball']
# myTrajFit.trajFeats['ball_fit']

# myTraj.spline()
# myTrajFit.spline()


# myShot.get_ball_hit_position()
# myShot.get_player_hit_position()
# myShot.get_bounce_positions()
