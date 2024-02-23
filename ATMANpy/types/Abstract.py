"""

As opposed to AbstractHMM, this Abstract is based on DataFrames, not StreamFeatures 
This is because we likely will not need the generality of SFs, having only coordinates in 3D space

In addition, there are some other changes, like
    - event labelling simplified
    - adaptation to flight + bounce trajectory modelling (not just flight).


### Events
# -> F: Flight
# -> STD/STA/SBD/SBA: Serve Top Deuce etc.
# -> HT/HB: Hit Top, Hit Bottom
# -> BI/BO: Bounce In, Bounce Out
# -> N: Net

"""


from typing import Optional, List

import numpy as np
import pandas as pd

from dataclasses import dataclass

from ATMANpy.types.Features import StreamFeature, StreamFeatures, SFList
from ATMANpy.types.Shapes3D import Point3D, Line3D
from ATMANpy.types.ShapesTennis3D import Player

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.sf_apply_funcs as sa

@dataclass(frozen = True)
class PlayerMovement:
    playerFrm: pd.DataFrame
    name: str
    
    @staticmethod
    def from_features(ply3DSF, which):
        ### which: top, bot
        player_features = ply3DSF.get_features()
    
        trunc_features = [f[:3] for f in player_features] ### truncated features -> keep top or bot
        chosen_feature = player_features[trunc_features.index(which)]
        
        playerFrm = ply3DSF[chosen_feature].apply(sa.point3D_to_coords).to_dataframe()
        return PlayerMovement(playerFrm, name = which)
    
    def __getitem__(self, item):
        if item in ['x', 'y', 'z']:
            return self.playerFrm[item]
        else:
            return self.playerFrm.loc[item] ### TODO: maybe as point?
        
    def get_frames(self):
        return self.playerFrm.index.to_list()
    
    def get_hit_frame(self):
        return self.get_frames()[0]
    
    def get_initial_position(self):
        x, y, z = self.playerFrm.loc[self.get_hit_frame()]
        return Point3D((x, y, z))
    
    def feat_player(self):
        def applicator(x, y, z):
            coords = x, y, z
            player = Player.make(coords)
            return player 
            
        plSF = StreamFeatures.from_dataframe(self.playerFrm).apply(applicator)
        plSF = plSF.rename(f'{self.name}Player3D')
        return plSF
    
@dataclass(frozen = True)
class Trajectory:
    trajFrm: pd.DataFrame
    """ possible keys:
        - x_o, y_o, z_o: observed ball in 3D space
        - px_o, pz_o: observed ball in 2D planar space (not projection space!!)
        - x_t, y_t, z_t: fitted ball in 3D space
        - px_t, pz_t: fitted ball in 2D planar space
        - rpm_t: fitted spin (can change after bounce)
        - vx0, vz0, rpm0: initial conditions - ground speed, vertical speed, spin
        - event during frame.
    """
    
    def __getitem__(self, item):
        if isinstance(item, list):
            check_item = item[0]
        else:
            check_item = item
            
        if isinstance(check_item, str):
            return self.trajFrm[item]
        else:
            return self.trajFrm.loc[item]
    
    def get_frames(self):
        return self.trajFrm.index.to_list()
    
    def get_hit_series(self):
        return self.trajFrm.iloc[0]
        
    def get_hit_event(self):
        return self['event'].iloc[0]
        
    def get_hit_frame(self):
        return self.get_frames()[0]
    
    def get_bounce_events(self):
        return self['event'][self['event'].str.startswith('B')].values.tolist()
    
    def get_bounce_frames(self):
        return self['event'][self['event'].str.startswith('B')].index.tolist()
    
    def get_num_bounces(self):
        return len(self.get_bounce_events())
    
    def get_events(self):
        return [self.get_hit_event(), *self.get_bounce_events()]
    
    def get_initial_position(self, how = 'trajectory'):
        ### how: trajectory / observed
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
            
        x, y, z = self.get_hit_series()[coord_cols] 
        return Point3D((x, y, z)) ### return point?
    
    def get_bounce_positions(self):
        coord_cols = ['x_t', 'y_t', 'z_t']
        bounce_frames = self.get_bounce_frames()
        
        bounce_points = []
        for frm in bounce_frames:
            x, y, z = self.trajFrm.loc[frm, coord_cols]
            bounce_points.append(Point3D((x, y, z)))
        return bounce_points
    
    def get_initial_height(self, how = 'trajectory'):
        return self.get_initial_position(how = how).z
    
    #### FEATURE GETTERS ######
    def feat_ball(self, how = 'trajectory'):
        ### how: trajectory / observed
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
           
        ballFrm = self.trajFrm[coord_cols]
        ballFrm.columns = ['x', 'y', 'z']
        ballSF = StreamFeatures.from_dataframe(ballFrm).apply(sa.coords_to_ball)
        return ballSF
    
    def feat_spline(self, how = 'trajectory'):
        ballSF = self.feat_ball(how = how)
        ballSF_before = ballSF.shift(offset = 1)
        def make_spline(outBall_bef1, outBall):
            if outBall_bef1 is None:
                spLn = None
            else:
                spLn = Line3D(outBall_bef1, outBall)
                return spLn
        return ballSF_before.merge(ballSF).apply(make_spline)
    
    
@dataclass(frozen = True)
class Shot:
    shotTraj: Trajectory
    hitPlayer: PlayerMovement
    recPlayer: PlayerMovement
    
    def __post_init__(self):
        assert self.shotTraj.get_frames() == self.hitPlayer.get_frames() == self.recPlayer.get_frames()
    
    def __repr__(self):
        frames = self.get_frames()
        min_frame, max_frame = frames[0], frames[-1]
        events = self.get_events()
        event_strings = '->'.join(events)
        hitStr, recStr = self.hitPlayer.name, self.recPlayer.name
        return f'Shot({event_strings}, hitting: {hitStr}, receiving: {recStr}, frames: {min_frame}...{max_frame})'
        
    def get_frames(self):
        return self.shotTraj.get_frames()

    def get_events(self):
        return self.shotTraj.get_events()
    
    def get_hit_frame(self):
        return self.shotTraj.get_frames()
    
    def get_hit_event(self):
        return self.shotTraj.get_hit_event()
    
    def get_player_hit_position(self):
        return self.hitPlayer.get_initial_position()
    
    def get_ball_hit_position(self):
        return self.shotTraj.get_initial_position()
    
    def get_ball_hit_height(self):
        return self.shotTraj.get_initial_height()
    
    def get_bounce_events(self):
        return self.shotTraj.get_bounce_events()
    
    def get_bounce_positions(self):
        return self.shotTraj.get_bounce_positions()
    
    def get_num_bounces(self):
        return self.shotTraj.get_num_bounces()
    
    ### FILTERS ###
    def is_serve(self):
        return self.get_hit_event()[0] == 'S'
    
    def is_hit(self):
        return not self.is_serve()
    
    def is_fore(self):
        ### TODO: Handedness of player!
        playerPt = self.get_player_hit_position()
        ballPt = self.get_ball_hit_position()
        
        if self.hitPlayer.name == 'bot':
            is_fore_side = ballPt.is_right(playerPt)
        else:
            is_fore_side = ballPt.is_left(playerPt)
    
        return self.is_hit() and is_fore_side
    
    def is_back(self):
        return self.is_hit() and not self.is_fore()
    
    def is_volley(self, prevShot):
        ### prevShot: None or Shot
        return prevShot.get_num_bounces() == 0 
    
    def is_ground(self, prevShot):
        ### prevShot: None or Shot
        return not self.is_volley(prevShot)
        
    def is_double_bounce(self):
        return self.get_num_bounces() > 1
    
    def is_winner(self, nextShot):
        ### nextShot: None or Shot
        if nextShot is None:
            return self.is_fair(nextShot)
        else:
            return self.is_fair(nextShot) & self.is_double_bounce()
    
    ### HAVE TO THINK ABOUT THESE TWO 
        ### need direction vector in Trajectory (direction independent of bounce!!)
        ### need also to know whether we are in the corner 
            # -> shot hit from middle is not really 'cross', shot hit down the middle not really 'down the line' 
    # def is_cross(self):
    #     ### if the x dimension is more than half the singles court width
    #     CROSS_THRESHOLD = const.SINGLES_WIDTH / 2
    #     trajVec = self.get_ball_hit_position() - self.get_bounce_positions()[0]
    #     return abs(trajVec.y) > CROSS_THRESHOLD
    
    # def is_dtl(self):
    #     return not self.is_cross()
    
    def is_overhead(self):
        OVERHEAD_THRESHOLD = 2.25 * const.METER_TO_FOOT * const.SCALE ### 2.25 meter 
        return self.is_hit() and self.get_ball_hit_height() > OVERHEAD_THRESHOLD
        
    def is_fair(self, nextShot):
        ### nextShot: None or Shot
        # Note: nextShot only needed if we record no bounce: 
        # if there is no nextShot, it's fault, otherwise, it's fair.
        try:
            first_bounce = self.get_bounce_events()[0]
        except IndexError: ### no bounce!
            if nextShot is None:
                return False
            else:
                return True
        else:
            if first_bounce[1] == 'I': ### IN
                return True
            else:
                return False
    
    def is_fault(self, nextShot):
        ### nextShot: None or Shot
        # Note: nextShot only needed if we record no bounce: 
        # if there is no nextShot, it's fault, otherwise, it's fair.
        return not self.is_fair(nextShot)
    
    
    #### FEATURE GETTERS ######
    def feat_ball(self, how = 'trajectory'):
        return self.shotTraj.feat_ball(how = how)
    
    def feat_hit_player(self):
        return self.hitPlayer.feat_player()
    
    def feat_rec_player(self):
        return self.recPlayer.feat_player()
    
    def feat_top_player(self):
        if self.hitPlayer.name == 'top':
            return self.hitPlayer.feat_player()
        else:
            return self.recPlayer.feat_player()
        
    def feat_bot_player(self):
        if self.hitPlayer.name == 'bot':
            return self.hitPlayer.feat_player()
        else:
            return self.recPlayer.feat_player()
        
    def feat_spline(self, how = 'trajectory'):
        return self.shotTraj.feat_spline(how = how)
        
@dataclass(frozen = True)
class Shots:
    ### a general collection of shots.
    shotList: List[Shot]
    
    def __iter__(self):
        return iter(self.shotList)
    
    def __getitem__(self, item):
        return self.shotList[item]
    
    def __len__(self):
        return len(self.shotList)
     
    # ### FEATURE GETTERS ######
    ### done with make_method below!!!
        
@dataclass(frozen = True)
class Point(Shots):
    ### every point is a collection of shots in sequence!
    def __repr__(self):
        serveShot = self.shotList[0]
        server, receiver = serveShot.hitPlayer.name, serveShot.recPlayer.name
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

#### CREATE FEATURE GETTERS FOR POINT, SHOTS
def __make_method(myClass, shotMethodStr): ### helper to automatically make all "feat_" methods that exist in Shot for Shots also.
    def method(self):
        SFs = SFList([getattr(myShot, shotMethodStr)() for myShot in self])
        return SFs.concatenate()
    method.__name__ = shotMethodStr
    setattr(myClass, shotMethodStr, method)
    
for methodStr in dir(Shot):
    if methodStr.startswith('feat_'):
        if not methodStr in dir(Shots): ### do not overwrite
            __make_method(Shots, methodStr)
        if not methodStr in dir(Point): ### do not overwrite
            __make_method(Point, methodStr)
    
        