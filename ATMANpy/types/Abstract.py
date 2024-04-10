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

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Streams import Stream
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
    
    def flip(self, courtLines = None, across_net = True, across_middle = True):
        if courtLines is None:
            courtLines = CourtLines.make()
        
        newFrm = self.playerFrm.copy()
        centre_x, centre_y = courtLines.get_center_point().coords

        coords = []
        if across_middle:
            signed_distance_to_center = newFrm['x'] - centre_x
            newFrm['x'] = centre_x - signed_distance_to_center
        if across_net:
            signed_distance_to_center = newFrm['y'] - centre_y
            newFrm['y'] = centre_y - signed_distance_to_center
        
        flipStr = 'top' if self.name == 'bot' else 'bot'
        
        return PlayerMovement(newFrm, name = flipStr)
    
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
    
    def get_hit_event_info(self):
        return self['event_info'].iloc[0]

    def get_hit_player(self):
        hit_event = self.get_hit_event()
        if hit_event[1] == 'T':
            return 'top'
        elif hit_event[1] == 'B':
            return 'bot'
    
    def get_rec_player(self):
        hit_event = self.get_hit_event()
        if hit_event[1] == 'T':
            return 'bot'
        elif hit_event[1] == 'B':
            return 'top'
            
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
        return x, y, z
    
    def get_bounce_positions(self):
        coord_cols = ['x_t', 'y_t', 'z_t']
        bounce_frames = self.get_bounce_frames()
        
        bounce_points = []
        for frm in bounce_frames:
            x, y, z = self.trajFrm.loc[frm, coord_cols]
            bounce_points.append((x, y, z))
        return bounce_points
    
    def get_first_bounce_position(self):
        try:
            x, y, z = self.get_bounce_positions()[0]
        except IndexError:
            x, y, z = None, None, None
        return x, y, z

    def get_initial_height(self, how = 'trajectory'):
        return self.get_initial_position(how = how).z
    
    def get_ball_array(self, how = 'trajectory'):
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
        
        coordArr = self.trajFrm[coord_cols].values
        return coordArr
    
    def stream(self, cap, P, how = 'trajectory'):
        coordArr = self.get_ball_array(how = how)
        oneArr = np.ones(len(self.trajFrm))[:, np.newaxis]
        homArr = np.concatenate([coordArr, oneArr], axis = 1)
        
        homArr_2D = np.einsum('ij, nj -> ni', P, homArr)
        pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
        ballFrm = pd.DataFrame(pixArr, index = self.trajFrm.index, columns = ['x', 'y'])
        
        ballSF = StreamFeatures.from_dataframe(ballFrm).apply(sa.coords_to_ball2D)
        return Stream.from_SF(ballSF, cap)
        
    def flip(self, courtLines = None, across_net = True, across_middle = True):
        if courtLines is None:
            courtLines = CourtLines.make()
        
        newFrm = self.trajFrm.copy()
        center_x, center_y = courtLines.get_center_point().coords
        
        if across_middle:
            signed_distance_to_center = self.trajFrm[['x_t', 'x_o']] - center_x
            newFrm[['x_t', 'x_o']] = center_x - signed_distance_to_center

        if across_net:
            signed_distance_to_center = self.trajFrm[['y_t', 'y_o']] - center_y
            newFrm[['y_t', 'y_o']] = center_y - signed_distance_to_center

        newFrm['event'] = newFrm['event'].replace({'STD': 'SBD', 'STA': 'SBA', 
                                                   'SBD': 'STD', 'SBA': 'STA',
                                                   'HT': 'HB', 'HB': 'HT'})
            
        return Trajectory(newFrm)
    
    ### TRAJECTORY QUALITY MEASURES
    def mean_deviation(self, dof_correction = 0):
        obsFrm = self.trajFrm[['x_o', 'y_o', 'z_o']]
        obsFrm.columns = ['x', 'y', 'z']

        trjFrm = self.trajFrm[['x_t', 'y_t', 'z_t']]
        trjFrm.columns = ['x', 'y', 'z']
        
        diffFrm = (trjFrm - obsFrm).dropna()
        distFrm = (diffFrm ** 2).sum(axis = 1) ** 0.5
        
        return distFrm.sum() / max(len(distFrm) - dof_correction, 0)
        
    ### SHAPE GETTERS ######
    # -> here, we have all functions that return a 2D or 3D Shape
    def shape_initial_position(self, how = 'trajectory'):
        return Point3D(self.get_initial_position(how = how))
    
    def shape_bounce_positions(self, how = 'trajectory'):
        return [Point3D(pt) for pt in self.get_bounce_positions(how = how)]
        
    #### FEATURE GETTERS ######
    def feat_ball(self, how = 'trajectory'):
        ### how: trajectory / observed
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
           
        ballFrm = self.trajFrm[coord_cols]
        ballFrm.columns = ['x', 'y', 'z']
        ballSF = StreamFeatures.from_dataframe(ballFrm).apply(sa.coords_to_ball3D)
        return ballSF
    
    def feat_spline(self, how = 'trajectory'):
        ballSF = self.feat_ball(how = how)
        ballSF_before = ballSF.shift(offset = 1)
        def make_spline(ball2D_bef1, ball2D):
            if ball2D_bef1 is None:
                spLn = None
            else:
                spLn = Line3D(ball2D_bef1, ball2D)
                return spLn
        return ballSF_before.merge(ballSF).apply(make_spline)
    
    
@dataclass(frozen = True)
class Shot:
    shotTraj: Trajectory
    hitPlayer: PlayerMovement
    recPlayer: PlayerMovement
    
    @staticmethod
    def from_shot_frame_and_player_sfs(shotFrm, topSF, botSF):
        shotFrm = shotFrm.dropna(how = 'all')
        shot_frames = shotFrm.index
        shotTraj = Trajectory(shotFrm)
        
        frmDict = {'top': topSF.conform(shot_frames).to_dataframe(),
                   'bot': botSF.conform(shot_frames).to_dataframe()}
        hit_player = shotTraj.get_hit_player()
        rec_player = shotTraj.get_rec_player()
        
        hitMvmt = PlayerMovement(frmDict[hit_player], name = hit_player)
        recMvmt = PlayerMovement(frmDict[rec_player], name = rec_player)
    
        return Shot(shotTraj, hitMvmt, recMvmt)
        
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
    
    def get_hit_event_info(self):
        return self.shotTraj.get_hit_event_info()
    
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
    
    def get_first_bounce_position(self):
        return self.shotTraj.get_first_bounce_position()
    
    def get_num_bounces(self):
        return self.shotTraj.get_num_bounces()
    
    def flip(self, courtLines = None, across_net = True, across_middle = True):
        if courtLines is None:
            courtLines = CourtLines.make()
            
        return Shot(self.shotTraj.flip(courtLines = courtLines, across_net = across_net, across_middle = across_middle),
                    self.hitPlayer.flip(courtLines = courtLines, across_net = across_net, across_middle = across_middle),
                    self.recPlayer.flip(courtLines = courtLines, across_net = across_net, across_middle = across_middle))
    
    ### FILTERS ###
    def is_serve(self):
        return self.get_hit_event()[0] == 'S'
    
    def is_serve_ad(self):
        if self.is_serve():
            return self.get_hit_event()[2] == 'A'
        else:
            return False
        
    def is_serve_deuce(self):
        if self.is_serve():
            return self.get_hit_event()[2] == 'D'
        else:
            return False
            
    def is_let(self):
        if self.is_serve():
            return self.get_hit_event_info() == 'let'
        else:
            return False
    
    def is_first_serve(self):
        if self.is_serve():
            return self.get_hit_event_info() == 'first_serve'
        else:
            return False
    
    def is_second_serve(self):
        if self.is_serve():
            return self.get_hit_event_info() == 'second_serve'
        else:
            return False

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
        ### TODO: replace this with: recognition of a shot as serve even though the shot_idx != 0
        ### (hitFlag == 'top_ovh' or 'bot_ovh')
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
    
    def is_top_to_bot(self):
        return self.get_hit_event()[1] == 'T'
    
    def is_bot_to_top(self):
        return self.get_hit_event()[1] == 'B'
    
    #### 
    def mean_deviation(self, dof_correction = 0):
        return self.shotTraj.mean_deviation(dof_correction = dof_correction)
    
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
    
    def __repr__(self):
        fsShot, lsShot = self[0], self[-1]
        num_shots = len(self)
        min_frame, max_frame = fsShot.get_frames()[0], lsShot.get_frames()[-1]
        return f'Shots(N = {num_shots}, between frames {min_frame} and {max_frame})'
    
    def __iter__(self):
        return iter(self.shotList)
    
    def __getitem__(self, item):
        return self.shotList[item]
    
    def __len__(self):
        return len(self.shotList)
    
    def filter(self, bool_method, is_below = None, is_above = None, **filter_kwargs):
        def filter_func(myShot):
            val = getattr(myShot, bool_method)(**filter_kwargs)
            if is_below is None and is_above is None:
                myBool = val
            elif is_below is None:
                myBool = val > is_above
            elif is_above is None:
                myBool = val < is_below
            else:
                myBool = is_above < val < is_below
            return myBool
        
        return Shots([myShot for myShot in self if filter_func(myShot)])
     
    def flip(self, courtLines = None, across_middle = True, across_net = True):
        if courtLines is None:
            courtLines = CourtLines.make()
        return Shots([myShot.flip(courtLines = courtLines, across_middle = across_middle, across_net = across_net) for myShot in self])
    
    def get_first_bounce_position(self):
        return [myShot.get_first_bounce_position() for myShot in self]
        
    def feat_ball_flat(self, how = 'trajectory'):
        ball_arrays = [myShot.shotTraj.get_ball_array(how = how) for myShot in self]
        longest_array_len = np.max([arr.shape[0] for arr in ball_arrays])
        
        padded_ball_arrays = [np.pad(arr, ((0, longest_array_len - arr.shape[0]), (0, 0)), constant_values = np.nan) for arr in ball_arrays]
        stacked_ball_arrays = np.stack(padded_ball_arrays, axis = 1) ### shape: time, traj, 3
    
        outDict = {}
        for idx, frameArr in enumerate(stacked_ball_arrays):
            frameArr = frameArr[~np.isnan(frameArr).all(axis = 1)]
            outDict[idx] = Point3D(frameArr.T)
    
        return StreamFeature(outDict, name = 'ball3D')
    
    # ### FEATURE GETTERS ######
    ### done with make_method below!!!
        
@dataclass(frozen = True)
class Point(Shots):
    ### every point is a collection of shots in sequence!
    
    @staticmethod
    def from_shot_frame_and_player_SFs(trajFrm, topSF, botSF):
        if trajFrm.dropna().empty:
            return Point([])
        
        last_frame = trajFrm.dropna(how = 'all').index[-1]
        hit_serve_filter = trajFrm['event'].str.startswith('S') | trajFrm['event'].str.startswith('H')
        hit_serve_frames = hit_serve_filter[hit_serve_filter].index
        next_frames = [*hit_serve_frames[1:], last_frame]
        
        shots = []
        for hit_frame, next_hit_frame in zip(hit_serve_frames, next_frames):
            shotFrm = trajFrm.loc[hit_frame: next_hit_frame - 1] 
            myShot = Shot.from_shot_frame_and_player_sfs(shotFrm, topSF, botSF)    
            shots.append(myShot)
            
        return Point(shots)
            
    def __repr__(self):
        serveShot = self.shotList[0]
        server, receiver = serveShot.hitPlayer.name, serveShot.recPlayer.name
        events = self.get_events()
        frames = self.get_frames()
        event_string = '->'.join(events)
        min_frame, max_frame = frames[0], frames[-1]
        return f'Point({event_string}, serving: {server}, receiving: {receiver} from {min_frame} to {max_frame})'
    
    def __iter__(self):
        return iter(self.shotList)
    
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
        
    def get_serve(self):
        if len(self) == 0:
            return None
        else:
            fsShot = self[0]
            if fsShot.is_serve():
                return fsShot
            else:
                return None
    
@dataclass(frozen = True)
class Points:
    pointList: List[Point]
    name: str = 'Points'
    
    def from_SFs(trajSFs, topSFs, botSFs):
        ### these have to be segmented!
        pointList = []
        
        for idx in range(trajSFs.length()):
            trajSF, topSF, botSF = trajSFs[idx], topSFs[idx], botSFs[idx]
            trajFrm = trajSF.to_dataframe()
            
            myPnt = Point.from_shot_frame_and_player_SFs(trajFrm, topSF, botSF)
            pointList.append(myPnt)
            
        return Points(pointList)
    
    def __iter__(self):
        return iter(self.pointList)
    
    def __getitem__(self, item):
        return self.pointList[item]
    
    def __len__(self):
        return len(self.pointList)
    
    def filter(self, bool_method):
        return Points([myPnt for myPnt in self if getattr(myPnt, bool_method)()])
    
    def flatten(self):
        ### turn into shots
        return Shots([myShot for myPnt in self for myShot in myPnt])
    
  
    
    
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
    
        