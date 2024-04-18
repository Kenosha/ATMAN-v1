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
from ATMANpy.types.Shapes3D import Point3D, Circle3D, Ball3D, Line3D, Points3D, Circles3D, Balls3D, Lines3D
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
    
    def __repr__(self):
        v = self.get_initial_speed()
        rpm = self.get_initial_spin()
        evt = self.get_hit_event()

        return f'Trajectory({evt}: {v:.0f}km/h, {rpm:.0f} rpm)'

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
    
    def get_initial_horizontal_speed(self, unit = 'km/h'):
        assert unit in ['m/s', 'km/h']
        speed_ms = self['vx0'].iloc[0]
        if unit == 'km/h':
            return speed_ms * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)
        
    def get_initial_vertical_speed(self, unit = 'km/h'):
        assert unit in ['m/s', 'km/h']
        speed_ms = self['vz0'].iloc[0]
        if unit == 'km/h':
            return speed_ms * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)

    def get_initial_speed(self, unit = 'km/h'):
        vx, vz = self.get_initial_horizontal_speed(unit = unit), self.get_initial_vertical_speed(unit = unit) 
        return (vx ** 2 + vz ** 2) ** 0.5
    
    def get_initial_spin(self):
        return self['rpm0'].iloc[0]

    def get_ball_array(self, how = 'trajectory'):
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
        
        coordArr = self.trajFrm[coord_cols].values
        return coordArr
    
    def get_extrapolated_next_hit_position(self):
        coord_cols = ['x_t', 'y_t', 'z_t']
        tFrm = self.trajFrm[coord_cols]
        ser0, ser1, ser2 = tFrm.iloc[-4], tFrm.iloc[-3], tFrm.iloc[-2] ### note that -1 is the next hit
        
        speed2 = ser2 - ser1
        speed1 = ser1 - ser0
        accel = speed2 - speed1

        outSer = ser2 + speed2 + accel
        return outSer.values
    
    def get_actual_next_hit_position(self):
        coord_cols = ['x_t', 'y_t', 'z_t']
        tFrm = self.trajFrm[coord_cols]
        return tFrm.iloc[-1].values

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
            newFrm['event'] = newFrm['event'].replace({'STD': 'STA', 'STA': 'STD',
                                                       'SBD': 'SBA', 'SBA': 'SBD'})

        if across_net:
            signed_distance_to_center = self.trajFrm[['y_t', 'y_o']] - center_y
            newFrm[['y_t', 'y_o']] = center_y - signed_distance_to_center
            newFrm['event'] = newFrm['event'].replace({'STD': 'SBA', 'STA': 'SBD', 
                                                        'SBD': 'STA', 'SBA': 'STD',
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
    def feat_ball(self, how = 'trajectory', which_endpoint = 'smooth'):
        ### how: trajectory / observed
        ### which_endpoint: 'extrapolated' or 'actual' or 'smooth'
            # 1: extrapolated -> extend ball flight trajectory by one frame
            # 2: actual -> use actual last entry in trajFrm, which is hit position of next hit (discontinuous with rest of trajectory!) 
            # 3: smooth -> interpolate between extrapolated point and actual point smoothly, so that you arrive at next hit point without great discontinuity.
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
           
        if which_endpoint == 'actual':
            ballFrm = self.trajFrm[coord_cols]

        if which_endpoint == 'extrapolated':
            ballFrm = self.trajFrm[coord_cols]
            ballFrm.iloc[-1] = self.get_extrapolated_next_hit_position()

        if which_endpoint == 'smooth':
            ballFrm = self.trajFrm[coord_cols]
            ballFrm.iloc[-1] = self.get_extrapolated_next_hit_position()

            ### distance between where we want to get to smoothly (actual) and where we would get to if we didn't do anything (extrapolated)
            d = self.get_actual_next_hit_position() - self.get_extrapolated_next_hit_position()

            ### Calculate beginning and end of interpolation. We use either the first bounce or (if not there), the hit.
            ### note that the zero_frame and one_frame are interpolated with 
            ### lambda = 0 and 1 respectively (trivial solutions), 
            ### but we still include them here.
            try:
                zero_frame = self.get_bounce_frames()[0]
            except IndexError:
                zero_frame = self.get_hit_frame()

            one_frame = self.get_frames()[-1]
        
            itpl_lambda = np.linspace(0, 1, endpoint = True, num = one_frame - zero_frame + 1)
            itpl_lambda_full = np.concatenate([np.zeros(len(ballFrm) - len(itpl_lambda)), itpl_lambda])
            
            ballFrm = ballFrm + itpl_lambda_full[:, np.newaxis] * d[np.newaxis, :]

        ballFrm.columns = ['x', 'y', 'z']
        ballSF = StreamFeatures.from_dataframe(ballFrm).apply(sa.coords_to_ball3D)
        return ballSF
    
    def feat_spline(self, how = 'trajectory', which_endpoint = 'smooth'):
        ballSF = self.feat_ball(how = how, which_endpoint = which_endpoint).rename('ball_curr')
        ballSF_before = ballSF.shift(offset = 1).rename('ball_bef')
        def make_spline(ball_bef, ball_curr):
            if ball_bef is None:
                spLn = None
            else:
                spLn = Line3D(ball_bef.get_point(), ball_curr.get_point())
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
        return self.shotTraj.get_hit_frame()
    
    def get_hit_event(self):
        return self.shotTraj.get_hit_event()
    
    def get_hit_event_info(self):
        return self.shotTraj.get_hit_event_info()
    
    def get_hit_series(self):
        return self.shotTraj.get_hit_series()
    
    def get_player_hit_position(self):
        return self.hitPlayer.get_initial_position()
    
    def get_ball_hit_position(self):
        return self.shotTraj.get_initial_position()
    
    def get_ball_hit_height(self):
        return self.shotTraj.get_initial_height()
    
    def get_initial_horizontal_speed(self, unit = 'km/h'):
        return self.shotTraj.get_initial_horizontal_speed(unit = unit)
        
    def get_initial_vertical_speed(self, unit = 'km/h'):
        return self.shotTraj.get_initial_vertical_speed(unit = unit)

    def get_initial_speed(self, unit = 'km/h'):
        return self.shotTraj.get_initial_speed(unit = unit)
    
    def get_initial_spin(self):
        return self.shotTraj.get_initial_spin()
    
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
        
    def feat_spline(self, how = 'trajectory', which_endpoint = 'smooth'):
        return self.shotTraj.feat_spline(how = how, which_endpoint = which_endpoint)
        
@dataclass(frozen = True)
class Shots:
    ### a general collection of shots.
    shotList: List[Shot]

    # def __repr__(self):
    #     fsShot, lsShot = self[0], self[-1]
    #     num_shots = len(self)
    #     min_frame, max_frame = fsShot.get_frames()[0], lsShot.get_frames()[-1]
    #     return f'Shots(N = {num_shots}, between frames {min_frame} and {max_frame})'

    def __repr__(self):
        shotStrs = [f'  {i}: {s},' for i, s in enumerate(self)]
        innerStr = '\n'.join(shotStrs)
        return f'Shots([\n{innerStr}\n])'

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
        return self.__class__([myShot.flip(courtLines = courtLines, across_middle = across_middle, across_net = across_net) for myShot in self])
    
    def join(self, otherShots):
        return Shots([*self, *otherShots])

    def sort(self):
        keyFunc = lambda el: el.get_hit_frame()
        return Shots(sorted(self.shotList, key = keyFunc))
    

    ### STAT GETTERS ###
    # -> .get methods of Shot, done with make_method below
        
    ### FEATURE GETTERS ######
    # -> .feat methods of Shot, done with make_method below!!!
        
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
            shotFrm = trajFrm.loc[hit_frame: next_hit_frame] 
            myShot = Shot.from_shot_frame_and_player_sfs(shotFrm, topSF, botSF)    
            shots.append(myShot)
            
        return Point(shots)
            
    def __repr__(self):
        if len(self.shotList) == 0:
            return 'Point()'
        else:
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
            
    def get_return(self):
        if len(self) <= 1:
            return None
        elif not self.is_serve():
            return None
        else:
            return self[1]

    def is_serve(self):
        return self.get_serve() is not None
    
    def is_serve_fair(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_fair(nextShot = None)
        else:
            return False

    def is_serve_ad(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_serve_ad()
        else:
            return False

    def is_serve_deuce(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_serve_deuce()
        else:
            return False
           
    def is_first_serve(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_first_serve()
        else:
            return False
    
    def is_second_serve(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_second_serve()
        else:
            return False

    def is_top_to_bot(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_top_to_bot()
        else:
            return False
    
    def is_bot_to_top(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.is_bot_to_top()
        else:
            return False
    
@dataclass(frozen = True)
class Points:
    pointList: List[Point]
    name: str = 'Points'

    @staticmethod
    def from_SFs(trajSFs, topSFs, botSFs):
        ### these have to be segmented!
        pointList = []
        
        for idx in range(trajSFs.length()):
            trajSF, topSF, botSF = trajSFs[idx], topSFs[idx], botSFs[idx]
            trajFrm = trajSF.to_dataframe()
            
            myPnt = Point.from_shot_frame_and_player_SFs(trajFrm, topSF, botSF)
            pointList.append(myPnt)
            
        return Points(pointList)
    
    def __repr__(self):
        pointStrs = [f'  {i}: {p},' for i, p in enumerate(self)]
        innerStr = '\n'.join(pointStrs)
        return f'Points([\n{innerStr}\n])'

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
    
    def flip(self, courtLines = None, across_middle = True, across_net = True):
        if courtLines is None:
            courtLines = CourtLines.make()
        return Points([myPnt.flip(courtLines = courtLines, across_middle = across_middle, across_net = across_net) for myPnt in self])
    
    def join(self, otherPnts):
        return Points([*self, *otherPnts])
    
    def sort(self):
        keyFunc = lambda el: el.get_serve().get_hit_frame()
        return Points(sorted(self.pointList, key = keyFunc))
    
    def enforce_direction(self, from_where):
        ### TODO: ADD SORT!
        assert from_where in ['Top', 'Bot']
        t2bPnts = self.filter('is_top_to_bot')
        b2tPnts = self.filter('is_bot_to_top')

        if from_where == 'Top':
            return t2bPnts.join(b2tPnts.flip())
        elif from_where == 'Bot':
            return b2tPnts.join(t2bPnts.flip())
        
    def get_serves(self):
        pntList = [myPnt.get_serve() for myPnt in self]
        return Shots([p for p in pntList if p is not None])

    def get_returns(self):
        pntList = [myPnt.get_return() for myPnt in self]
        return Shots([p for p in pntList if p is not None])

#### CREATE FEATURE GETTERS FOR POINT, SHOTS
def __make_method_feat(myClass, shotMethodStr): ### helper to automatically make all "feat_" methods that exist in Shot for Shots also.
    def method(self, color = None, alpha = None, **kwargs):
        if color is None:
            color = const.BGR['bright_grey']
        if alpha is None:
            alpha = 1

        SFs = SFList([getattr(myShot, shotMethodStr)(**kwargs) for myShot in self])
        return SFs.concatenate()
    method.__name__ = shotMethodStr
    setattr(myClass, shotMethodStr, method)
    
def __make_method_feat_simul(myClass, shotMethodStr):
    def method(self, color = None, alpha = None, **kwargs):
        if color is None:
            color = const.BGR['bright_grey']
        if alpha is None:
            alpha = 1

        SFs = SFList([getattr(myShot, shotMethodStr)(**kwargs) for myShot in self])
        SFs = SFs.start_at_zero()
        name = SFs[0].name
        SFs = SFList([sf.rename(f'{name}_{idx}') for idx, sf in enumerate(SFs)])
        jntSF = SFs.join()
        vecDict = {}    
        for frame in jntSF.get_frames():
            featDict = jntSF[frame]
            shapes = [shape for shape in featDict.values() if shape is not None]
            if len(shapes) == 0:
                vecDict[frame] = None
            else:
                containerClass = shapes[0].__class__.get_container()
                vecDict[frame] = containerClass(shapes, color = color, alpha = alpha)
        return StreamFeature(vecDict, name = name)
                             
    method.__name__ = shotMethodStr + '_simul'
    setattr(myClass, shotMethodStr + '_simul', method)

def __make_method_get(myClass, shotMethodStr):
    def method(self, **kwargs):
        outList = [getattr(myShot, shotMethodStr)(**kwargs) for myShot in self]
        try:
            return np.array(outList)
        except: ### sometimes, outList is a list of lists of unequal size
            return outList

    method.__name__ = shotMethodStr
    setattr(myClass, shotMethodStr, method)

for methodStr in dir(Shot):
    if methodStr.startswith('feat_'):
        ### feat getters
        if not methodStr in dir(Shots): ### do not overwrite
            __make_method_feat(Shots, methodStr)
        if not methodStr in dir(Point): ### do not overwrite
            __make_method_feat(Point, methodStr)

        ### simul feat getters
        if not methodStr + '_simul' in dir(Shots): ### do not overwrite
            __make_method_feat_simul(Shots, methodStr)
        if not methodStr + '_simul' in dir(Point): ### do not overwrite
            __make_method_feat_simul(Point, methodStr)
    
    if methodStr.startswith('get_'):
        ### normal getters
        if not methodStr in dir(Shots): ### do not overwrite
            __make_method_get(Shots, methodStr)
        if not methodStr in dir(Point): ### do not overwrite
            __make_method_get(Point, methodStr)