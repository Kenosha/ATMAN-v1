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
from ATMANpy.types.Streams import Stream, StreamList
from ATMANpy.types.Features import StreamFeature, StreamFeatures, SFList
from ATMANpy.types.Shapes3D import Point3D, Circle3D, Ball3D, Line3D, Points3D, Circles3D, Balls3D, Lines3D
from ATMANpy.types.ShapesTennis3D import Player
from ATMANpy.types.Lines import Line
from ATMANpy.types.Points import Point2D

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.sf_apply_funcs as sa
import ATMANpy.funcs.util_funcs as ut
import ATMANpy.funcs.feature_funcs as ff

from ATMANpy.models.ScoreModel import ScoreModel
P_SERVER = 0.65
SETS_TO_WIN = 2
scoreMod = ScoreModel(P_SERVER, SETS_TO_WIN)

@dataclass(frozen = True)
class PlayerMovement:
    playerFrm: pd.DataFrame
    
    @staticmethod
    def from_features(ply3DSF, which):
        ### which: top, bot
        player_features = ply3DSF.get_features()
    
        trunc_features = [f[:3] for f in player_features] ### truncated features -> keep top or bot
        chosen_feature = player_features[trunc_features.index(which)]
        
        playerFrm = ply3DSF[chosen_feature].apply(sa.point3D_to_coords).to_dataframe()
        return PlayerMovement(playerFrm)
    
    def __getitem__(self, item):
        if item in self.playerFrm.columns:
            return self.playerFrm[item]
        else:
            return self.playerFrm.loc[item] ### TODO: maybe as point?
        
    def get_frames(self):
        return self.playerFrm.frame_num.to_list()
    
    def get_num_frames(self):
        return len(self.get_frames())
    
    def get_initial_position(self):
        x, y = self.playerFrm.loc[0, ['x', 'y']]
        # return Point3D((x, y, 0))
        return x, y
    
    def get_name(self):
        return self.playerFrm.loc[0, 'name']
       
    def get_hand(self):
        return self.playerFrm.loc[0, 'hand']

    def get_side(self):
        return self.playerFrm.loc[0, 'side']
    
    def get_point_score(self):
        return str(self.playerFrm.loc[0, 'score']) ### just to be sure, make this a string! -> a score of '40' does not mean the integer 40, a score of 'A' is not an integer at all!
    
    def get_game_score(self):
        return int(self.playerFrm.loc[0, 'games']) ### these by contrast are integers
    
    def get_set_score(self):
        return int(self.playerFrm.loc[0, 'sets']) ### these by contrast are integers

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
            
        return PlayerMovement(newFrm)
    
    def feat_player(self):
        def applicator(x, y):
            coords = x, y, 0
            player = Player.make(coords)
            return player 
            
        sfFrm = self.playerFrm[['x', 'y']]
        sfFrm.index = self.playerFrm['frame_num']
        sfFrm = sfFrm.loc[sfFrm.index[:-1]] ### last element is the next player!

        plSF = StreamFeatures.from_dataframe(sfFrm).apply(applicator)
        plSF = plSF.rename('Player3D')
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
    
    def get_match_index(self):
        return int(self['match_num'].iloc[0])

    def get_point_index(self):
        return int(self['point_idx'].iloc[0])

    def get_shot_index(self):
        return int(self['shot_idx'].iloc[0])

    def get_frames(self):
        return self.trajFrm.frame_num.to_list()
    
    def get_num_frames(self):
        return len(self.get_frames())
    
    def get_num_frames_until_bounce(self):
        try:
            first_bounce_idx = self.get_bounce_indices()[0]
        except IndexError:
            return None
        else:
            return first_bounce_idx
    
    def get_hit_series(self):
        return self.trajFrm.iloc[0]
        
    def get_hit_event(self):
        return self['event'].iloc[0]
        
    def get_hit_event_info(self):
        return self['event_info'].iloc[0]

    def get_hit_event_descriptor(self):
        return self['event_descriptor'].iloc[0]

    def get_hit_side(self):
        hit_event = self.get_hit_event()
        if hit_event[1] == 'T':
            return 'Top'
        elif hit_event[1] == 'B':
            return 'Bot'
    
    def get_rec_side(self):
        hit_event = self.get_hit_event()
        if hit_event[1] == 'T':
            return 'Bot'
        elif hit_event[1] == 'B':
            return 'Top'
            
    def get_bounce_events(self):
        return self['event'][self['event'].str.startswith('B')].values.tolist()
    
    def get_bounce_indices(self):
        return self['event'][self['event'].str.startswith('B')].index.tolist()
    
    def get_num_bounces(self):
        return len(self.get_bounce_events())
    
    def is_in_net(self):
        return (self['event'] == 'N').any()

    def get_events(self):
        if self.is_in_net():
            return [self.get_hit_event(), *self.get_bounce_events(), 'N'] ### TODO: can not deal with net cord right now!
        else:
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
        bounce_indices = self.get_bounce_indices()
        
        bounce_points = []
        for idx in bounce_indices:
            x, y, z = self.trajFrm.loc[idx, coord_cols]
            bounce_points.append((x, y, z))
        return bounce_points
    
    def get_first_bounce_position(self):
        try:
            x, y, z = self.get_bounce_positions()[0]
        except IndexError:
            x, y, z = None, None, None
        return x, y ### z = 0 always!
    
    def get_virtual_planar_bounce_point(self):
        from ATMANpy.models.BallTrajectory.flight_bounce_model import calculate_speed_step

        try:
            bounce_idx =  self.trajFrm[self.trajFrm.event.str.startswith('B')].index[0]
        except IndexError:
            return None, None
        
        px0, pz0 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx - 2]
        px1, pz1 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx - 1]
        rpm = self.get_initial_spin()

        dx, dz, _ = calculate_speed_step((px1 - px0), (pz1 - pz0), rpm)
        vx, vz = px1 + dx, pz1 + dz
        return vx, vz
    
    def get_net_crossing_position(self, net_y = 637):
        coord_cols = ['x_t', 'y_t', 'z_t']
        trajArr = self.trajFrm[coord_cols].dropna()
        if self.get_hit_side() == 'Bot':
            is_after_net = trajArr['y_t'] <= net_y
        else:
            is_after_net = trajArr['y_t'] >= net_y

        past_net_idx = np.argmax(is_after_net)
        assert past_net_idx > 0

        x0, y0, z0 = trajArr.loc[past_net_idx - 1]
        x1, y1, z1 = trajArr.loc[past_net_idx]

        lmbd = (net_y - y0) / (y1 - y0)
        itpl_func = lambda a, b: (1 - lmbd) * a + lmbd * b

        x, y, z = itpl_func(x0, x1), net_y, itpl_func(z0, z1)
        return x, y, z
    
    def get_flight_line(self):
        hit_x, hit_y, _ = self.get_initial_position()
        lst_x, lst_y, _ = self.get_last_hit_position()

        flightLn = Line(Point2D((hit_x, hit_y)), Point2D((lst_x, lst_y)))
        return flightLn

    def get_initial_width(self, courtLines = None):
        hit_x = self.get_initial_position()[0]
        cnt_x = courtLines.get_center_point().x
        return np.abs(hit_x - cnt_x)
    
    def get_initial_depth(self, courtLines = None):
        ### depth of hit, measured relative to the baseline.
        hit_y = self.get_initial_position()[1]

        if courtLines is None:
            courtLines = CourtLines.make()

        if self.is_top_to_bot():
            base_y = courtLines['Top Baseline'].center().y
            return base_y - hit_y
        elif self.is_bot_to_top():
            base_y = courtLines['Bottom Baseline'].center().y
            return hit_y - base_y
        
    def get_initial_height(self, how = 'trajectory'):
        return self.get_initial_position(how = how)[2]
    
    def get_bounce_distance_from_baseline(self, courtLines = None):
        _, bnc_y = self.get_first_bounce_position()
        if bnc_y is None:
            return None
        
        if courtLines is None:
            courtLines = CourtLines.make()
        
        if self.is_top_to_bot():
            base_y = courtLines['Bottom Baseline'].center().y
            return base_y - bnc_y ### positive: in front of baseline; negative: behind baseline
        elif self.is_bot_to_top():
            base_y = courtLines['Top Baseline'].center().y
            return bnc_y - base_y

    def get_initial_horizontal_speed(self, unit = 'km/h'):
        assert unit in ['m/s', 'km/h']
        speed_ms = self['vx0'].iloc[0]
        if unit == 'km/h':
            return speed_ms * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)
        else:
            return speed_ms
        
    def get_initial_vertical_speed(self, unit = 'km/h'):
        assert unit in ['m/s', 'km/h']
        speed_ms = self['vz0'].iloc[0]
        if unit == 'km/h':
            return speed_ms * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)
        else:
            return speed_ms
    
    def get_post_bounce_horizontal_speed(self, unit = 'km/h', how = 'virtual'):
        assert unit in ['m/s', 'km/h']
        assert how in ['virtual', 'corrected', 'corrected_next']
        
        try:
            bounce_idx =  self.trajFrm[self.trajFrm.event.str.startswith('B')].index[0]
        except IndexError:
            return None
        
        if how == 'virtual':
            px0, _  = self.get_virtual_planar_bounce_point()
            px1     = self.trajFrm['px_t'].iloc[bounce_idx + 1]
        elif how == 'corrected':
            px0     = self.trajFrm['px_t'].iloc[bounce_idx]
            px1     = self.trajFrm['px_t'].iloc[bounce_idx + 1]
        elif how == 'corrected_next':
            px0     = self.trajFrm['px_t'].iloc[bounce_idx + 1]
            px1     = self.trajFrm['px_t'].iloc[bounce_idx + 2]

        speed_ms = px1 - px0
        if unit == 'km/h':
            return speed_ms * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)
        else:
            return speed_ms
    
    def get_post_bounce_vertical_speed(self, unit = 'km/h', how = 'virtual'):
        assert unit in ['m/s', 'km/h']
        assert how in ['virtual', 'corrected', 'corrected_next']
        
        try:
            bounce_idx =  self.trajFrm[self.trajFrm.event.str.startswith('B')].index[0]
        except IndexError:
            return None
        
        if how == 'virtual':
            _, pz0  = self.get_virtual_planar_bounce_point()
            pz1     = self.trajFrm['pz_t'].iloc[bounce_idx + 1]
        elif how == 'corrected':
            pz0     = self.trajFrm['pz_t'].iloc[bounce_idx]
            pz1     = self.trajFrm['pz_t'].iloc[bounce_idx + 1]
        elif how == 'corrected_next':
            pz0     = self.trajFrm['pz_t'].iloc[bounce_idx + 1]
            pz1     = self.trajFrm['pz_t'].iloc[bounce_idx + 2]

        speed_ms = pz1 - pz0
        if unit == 'km/h':
            return speed_ms * const.M_TO_KM / (const.SEC_TO_MIN * const.MIN_TO_HOUR)
        else:
            return speed_ms

    def get_launch_angle(self):
        h = self.get_initial_horizontal_speed()
        v = self.get_initial_vertical_speed()
        return np.arctan2(v, h) * const.RAD_TO_ANGLE
        
    def get_angle_post_bounce(self, how = 'virtual'):
        assert how in ['virtual', 'corrected', 'corrected_next']

        try:
            bounce_idx =  self.trajFrm[self.trajFrm.event.str.startswith('B')].index[0]
        except IndexError:
            return None
        
        if how == 'virtual':
            px0, pz0 = self.get_virtual_planar_bounce_point()
            px1, pz1 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx + 1]
        elif how == 'corrected':
            px0, pz0 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx]
            px1, pz1 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx + 1]
        elif how == 'corrected_next':
            px0, pz0 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx + 1]
            px1, pz1 = self.trajFrm[['px_t', 'pz_t']].iloc[bounce_idx + 2]

        rad = np.arctan2(pz1 - pz0, px1 - px0)
        return rad * const.RAD_TO_ANGLE

    def get_initial_speed(self, unit = 'km/h'):
        vx, vz = self.get_initial_horizontal_speed(unit = unit), self.get_initial_vertical_speed(unit = unit) 
        return (vx ** 2 + vz ** 2) ** 0.5
    
    def get_initial_spin(self):
        return self['rpm0'].iloc[0]
    
    def get_apex(self):
        tFrm = self.trajFrm[['x_t', 'y_t', 'z_t']]
        x, y, z = tFrm.iloc[tFrm.z_t.argmax()]
        return x, y, z
    
    def get_apex_post_bounce(self):
        try:
            bounce_idx =  self.trajFrm[self.trajFrm.event.str.startswith('B')].index[0]
        except IndexError:
            return None, None, None
        
        tFrm = self.trajFrm[['x_t', 'y_t', 'z_t']].iloc[bounce_idx:]
        x, y, z = tFrm.iloc[tFrm.z_t.argmax()]
        return x, y, z
    
    def get_obs_ball_array(self):
        coord_cols = ['x_o', 'y_o', 'z_o']
        coordArr = self.trajFrm[coord_cols].values
        return coordArr
    
    def get_traj_ball_array(self):
        coord_cols = ['x_t', 'y_t', 'z_t']
        coordArr = self.trajFrm[coord_cols].dropna().values ### dropna because net ball have nans in their trajectory
        return coordArr
        
    def get_last_hit_position(self):
        tArr = self.get_traj_ball_array()
        return tArr[-2]

    def get_actual_next_hit_position(self):
        tArr = self.get_traj_ball_array()
        return tArr[-1]
    
    def get_extrapolated_next_hit_position(self):
        tArr = self.get_traj_ball_array()
        ser0, ser1, ser2 = tArr[-4], tArr[-3], tArr[-2] ### note that -1 is the next hit
        
        speed2 = ser2 - ser1
        speed1 = ser1 - ser0
        accel = speed2 - speed1

        outArr = ser2 + speed2 + accel
        return outArr

    def get_capture(self, confs):
        idx = self.get_match_index()
        return confs[idx]['cap']

    def is_top_to_bot(self):
        return self.get_hit_event()[1] == 'T'
    
    def is_bot_to_top(self):
        return self.get_hit_event()[1] == 'B'
    
    def stream(self, cap = None, confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        if cap is None:
            cap = self.get_capture(confs)

        fs, ls = self.get_frames()[0], self.get_frames()[-1]
        return Stream(cap, fs, ls - 1) ### last frame is next shot frame

    def stream_ball(self, P, how = 'trajectory', cap = None, confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        if cap is None:
            cap = self.get_capture(confs)

        if how == 'trajectory':
            coordArr = self.get_traj_ball_array()
        elif how == 'observed':
            coordArr = self.get_obs_ball_array()

        oneArr = np.ones(len(self.trajFrm))[:, np.newaxis]
        homArr = np.concatenate([coordArr, oneArr], axis = 1)
        
        homArr_2D = np.einsum('ij, nj -> ni', P, homArr)
        pixArr = homArr_2D[:, :2] / homArr_2D[:, 2][:, np.newaxis]
        ballFrm = pd.DataFrame(pixArr, index = self.trajFrm.frame_num, columns = ['x', 'y'])
        ballFrm = ballFrm.iloc[:-1] ### last ball is duplicated!

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
        
    def reanchor(self, anchorPt):
        newFrm = self.trajFrm.copy()
        obs_cols = ['x_o', 'y_o', 'z_o']
        trj_cols = ['x_t', 'y_t', 'z_t']

        anchor_coords = anchorPt.coords
        obs_start_coords = self.get_initial_position(how = 'observed')
        trj_start_coords = self.get_initial_position(how = 'trajectory')

        newFrm[obs_cols] = newFrm[obs_cols] - obs_start_coords + anchor_coords
        newFrm[trj_cols] = newFrm[trj_cols] - trj_start_coords + anchor_coords
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
            # 4: penultimate -> remove the last frame completely, which is the point at which the ball is hit by the next guy
        if how == 'trajectory':
            coord_cols = ['x_t', 'y_t', 'z_t']
        elif how == 'observed':
            coord_cols = ['x_o', 'y_o', 'z_o']
           
        if which_endpoint == 'actual':
            ballFrm = self.trajFrm[coord_cols]

        if which_endpoint == 'penultimate':
            ballFrm = self.trajFrm[coord_cols]
            ballFrm = ballFrm.loc[ballFrm.index[:-1]]

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
                zero_idx = self.get_bounce_indices()[0]
            except IndexError:
                zero_idx = 0

            one_idx = self.trajFrm.index[-1]
        
            itpl_lambda = np.linspace(0, 1, endpoint = True, num = one_idx - zero_idx + 1)
            itpl_lambda_full = np.concatenate([np.zeros(len(ballFrm) - len(itpl_lambda)), itpl_lambda])
            
            ballFrm = ballFrm + itpl_lambda_full[:, np.newaxis] * d[np.newaxis, :]

        ballFrm.columns = ['x', 'y', 'z']
        if which_endpoint == 'penultimate':
            ballFrm.index = self.trajFrm.frame_num[:-1]
        else:
            ballFrm.index = self.trajFrm.frame_num

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
    
    def feat_spline_cumul(self, how = 'trajectory', which_endpoint = 'smooth'):
        splineSF = self.feat_spline(how = how, which_endpoint = which_endpoint)

        cumulSF = {}
        for frame in splineSF:
            spLn = splineSF[frame]
            try:
                prev_spline_list = cumulSF[frame - 1]
            except:
                cumulSF[frame] = [spLn]
            else:
                cumulSF[frame] = [*prev_spline_list, spLn]

        return StreamFeature(cumulSF, name = 'spLn_cumul')
    
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
        
        frmDict = {'Top': topSF.conform(shot_frames).to_dataframe(),
                   'Bot': botSF.conform(shot_frames).to_dataframe()}
        hit_player = shotTraj.get_hit_side()
        rec_player = shotTraj.get_rec_side()
        
        hitMvmt = PlayerMovement(frmDict[hit_player])
        recMvmt = PlayerMovement(frmDict[rec_player])
    
        return Shot(shotTraj, hitMvmt, recMvmt)
    
    @staticmethod
    def from_shot_table(shotFrm):
        ### reindex, to have a clean index that starts at zero! But keep the index in the frame, so we can play back the real thing.
        if len(shotFrm.index.names) == 1:
            id_indices = ['frame_num']
        elif len(shotFrm.index.names) == 2:
            id_indices = ['match_num', 'frame_num']
        else:
            raise NotImplementedError('More than two levels in shot Index!!')

        shotFrm.index.names = id_indices
        shotFrm = shotFrm.reset_index()    

        hitFrm = shotFrm[[*id_indices, *[c for c in shotFrm if 'hit_' in c]]]
        recFrm = shotFrm[[*id_indices, *[c for c in shotFrm if 'rec_' in c]]]

        trajFrm = shotFrm[[*id_indices, *[c for c in shotFrm if c not in hitFrm.columns and c not in recFrm.columns]]]
        shotTraj = Trajectory(trajFrm)
        
        hitFrm.columns = hitFrm.columns.str.replace('hit_', '')
        recFrm.columns = recFrm.columns.str.replace('rec_', '')
        hitMvmt = PlayerMovement(hitFrm)
        recMvmt = PlayerMovement(recFrm)

        return Shot(shotTraj, hitMvmt, recMvmt)

    def __post_init__(self):
        assert self.shotTraj.get_frames() == self.hitPlayer.get_frames() == self.recPlayer.get_frames()
    
    def __repr__(self):
        frames = self.get_frames()
        events = self.get_events()
        event_strings = '->'.join(events)
        hitStr, recStr = self.hitPlayer.get_name(), self.recPlayer.get_name()
        return f'Shot({event_strings}, hitting: {hitStr}, receiving: {recStr}. Duration: {len(frames)} frames)'

    def get_match_index(self):
        return self.shotTraj.get_match_index() 
    
    def get_point_index(self):
        return self.shotTraj.get_point_index() 
    
    def get_shot_index(self):
        return self.shotTraj.get_shot_index()

    def get_frames(self):
        return self.shotTraj.get_frames()
    
    def get_num_frames(self):
        return len(self.get_frames())
    
    def get_num_frames_until_bounce(self):
        return self.shotTraj.get_num_frames_until_bounce()

    def get_events(self):
        return self.shotTraj.get_events()
    
    def get_hit_frame(self):
        return self.shotTraj.get_hit_frame()
    
    def get_hit_event(self):
        return self.shotTraj.get_hit_event()
    
    def get_hit_side(self):
        return self.shotTraj.get_hit_side()
    
    def get_hit_event_info(self):
        return self.shotTraj.get_hit_event_info()
    
    def get_hit_event_descriptor(self):
        return self.shotTraj.get_hit_event_descriptor()
    
    def get_hit_series(self):
        return self.shotTraj.get_hit_series()
    
    def get_hit_player_name(self):
        return self.hitPlayer.get_name()
    
    def get_rec_player_name(self):
        return self.recPlayer.get_name()
    
    def get_player_hit_position(self):
        return self.hitPlayer.get_initial_position()
    
    def get_player_rec_position(self):
        return self.recPlayer.get_initial_position()

    def get_ball_hit_position(self):
        return self.shotTraj.get_initial_position()
    
    def get_ball_last_position(self):
        return self.shotTraj.get_last_hit_position()
    
    def get_ball_next_position(self):
        return self.shotTraj.get_actual_next_hit_position()
    
    def get_ball_hit_height(self):
        return self.shotTraj.get_initial_height()
    
    def get_ball_hit_depth(self, courtLines = None):
        return self.shotTraj.get_initial_depth(courtLines = courtLines)
    
    def get_ball_hit_width(self, courtLines = None):
        return self.shotTraj.get_initial_width(courtLines = courtLines)

    def get_bounce_distance_from_baseline(self, courtLines = None):
        return self.shotTraj.get_bounce_distance_from_baseline(courtLines = courtLines)

    def get_initial_horizontal_speed(self, unit = 'km/h'):
        return self.shotTraj.get_initial_horizontal_speed(unit = unit)
        
    def get_initial_vertical_speed(self, unit = 'km/h'):
        return self.shotTraj.get_initial_vertical_speed(unit = unit)

    def get_post_bounce_horizontal_speed(self, unit = 'km/h', how = 'virtual'):
        return self.shotTraj.get_post_bounce_horizontal_speed(unit = unit, how = how)

    def get_post_bounce_vertical_speed(self, unit = 'km/h', how = 'virtual'):
        return self.shotTraj.get_post_bounce_vertical_speed(unit = unit, how = how)

    def get_initial_speed(self, unit = 'km/h'):
        return self.shotTraj.get_initial_speed(unit = unit)
    
    def get_initial_spin(self):
        return self.shotTraj.get_initial_spin()

    def get_launch_angle(self):
        return self.shotTraj.get_launch_angle()

    def get_angle_post_bounce(self, how = 'virtual'):
        return self.shotTraj.get_angle_post_bounce(how = how)

    def get_apex(self):
        return self.shotTraj.get_apex()
    
    def get_apex_post_bounce(self):
        return self.shotTraj.get_apex_post_bounce()

    def get_bounce_events(self):
        return self.shotTraj.get_bounce_events()
    
    def get_bounce_positions(self):
        return self.shotTraj.get_bounce_positions()
    
    def get_first_bounce_position(self):
        return self.shotTraj.get_first_bounce_position()
    
    def get_net_crossing_position(self, net_y = 637):
        return self.shotTraj.get_net_crossing_position(net_y = net_y)
    
    def get_flight_line(self):
        return self.shotTraj.get_flight_line()

    def get_num_bounces(self):
        return self.shotTraj.get_num_bounces()
    
    def get_optimal_repositioning_position(self, courtLines = None,
                                           reposition_to_meters_behind_baseline = 1, shot_is_met_at_meters_behind_baseline = 1):
        ### calculate the optimal point for the hitting player to move to after the shot has been made

        ### get two court halves
        if courtLines is None:
            courtLines = CourtLines.make().add_singles_netline()
        botReg = courtLines.get_half_region(side = 'Bottom', width = 'Singles')
        topReg = courtLines.get_half_region(side = 'Top', width = 'Singles')

        flyLn = self.get_flight_line()

        # calculate, in points, how far behind the respective baselines the shots are expected to be made.
        pt_shift_repos = reposition_to_meters_behind_baseline * const.SCALE * const.METER_TO_FOOT
        pt_shift_hit = shot_is_met_at_meters_behind_baseline * const.SCALE * const.METER_TO_FOOT

        ### The line through which point neatly divides the court to cover into two parts?
        # using the center of the opposite service line basically assumes that 
        # the best angle shot the opponent can hit bounces at the intersection of the service line and the side line.
        if self.get_hit_side() == 'Top':
            from ATMANpy.auxil.court import top_srv_cnt_2D
            cntPt = top_srv_cnt_2D
        elif self.get_hit_side() == 'Bot':
            from ATMANpy.auxil.court import bot_srv_cnt_2D
            cntPt = bot_srv_cnt_2D
            
        if self.get_hit_side() == 'Top':
            ownLn = topReg['Top'].up(pt_shift_repos)
            metLn = botReg['Bottom'].down(pt_shift_hit)
        elif self.get_hit_side() == 'Bot':
            ownLn = botReg['Bottom'].down(pt_shift_repos)
            metLn = topReg['Top'].up(pt_shift_hit)
        
        ### calculate where the opposing player will meet the shot.
        intPt = flyLn.weak_intersect(metLn)

        ### the optimal point is opposite the intersection of the incoming ball's flight with the shifted baseline, on the original player's side.
        mirrorLn = Line(intPt, cntPt)
        optPt = mirrorLn.weak_intersect(ownLn)
        
        return optPt
    
    def get_point_score(self):
        if self.is_by_server():
            return self.hitPlayer.get_point_score(), self.recPlayer.get_point_score()
        else:
            return self.recPlayer.get_point_score(), self.hitPlayer.get_point_score()
           
    def get_game_score(self):
        if self.is_by_server():
            return self.hitPlayer.get_game_score(), self.recPlayer.get_game_score()
        else:
            return self.recPlayer.get_game_score(), self.hitPlayer.get_game_score()
    
    def get_set_score(self):
        if self.is_by_server():
            return self.hitPlayer.get_set_score(), self.recPlayer.get_set_score()
        else:
            return self.recPlayer.get_set_score(), self.hitPlayer.get_set_score()

    def get_score_string(self):
        srv_pt, rec_pt = self.get_point_score()
        srv_gm, rec_gm = self.get_game_score()
        srv_st, rec_st = self.get_set_score()

        return f'{srv_st}-{rec_st} {srv_gm}-{rec_gm} {srv_pt}-{rec_pt}'

    def get_point_score_pressure(self):
        srv_str, rec_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        return scoreMod.point_score_pressure(srv_str, rec_str, srv_game, rec_game)

    def get_game_score_pressure(self):
        srv, rec = self.get_game_score()
        return scoreMod.game_score_pressure(srv, rec)

    def get_set_score_pressure(self):
        srv, rec = self.get_set_score()
        return scoreMod.set_score_pressure(srv, rec)
    
    def get_total_set_score_pressure(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        return scoreMod.total_set_score_pressure(srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_total_set_win_probabilities(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        return scoreMod.total_set_win_probabilities(srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_total_match_score_pressure(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        srv_set, rec_set = self.get_set_score()
        return scoreMod.total_match_score_pressure(srv_set, rec_set, srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_total_match_win_probabilities(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        srv_set, rec_set = self.get_set_score()
        return scoreMod.total_match_win_probabilities(srv_set, rec_set, srv_game, rec_game, srv_pt_str, rec_pt_str)

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
        desc = self.get_hit_event_descriptor()
        stroke = desc.split('_', 1)[0]
        return stroke == 'fh'

    def is_back(self):
        return self.is_hit() and not self.is_fore()
    
    def is_volley(self, prevShot):
        ### prevShot: None or Shot
        return prevShot.get_num_bounces() == 0 
    
    def is_ground(self, prevShot):
        ### prevShot: None or Shot
        return not self.is_volley(prevShot)
    
    def is_bounce(self):
        return self.get_num_bounces() > 0
        
    def is_double_bounce(self):
        return self.get_num_bounces() > 1
    
    def is_winner(self, nextShot):
        ### nextShot: None or Shot
        if nextShot is None:
            return self.is_fair(nextShot)
        else:
            return self.is_fair(nextShot) & self.is_double_bounce()
        
    def is_slice_rpm(self, threshold_rpm = 500):
        return self.get_initial_spin() < -threshold_rpm ### slices have negative rpm
    
    def is_topspin_rpm(self, threshold_rpm = 500):
        return self.get_initial_spin() > threshold_rpm
    
    def is_slice_visual(self):
        desc = self.get_hit_event_descriptor()
        if desc in ['srv', 'ovh']:
            return False
        else:
            _, spin = desc.split('_')
            return spin == 'sf'    

    def is_topspin_visual(self):
        desc = self.get_hit_event_descriptor()
        if desc in ['srv', 'ovh']:
            return False
        else:
            _, spin = desc.split('_')
            return spin == 'ts'

    
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
    
    def is_in_net(self):
        return self.shotTraj.is_in_net() ### TODO: can not deal with net cord right now.

    def is_top_to_bot(self):
        return self.get_hit_event()[1] == 'T'
    
    def is_bot_to_top(self):
        return self.get_hit_event()[1] == 'B'
    
    def is_by_server(self):
        return self.get_shot_index() % 2 == 0
    
    def is_by_returner(self):
        return self.get_shot_index() % 2 == 1

    #### 
    def mean_deviation(self, dof_correction = 0):
        return self.shotTraj.mean_deviation(dof_correction = dof_correction)
    
    #### FEATURE GETTERS ######
    def feat_ball(self, how = 'trajectory', which_endpoint = 'smooth'):
        return self.shotTraj.feat_ball(how = how, which_endpoint = which_endpoint)
    
    def feat_hit_player(self):
        return self.hitPlayer.feat_player()
    
    def feat_rec_player(self):
        return self.recPlayer.feat_player()
    
    def feat_top_player(self):
        if self.hitPlayer.get_side() == 'top':
            return self.hitPlayer.feat_player()
        else:
            return self.recPlayer.feat_player()
        
    def feat_bot_player(self):
        if self.hitPlayer.get_side() == 'bot':
            return self.hitPlayer.feat_player()
        else:
            return self.recPlayer.feat_player()
        
    def feat_spline(self, how = 'trajectory', which_endpoint = 'smooth'):
        return self.shotTraj.feat_spline(how = how, which_endpoint = which_endpoint)
        
    def feat_spline_cumul(self, how = 'trajectory', which_endpoint = 'smooth'):
        return self.shotTraj.feat_spline_cumul(how = how, which_endpoint = which_endpoint)

    def get_capture(self, confs):
        return self.shotTraj.get_capture(confs)

    def stream(self, cap = None, confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        if cap is None:
            cap = self.get_capture(confs)

        return self.shotTraj.stream(cap)
    
    def stream_ball(self, P, how = 'trajectory', cap = None, confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        if cap is None:
            cap = self.get_capture(confs)

        return self.shotTraj.stream_ball(P, how = how, cap = cap, confs = confs)

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
        SHOW_THRESHOLD = 20
        if len(self) > SHOW_THRESHOLD:
            firstShots, lastshots = self[:SHOW_THRESHOLD // 2], self[-SHOW_THRESHOLD // 2:] 
            firstIdxs, lastIdxs = range(len(self))[:SHOW_THRESHOLD // 2], range(len(self))[-SHOW_THRESHOLD // 2:] 
            firstStrs, lastStrs = [f'  {i}: {p},' for i, p in zip(firstIdxs, firstShots)], [f'  {i}: {p},' for i, p in zip(lastIdxs, lastshots)]
            shotStrs = [*firstStrs, '...', *lastStrs]
        else:   
            shotStrs = [f'  {i}: {p},' for i, p in enumerate(self)]
            
        innerStr = '\n'.join(shotStrs)
        return f'Shots([\n{innerStr}\n])'

    def __iter__(self):
        return iter(self.shotList)
    
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.shotList[item]
        elif isinstance(item, slice):
            return Shots(self.shotList[item])
        else:
            raise KeyError('Slicing Shots object requires either integer or slice.')

    def __len__(self):
        return len(self.shotList)
    
    def filter(self, bool_func_or_array, is_below = None, is_above = None, negate = False, **filter_kwargs):
        def filter_func(myShot, index = None):
            if isinstance(bool_func_or_array, np.ndarray):
                val = bool_func_or_array[index]
            elif isinstance(bool_func_or_array, str):
                val = getattr(myShot, bool_func_or_array)(**filter_kwargs)
            else:
                val = bool_func_or_array(myShot, **filter_kwargs)

            if is_below is None and is_above is None:
                if negate:
                    myBool = not val
                else:
                    myBool = val
            elif is_below is None:
                myBool = val > is_above
            elif is_above is None:
                myBool = val < is_below
            else:
                myBool = is_above < val < is_below
            return myBool
        
        return Shots([myShot for idx, myShot in enumerate(self) if filter_func(myShot, index = idx)])
     
    def apply(self, appl_func, dtype = float, **appl_kwargs):
        if isinstance(appl_func, str):
            return np.array([getattr(myShot, appl_func)(**appl_kwargs) for myShot in self], dtype = dtype)
        else:
            return np.array([appl_func(myShot, **appl_kwargs) for myShot in self], dtype = dtype)

    def flip(self, courtLines = None, across_middle = True, across_net = True):
        if courtLines is None:
            courtLines = CourtLines.make()
        return self.__class__([myShot.flip(courtLines = courtLines, across_middle = across_middle, across_net = across_net) for myShot in self])
    
    def join(self, otherShots):
        return Shots([*self, *otherShots])

    def sort(self, sort_by = None, descending = False):
        if sort_by is None:
            keyFunc = lambda el: (el.get_match_index(), el.get_point_index())
            return Shots(sorted(self.shotList, key = keyFunc, reverse = descending))
        elif isinstance(sort_by, str):
            keyFunc = lambda el: getattr(el, sort_by)()
            return Shots(sorted(self.shotList, key = keyFunc, reverse = descending))
        elif isinstance(sort_by, np.ndarray) or isinstance(sort_by, list):
            sort_idxs = np.argsort(sort_by)
            if descending:
                sort_idxs = np.flip(sort_idxs)
            sorted_shots = [self[int(idx)] for idx in sort_idxs]
            return Shots(sorted_shots)

    def enforce_direction(self, from_where):
        ### TODO: ADD SORT!
        assert from_where in ['Top', 'Bot']
        t2bShots = self.filter('is_top_to_bot')
        b2tShots = self.filter('is_bot_to_top')

        if from_where == 'Top':
            jointShots = t2bShots.join(b2tShots.flip())
        elif from_where == 'Bot':
            jointShots = b2tShots.join(t2bShots.flip())
        
        return jointShots.sort()

    def stream(self, cap = None, confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        if cap is None:
            return StreamList([myShot.stream(myShot.get_capture(confs)) for myShot in self]).join()
        else:
            return StreamList([myShot.stream(cap) for myShot in self]).join()
        
    
    def stream_ball(self, P, cap = None, how = 'trajectory', confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        return StreamList([myShot.stream_ball(P, how = how, cap = cap, confs = confs) for myShot in self]).join()

    ### STAT GETTERS ###
    # -> .get methods of Shot, done with make_method below
        
    ### FEATURE GETTERS ######
    # -> .feat methods of Shot, done with make_method below!!!
        
@dataclass(frozen = True)
class Point(Shots):
    ### every point is a collection of shots in sequence!
    
    @staticmethod
    def from_shot_frame_and_player_SFs(trajFrm, topSF, botSF):
        if trajFrm.dropna(how = 'all', axis = 0).empty:
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
    
    @staticmethod
    def from_point_table(pointFrm):
        shotFrms = []
        currFrm = None
        for _, nextFrm in pointFrm.groupby('shot_idx'):
            if currFrm is not None:
                shotFrm = pd.concat([currFrm, nextFrm.iloc[[0]]]) ### to ensure that the beginning of the next hit is included!
                shotFrms.append(shotFrm)
            currFrm = nextFrm
            
        shotFrms.append(currFrm) ### last shot does not have a next hit
        shotList = [Shot.from_shot_table(shotFrm) for shotFrm in shotFrms]
        return Point(shotList)

    def __repr__(self):
        if len(self.shotList) == 0:
            return 'Point()'
        else:
            serveShot = self.shotList[0]
            server, receiver = serveShot.hitPlayer.get_name(), serveShot.recPlayer.get_name()
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
    
    def get_num_frames(self):
        return len(self.get_frames())
    
    def get_winner_from_trajectory(self):
        lastShot = self.shotList[-1]
        
        if lastShot.is_fair(None): ### None -> there is no nextShot
            return lastShot.hitPlayer.get_name()
        else:
            return lastShot.recPlayer.get_name()
            
    def get_winner_from_score(self, nextPnt, verbose = False):    
        score_func_regular = lambda scoreStr: {'0': 0, '15': 1, '30': 2, '40': 3, 'A': 4}[scoreStr]
        score_func_tiebreak = lambda scoreStr: int(scoreStr)

        srvShot = self.get_serve()
        if srvShot is None:
            if verbose: print(f'No serve detected: {self}')
            return None
        else:
            srv_name, rec_name = srvShot.hitPlayer.get_name(), srvShot.recPlayer.get_name()

        ### Deal with Current Score
        curr_srv_str, curr_rec_str = self.get_point_score()
        
        try:
            curr_srv_val, curr_rec_val = score_func_regular(curr_srv_str), score_func_regular(curr_rec_str)
        except:
            curr_srv_val, curr_rec_val = score_func_tiebreak(curr_srv_str), score_func_tiebreak(curr_rec_str)

        ### Deal with Next Score
        if nextPnt is None:
            if curr_srv_val > curr_rec_val:
                if verbose: print(f'Winner on last point: Server -> {curr_srv_str}-{curr_rec_str}')
                return srv_name
            elif curr_srv_val < curr_rec_val:
                if verbose: print(f'Winner on last point: Returner -> {curr_srv_str}-{curr_rec_str}')
                return rec_name 
            else:
                if verbose: print(f'Non-Decisive Score on Last Point: {curr_srv_str}-{curr_rec_str}')
                return None
        elif isinstance(nextPnt, NullPoint):
            try:
                lo_name, lo_score = nextPnt['lo_name'], nextPnt['lo_score']
                hi_name, hi_score = nextPnt['hi_name'], nextPnt['hi_score']
            except:
                return None
            else:
                if lo_name == srv_name:
                    next_srv_str, next_rec_str = str(lo_score), str(hi_score)
                elif hi_name == srv_name:
                    next_srv_str, next_rec_str = str(hi_score), str(lo_score) 
                else:
                    raise Exception(f'Neither {lo_name} nor {hi_name} are serving; server should be {srv_name}.')

        else:
            next_srv_str, next_rec_str = nextPnt.get_point_score()
            
            ### check if the server and the receiver switch places.
            if self.get_srv_player_name() != nextPnt.get_srv_player_name(): ### happens during tiebreak and change of sides,
                                                                            ### but the latter case could be handled without this if condition
                                                                            ### by the section below: next_srv_val == next_rec_val == 0
                next_srv_str, next_rec_str = next_rec_str, next_srv_str

        try:
            next_srv_val, next_rec_val = score_func_regular(next_srv_str), score_func_regular(next_rec_str)
        except:
            next_srv_val, next_rec_val = score_func_tiebreak(next_srv_str), score_func_tiebreak(next_rec_str)
            
        ### in this case, the current point is the last in the game, and the winner is whoever has the higher score.
        if next_srv_val == next_rec_val == 0: 
            if curr_srv_val > curr_rec_val:
                if verbose: print(f'Winner on game point: Server -> {curr_srv_str}-{curr_rec_str}')
                return srv_name
            elif curr_srv_val < curr_rec_val:
                if verbose: print(f'Winner on game point: Returner -> {curr_srv_str}-{curr_rec_str}')
                return rec_name
        
        ### Score does not contain advantage; server wins point
        elif next_srv_val > curr_srv_val and next_rec_val == curr_rec_val: 
            if verbose: print(f'Winner: Server -> {curr_srv_str}-{curr_rec_str} to {next_srv_str}-{next_rec_str}')
            return srv_name
        
        ### Score does not contain advantage; receiver wins point
        elif next_srv_val == curr_srv_val and next_rec_val > curr_rec_val: 
            if verbose: print(f'Winner: Returner -> {curr_srv_str}-{curr_rec_str} to {next_srv_str}-{next_rec_str}')
            return rec_name 
        
        ### Score is 40-A --> 40-40
        elif next_srv_val == curr_srv_val and next_rec_val < curr_rec_val:
            if verbose: print(f'Winner: Server -> {curr_srv_str}-{curr_rec_str} to {next_srv_str}-{next_rec_str}')
            return srv_name
        
        ### Score is A-40 --> 40-40
        elif next_srv_val < curr_srv_val and next_rec_val == curr_rec_val:
            if verbose: print(f'Winner: Returner -> {curr_srv_str}-{curr_rec_str} to {next_srv_str}-{next_rec_str}')
            return rec_name

        ### Score does not change -> first serve fault most likely!
        elif next_srv_val == curr_srv_val and next_rec_val == curr_rec_val: 
            if verbose: print(f'Score Unchanged: {curr_srv_str}-{curr_rec_str} --> {next_srv_str}-{next_rec_str}')
            return None

        else:
            if verbose: print(f'Unhandled: {curr_srv_str}-{curr_rec_str} --> {next_srv_str}-{next_rec_str}')
            return None

    def get_point_score_pressure(self):
        srv_str, rec_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        return scoreMod.point_score_pressure(srv_str, rec_str, srv_game, rec_game)

    def get_game_score_pressure(self):
        srv, rec = self.get_game_score()
        return scoreMod.game_score_pressure(srv, rec)

    def get_set_score_pressure(self):
        srv, rec = self.get_set_score()
        return scoreMod.set_score_pressure(srv, rec)
    
    def get_total_set_score_pressure(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        return scoreMod.total_set_score_pressure(srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_total_set_win_probabilities(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        return scoreMod.total_set_win_probabilities(srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_total_match_score_pressure(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        srv_set, rec_set = self.get_set_score()
        return scoreMod.total_match_score_pressure(srv_set, rec_set, srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_total_match_win_probabilities(self):
        srv_pt_str, rec_pt_str = self.get_point_score()
        srv_game, rec_game = self.get_game_score()
        srv_set, rec_set = self.get_set_score()
        return scoreMod.total_match_win_probabilities(srv_set, rec_set, srv_game, rec_game, srv_pt_str, rec_pt_str)

    def get_match_index(self):
        return self[0].get_match_index()
    
    def get_point_index(self):
        return self[0].get_point_index()

    def get_serve(self):
        if len(self) == 0:
            return None
        else:
            fsShot = self[0]
            if fsShot.is_serve():
                return fsShot
            else:
                return None
            
    def get_server(self):
        return self.get_serve().hitPlayer
    
    def get_returner(self):
        return self.get_serve().recPlayer

    def get_return(self):
        if len(self) <= 1:
            return None
        elif not self.is_serve():
            return None
        else:
            return self[1]

    def get_srv_player_name(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.hitPlayer.get_name()
    
    def get_rec_player_name(self):
        srvShot = self.get_serve()
        if srvShot is not None:
            return srvShot.recPlayer.get_name()
    
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
        
    def is_game_point(self):
        srv_pnt, rec_pnt = self.get_point_score()
        srv_gms, rec_gms = self.get_game_score()

        is_dec = scoreMod.is_deciding_point_on_serve(srv_pnt, rec_pnt, srv_gms, rec_gms)
        is_tb =  scoreMod.is_tiebreak(srv_gms, rec_gms)
        return is_dec and not is_tb

    def is_break_point(self):
        srv_pnt, rec_pnt = self.get_point_score()
        srv_gms, rec_gms = self.get_game_score()

        is_dec = scoreMod.is_deciding_point_against_serve(srv_pnt, rec_pnt, srv_gms, rec_gms)
        is_tb =  scoreMod.is_tiebreak(srv_gms, rec_gms)
        return is_dec and not is_tb
    
    def is_set_point(self):
        srv_pnt, rec_pnt = self.get_point_score()
        srv_gms, rec_gms = self.get_game_score()

        is_dec_srv = scoreMod.is_deciding_point_on_serve(srv_pnt, rec_pnt, srv_gms, rec_gms)
        is_dec_rec = scoreMod.is_deciding_point_against_serve(srv_pnt, rec_pnt, srv_gms, rec_gms)
        is_hold_to_win = scoreMod.is_hold_to_win_set(srv_gms, rec_gms)
        is_break_to_win = scoreMod.is_break_to_win_set(srv_gms, rec_gms)

        hold_point_to_win_set = is_dec_srv and is_hold_to_win
        break_point_to_win_set = is_dec_rec and is_break_to_win
        return hold_point_to_win_set or break_point_to_win_set

    def is_match_point(self):
        srv_pnt, rec_pnt = self.get_point_score()
        srv_gms, rec_gms = self.get_game_score()
        srv_set, rec_set = self.get_set_score()

        is_dec_srv = scoreMod.is_deciding_point_on_serve(srv_pnt, rec_pnt, srv_gms, rec_gms)
        is_dec_rec = scoreMod.is_deciding_point_against_serve(srv_pnt, rec_pnt, srv_gms, rec_gms)
        is_hold_to_win = scoreMod.is_hold_to_win_set(srv_gms, rec_gms)
        is_break_to_win = scoreMod.is_break_to_win_set(srv_gms, rec_gms)

        srv_set_to_win = scoreMod.is_server_deciding_set(srv_set, rec_set)
        rec_set_to_win = scoreMod.is_receiver_deciding_set(srv_set, rec_set)

        hold_point_to_win_match = is_dec_srv and is_hold_to_win and srv_set_to_win
        break_point_to_win_match = is_dec_rec and is_break_to_win and rec_set_to_win

        return hold_point_to_win_match or break_point_to_win_match

    def get_point_score(self):
        return self[0].get_point_score()
        
    def get_game_score(self):
        serveShot = self.get_serve()
        if serveShot is None:
            return None, None
        else:
            return serveShot.hitPlayer.get_game_score(), serveShot.recPlayer.get_game_score()
    
    def get_set_score(self):
        serveShot = self.get_serve()
        if serveShot is None:
            return None, None
        else:
            return serveShot.hitPlayer.get_set_score(), serveShot.recPlayer.get_set_score()
    
    def get_capture(self, confs):
        return self[0].get_capture(confs)

    def stream(self, cap = None, confs = None):
        if cap is None and confs is None:
            raise Exception('Need either a video capture or a list of configurations passed in as kwargs')

        if cap is None:
            cap = self.get_capture(confs)
            
        fs, ls = self.get_frames()[0], self.get_frames()[-1]
        return Stream(cap, fs, ls)
    
@dataclass(frozen = True)
class NullPoint:
    ### A Point for which we have no trajectory data, but for which we know at least that it exists (and maybe other stuff)
    infoDict: dict ### what we know about the NullPoint (score, player names, etc)

    @staticmethod
    def from_placeholder_series(placeSer):
        return NullPoint(placeSer.to_dict())

    def get_match_index(self):
        try:
            return int(self.infoDict['match_idx'])
        except KeyError:
            return None
    
    def get_point_index(self):
        try:
            return int(self.infoDict['point_idx'])
        except KeyError:
            return None

    def get_winner_from_score(self, nextPnt, verbose = False):    ### implement as needed.
        pass

@dataclass(frozen = True)
class Points:
    pointList: List[Point]

    @classmethod
    def from_SFs(cls, trajSFs, topSFs, botSFs):
        ### these have to be segmented!
        pointList = []
        
        for idx in range(trajSFs.length()):
            trajSF, topSF, botSF = trajSFs[idx], topSFs[idx], botSFs[idx]
            trajFrm = trajSF.to_dataframe()
            
            myPnt = Point.from_shot_frame_and_player_SFs(trajFrm, topSF, botSF)
            pointList.append(myPnt)
            
        return cls(pointList)
    
    @classmethod
    def from_datatable(cls, tableFrm, split_by_match = False):
        if split_by_match:
            matchList = []
            for _, matchFrm in tableFrm.groupby('match_idx'):
                pointList = [Point.from_point_table(pointFrm) for _, pointFrm in matchFrm.groupby('point_idx')]
                matchList.append(cls(pointList))
            return matchList
        else:   
            pointList = [Point.from_point_table(pointFrm) for _, pointFrm in tableFrm.groupby(['match_idx', 'point_idx'])]
            return cls(pointList)

    @classmethod
    def from_placeholder_table(cls, placeFrm):
        pointList = [NullPoint.from_placeholder_series(pointSer) for _, pointSer in placeFrm.iterrows()]
        return cls(pointList)

    def __repr__(self):
        SHOW_THRESHOLD = 20
        if len(self) > SHOW_THRESHOLD:
            firstPnts, lastPnts = self[:SHOW_THRESHOLD // 2], self[-SHOW_THRESHOLD // 2:] 
            firstIdxs, lastIdxs = range(len(self))[:SHOW_THRESHOLD // 2], range(len(self))[-SHOW_THRESHOLD // 2:] 
            firstStrs, lastStrs = [f'  {i}: {p},' for i, p in zip(firstIdxs, firstPnts)], [f'  {i}: {p},' for i, p in zip(lastIdxs, lastPnts)]
            pointStrs = [*firstStrs, '...', *lastStrs]
        else:   
            pointStrs = [f'  {i}: {p},' for i, p in enumerate(self)]
            
        innerStr = '\n'.join(pointStrs)
        return f'Points([\n{innerStr}\n])'

    def __iter__(self):
        return iter(self.pointList)
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            return Points(self.pointList[item])
        elif isinstance(item, int):
            return self.pointList[item]
        else:
            raise KeyError('Points __getitem__ has to be supplied an integer or a slice.')
    
    def __len__(self):
        return len(self.pointList)
    
    def get(self, match_index, point_index):
        getPnts = [myPnt for myPnt in self if myPnt.get_match_index() == match_index and myPnt.get_point_index() == point_index]
        if len(getPnts) == 0:
            print(f'Could not find Point {point_index} in Match {match_index}')
            return None
        elif len(getPnts) == 1:
            return getPnts[0]
        else:
            raise Exception(f'More than one Point {point_index} in Match {match_index} found! --> {getPnts}')
        
    def filter(self, bool_func_or_array, **filter_kwargs):
        ### TODO: harmonize Shots.filter and Points.filter
        if isinstance(bool_func_or_array, np.ndarray):
            return Points([myPnt for idx, myPnt in enumerate(self) if bool_func_or_array[idx]])
        elif isinstance(bool_func_or_array, str):
            return Points([myPnt for myPnt in self if getattr(myPnt, bool_func_or_array)(**filter_kwargs)])
        else:
            return Points([myPnt for myPnt in self if bool_func_or_array(myPnt, **filter_kwargs)])
        
    def apply(self, appl_func, dtype = float, **appl_kwargs):
        if isinstance(appl_func, str):
            return np.array([getattr(myPnt, appl_func)(**appl_kwargs) for myPnt in self], dtype = dtype)
        else:
            return np.array([appl_func(myPnt, **appl_kwargs) for myPnt in self], dtype = dtype)

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
        ### sort by match_index, then point_index
        keyFunc = lambda el: (el.get_match_index(), el.get_point_index())
        return Points(sorted(self.pointList, key = keyFunc))
    
    def enforce_direction(self, from_where):
        assert from_where in ['Top', 'Bot']
        t2bPnts = self.filter('is_top_to_bot')
        b2tPnts = self.filter('is_bot_to_top')

        if from_where == 'Top':
            jointPnts = t2bPnts.join(b2tPnts.flip())
        elif from_where == 'Bot':
            jointPnts = b2tPnts.join(t2bPnts.flip())
        
        return jointPnts.sort()

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
            return np.array(outList, dtype = float)
        except: ### sometimes, outList is a list of lists of unequal size
            return outList

    method.__name__ = shotMethodStr
    setattr(myClass, shotMethodStr, method)

def __make_method_is(myClass, shotMethodStr):
    def method(self, **kwargs):
        outList = [getattr(myShot, shotMethodStr)(**kwargs) for myShot in self]
        try:
            return np.array(outList, dtype = bool)
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
    
    if methodStr.startswith('is_'):
        ### normal getters
        if not methodStr in dir(Shots): ### do not overwrite
            __make_method_is(Shots, methodStr)

def read_in_data_tables(confs):
    ### READ IN DATA
    tables = []
    for conf_idx, conf in enumerate(confs):
        vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    
        dataTable = ff.read_feature_frame('dataTable', vid, sto, keep_only = 'last')
        dataTable.index = pd.MultiIndex.from_product([[conf_idx], dataTable.index])
        dataTable.index.names = ['match_idx', 'frame_num']
        
        tables.append(dataTable)

    allTable = pd.concat(tables)    
    return allTable

def read_in_all_placeholder_tables(confs):
    tables = []
    for conf_idx, conf in enumerate(confs):
        vid, sto = conf['video_id'], conf['store_name'] ### since they come up so often...
    
        placeTable = ff.read_feature_frame('placeTable', vid, sto, keep_only = 'last')
        placeTable['match_idx'] = conf_idx
        tables.append(placeTable)
    
    allTable = pd.concat(tables)    
    return allTable


if __name__ == '__main__':
    from ATMANpy.funcs.util_funcs import get_folder_struct
    import ATMANpy.scripts.pipeline_brest_tightseg as ppl
    import ATMANpy.funcs.feature_funcs as ff
    import ATMANpy.funcs.util_funcs as ut
    
    cnf_mod = ppl.config_files.default
       
    avail_vids = [f for f in get_folder_struct('video_files') if 'brest' in f]

    # confs = [cnf_mod.get_conf(avail_vids[0]), cnf_mod.get_conf(avail_vids[1])] 
    confs = [cnf_mod.get_conf(vid) for vid in avail_vids]

    dataTable = read_in_data_tables(confs)
    placeTable = read_in_all_placeholder_tables(confs)

    dataPoints = Points.from_datatable(dataTable)
    placePoints = Points.from_placeholder_table(placeTable)
    allPoints = dataPoints.join(placePoints).sort()

    print(placePoints)
    print('______')
    print(dataPoints)
    print('______')
    print(allPoints)

    # conf_idx = 10
    # conf = confs[conf_idx]

    # vid, sto = conf['video_id'], conf['store_name']
    # placeTable = ff.read_feature_frame('placeTable', vid, sto)
    
    # print(Points.from_placeholder_table(placeTable).sort())


    # for pnt_idx, placeSer in placeTable.iterrows():
    #     print(NullPoint.from_placeholder_series(placeSer))
    
    
    

    # tableFrm = read_in_all_tables(confs)
    
    # myPnts = Points.from_datatable(tableFrm)
    # print(Points.from_datatable(tableFrm))

