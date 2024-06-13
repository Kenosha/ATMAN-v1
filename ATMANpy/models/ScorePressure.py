
import os
import numpy as np
import pandas as pd
import ATMANpy.funcs.util_funcs as ut

def probability_server_wins_game(srv, rec, p_server = 0.5):
    if (srv > rec + 1) and (srv >= 4): ### server wins
        return 1
    elif (rec > srv + 1) and (rec >= 4): # receiver wins
        return 0
    elif srv == rec >= 2: ## deuce like situation
        return (p_server ** 2) / (1 - 2 * p_server + 2 * p_server ** 2)
    elif srv > rec >= 2: ### ad-in like situation 
        return p_server + (1 - p_server) * probability_server_wins_game(srv, rec + 1, p_server = p_server)
    elif rec > srv >= 2: ### ad-out like situation
        return p_server * probability_server_wins_game(srv + 1, rec, p_server = p_server)
    else:
        return p_server * probability_server_wins_game(srv + 1, rec, p_server = p_server) + (1 - p_server) * probability_server_wins_game(srv, rec + 1, p_server = p_server)

def probability_server_wins_tiebreak(srv, rec, points_to_win = 7, p_server = 0.5):
    if (srv > rec + 1) and (srv >= points_to_win): ### server wins
        return 1
    elif (rec > srv + 1) and (rec >= points_to_win): # receiver wins
        return 0
    elif srv == rec >= points_to_win - 1:
        return 0.5
    else:
        next_serve_switches = (srv + rec) % 2 == 0
        if next_serve_switches:
            return p_server * (1 - probability_server_wins_tiebreak(rec, srv + 1, points_to_win = points_to_win, p_server = p_server)) + (1 - p_server) * (1 - probability_server_wins_tiebreak(rec + 1, srv, points_to_win = points_to_win, p_server = p_server))
        else:
            return p_server * probability_server_wins_tiebreak(srv + 1, rec, points_to_win = points_to_win, p_server = p_server) + (1 - p_server) * probability_server_wins_tiebreak(srv, rec + 1, points_to_win = points_to_win, p_server = p_server)

def probability_server_wins_set(srv, rec, p_hold = 0.5):
    if (srv == 7) or (srv == 6 and rec <= 4):
        return 1
    elif (rec == 7) or (rec == 6 and srv <= 4):
        return 0
    elif srv == rec == 6:
        return 0.5
    else:
        return p_hold * (1 - probability_server_wins_set(rec, srv + 1, p_hold = p_hold)) + (1 - p_hold) * (1 - probability_server_wins_set(rec + 1, srv, p_hold = p_hold))

def probability_server_wins_match(srv, rec, sets_to_win = 2):
    if srv == sets_to_win:
        return 1
    elif rec == sets_to_win:
        return 0
    else:
        return 0.5 * probability_server_wins_match(srv + 1, rec, sets_to_win = sets_to_win) + 0.5 * probability_server_wins_match(srv, rec + 1, sets_to_win = sets_to_win)

class PressureModel:    
    def __init__(self, p_server = 0.65, sets_to_win = 2):
        # p_server -> probability that the server wins the point
        self.p_server = p_server
        self.p_hold = probability_server_wins_game(0, 0, p_server = p_server)
        self.sets_to_win = sets_to_win

    def point_score_pressure(self, srv_str, rec_str):
        score_func_regular = lambda scoreStr: {'0': 0, '15': 1, '30': 2, '40': 3, 'A': 4}[scoreStr]
        score_func_tiebreak = lambda scoreStr: int(scoreStr)

        try:
            srv_val, rec_val = score_func_regular(srv_str), score_func_regular(rec_str)
            regular_game = True
        except:
            srv_val, rec_val = score_func_tiebreak(srv_str), score_func_tiebreak(rec_str)
            regular_game = False

        ### the score pressure depends on the difference between 
        # -> the probability to win the service game if the point is won,
        # -> the probability to win the service game if the point is lost
        if regular_game: 
            p_srv_if_win = probability_server_wins_game(srv_val + 1, rec_val, p_server = self.p_server)
            p_srv_if_lose = probability_server_wins_game(srv_val, rec_val + 1, p_server = self.p_server)
        else: 
            p_srv_if_win = probability_server_wins_tiebreak(srv_val + 1, rec_val, p_server = self.p_server, points_to_win = 7)
            p_srv_if_lose = probability_server_wins_tiebreak(srv_val, rec_val + 1, p_server = self.p_server, points_to_win = 7)

        ### note that the score pressure is symmetric for the server and receiver if calculated as a difference, 
        ### since the probability that the receiver wins the game or tiebreak is 1 - p_if_win and 1 - p_if_lose depending on what happens
        ### -> has been checked!
        diff_p = p_srv_if_win - p_srv_if_lose
        
        return round(diff_p, 3)

    def game_score_pressure(self, srv, rec):
        ### the game score pressure depends on the difference between 
        # -> the probability for the server to win the set if the game is won,
        p_set_if_win = 1 - probability_server_wins_set(rec, srv + 1, p_hold = self.p_hold)
        # -> the probability for the server to win the set if the game is lost
        p_set_if_lose = 1 - probability_server_wins_set(rec + 1, srv, p_hold = self.p_hold)

        ### By the same argument as for point score pressure, the difference is symmetric between server and receiver
        diff_p = p_set_if_win - p_set_if_lose
        
        return round(diff_p, 3)

    def set_score_pressure(self, srv, rec):

        ### the set score pressure depends on the difference between 
        # -> the probability for the server to win the match if the set is won,
        p_match_if_win = probability_server_wins_match(srv + 1, rec, sets_to_win = self.sets_to_win)
        # -> the probability for the server to win the set if the game is lost
        p_match_if_lose = probability_server_wins_match(srv, rec + 1, sets_to_win = self.sets_to_win)

        ### By the same argument as for point score pressure, the difference is symmetric between server and receiver
        diff_p = p_match_if_win - p_match_if_lose
        
        return round(diff_p, 3)

if __name__ == '__main__':
        
    graph_dir = ut.get_folder_struct('graphs')['score_pressure']
    import seaborn as sn
    import matplotlib.pyplot as plt
    from itertools import product

    def save_heatmap(array, save_name, save_where = graph_dir):
        scoreFrm = pd.DataFrame(array, index = scores, columns = scores)
        fig, ax = plt.subplots(figsize = (10, 7))
        ax = sn.heatmap(scoreFrm, annot=True, cmap = 'Wistia')
        ax.set_xlabel('Receiver')
        ax.set_ylabel('Server')
        fig.savefig(os.path.join(save_where, f'{save_name}.png'))

    pressMod_bo3 = PressureModel(p_server = 0.65, sets_to_win = 2)
    pressMod_bo5 = PressureModel(p_server = 0.65, sets_to_win = 3)

    #### A normal game
    scores = ['0', '15', '30', '40', 'A']
    p_server = 0.66
    array = np.full((len(scores), len(scores)), fill_value = np.nan)
    for row_idx, a in enumerate(scores):
        for col_idx, b in enumerate(scores):
            if (a == 'A' and b != '40') or (b == 'A' and a != '40'):
                continue 
            array[row_idx, col_idx] = pressMod_bo3.point_score_pressure(a, b)
    save_heatmap(array, 'game')

    ### A tiebreak
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p_server = 0.5 ### technically, the server switches...
    array = np.full((len(scores), len(scores)), fill_value = np.nan)
    for row_idx, a in enumerate(scores):
        for col_idx, b in enumerate(scores):
            if (a + 1 < b and b >= 7) or (b + 1 < a and a >= 7):
                continue 
            array[row_idx, col_idx] = pressMod_bo3.point_score_pressure(a, b)
    save_heatmap(array, 'tiebreak')

    ### A set
    scores = [0, 1, 2, 3, 4, 5, 6]
    array = np.full((len(scores), len(scores)), fill_value = np.nan)
    for row_idx, a in enumerate(scores):
        for col_idx, b in enumerate(scores):
            if (a == 6 and b <= 4) or (b == 6 and a <= 4):
                continue 
            array[row_idx, col_idx] = pressMod_bo3.game_score_pressure(a, b)
    save_heatmap(array, 'set')

    ### A three-set match                                               
    scores = [0, 1]
    array = np.full((len(scores), len(scores)), fill_value = np.nan)
    for row_idx, a in enumerate(scores):
        for col_idx, b in enumerate(scores):
            array[row_idx, col_idx] = pressMod_bo3.set_score_pressure(a, b)
    save_heatmap(array, 'match_bestof3')

    ### A five-set match
    scores = [0, 1, 2]
    array = np.full((len(scores), len(scores)), fill_value = np.nan)
    for row_idx, a in enumerate(scores):
        for col_idx, b in enumerate(scores):
            array[row_idx, col_idx] = pressMod_bo5.set_score_pressure(a, b)
    save_heatmap(array, 'match_bestof5')

