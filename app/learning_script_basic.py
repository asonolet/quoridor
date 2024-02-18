# import tensorflow as tf

# from tensorflow.keras.layers import Dense
# from tensorflow.keras import Model
import multiprocessing as mp

import numpy as np
from Quoridor2 import Game, play_with_proba

# class AIPlayer(Model):
#     def __init__(self):
#         super(AIPlayer, self).__init__()
#         self.f = lambda x: tf.nn.dropout()
#         self.d1 = Dense(50, activation='relu')
#         self.d2 = Dense(25, activation='relu')
#         self.d3 = Dense(10, activation='relu')
#         self.d4 = Dense(10, activation='relu')
#         self.d5 = Dense(25, activation='relu')
#         self.d6 = Dense(50, activation='relu')
#
#     def __call__(self, state):
#         x = self.d1(state)
#         x = self.tan(x)
#         x = self.d2(x)
#         x = self.tan(x)
#         x = self.d3(x)
#         x = self.tan(x)
#         x = self.d4(x)
#         x = self.tan(x)
#         x = self.d5(x)
#         x = self.tan(x)
#         return self.d6(x)
#
#     def call_possibilities(self, state, possibilities):
#         return self.call(state) * possibilities
#
#
# aiplayer = AIPlayer()
#
# loss_object = 0 # il faut créer la bonne fonction cout
# optimizer = tf.keras.optimizers.Adam()
#
#
# def train_aiplayer(states, rewards):
#     with tf.GradientTape() as tape:
#         predictions = aiplayer(states)
#

# def test_aiplayer(state, reward):


# N_PARTIES = 1000


# def play_to_explore(player1, player2, n=N_PARTIES, chkpt=100):
#     states, rewards = [], []
#     i = 0
#     while i <= n:
#         game = Game('')
#         while game.board_state.winner is None:
#             if i % 2 == 0:
#                 player = player1
#             else:
#                 player = player2
#             state0 = game.board_state.to_universal_state(i % 2)
#             state1 = game.board_state.to_universal_state((i+1) % 2)
#             scores0 = game.evaluate_all_choices(i % 2)
#             scores1 = game.evaluate_all_choices((i+1) % 2)
#             game.coup(player(game, i % 2), i % 2, score=False)
#             states.append(state0)
#             states.append(state1)
#             rewards.append(scores0)
#             rewards.append(scores1)
#             if i % chkpt == 0:
#                 print('%.2d %%' % (i/N_PARTIES*100))
#                 np.save('play_with_proba/states_%.3d.npy' % (i//chkpt), states)
#                 np.save('play_with_proba/rewards_%.3d.npy' % (i//chkpt),
#                         rewards)
#                 states = []
#                 rewards = []
#             i = i+1
#
#     return states[:10], rewards[:10]
#

CHKPT = 100


def play_to_explore_without_score_mp(player1, player2, n, i_start) -> None:
    states = []
    i = i_start
    while i <= n + i_start:
        game = Game("")
        while (game.board_state.winner is None) and (i <= n + i_start):
            player = player1 if i % 2 == 0 else player2
            state0 = game.board_state.to_universal_state(i % 2)
            game.coup(player(game, i % 2), i % 2, score_=False)
            states.append(state0)
            if i % CHKPT == 0:
                print(
                    mp.current_process().name + " avencement : %.2d %% "
                    "writting file %.3d" % (((i - i_start) / n * 100), i // CHKPT),
                )
                np.save(
                    "play_with_proba_without_score/states_%.3d.npy" % (i // CHKPT),
                    states,
                )
                states = []
            i = i + 1

    # return states[:10], rewards[:10]


if __name__ == "__main__":
    import time

    start = time.time()
    procs = 3
    N_COUPS = 1000
    jobs = []
    for ip in range(procs):
        p = mp.Process(
            target=play_to_explore_without_score_mp,
            args=(play_with_proba, play_with_proba, N_COUPS, ip * N_COUPS),
        )
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    stop = time.time()
    print("time of exectution : %.2f" % (stop - start))
    print("time of execution per game : %.2f" % ((stop - start) / (procs * N_COUPS)))
