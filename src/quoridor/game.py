import time

import numpy as np
from scipy import sparse as sp

from quoridor.board_state import BoardState, Player
from quoridor.scorer import score_with_relative_path_length_dif


class Game:
    """
    This object contains anything a game needs to be played
    """

    def __init__(self, game_name):
        self.game_name = game_name
        self.board_state = BoardState()

        self.coup_joues = []
        self.coup_joues.append((4, 0, -1))
        self.coup_joues.append((4, 8, -1))

        self.all_walls_choices = np.transpose(
            np.nonzero(self.board_state.wall_possibilities > 0)
        )
        self.set = (
            100 * self.all_walls_choices[:, 0]
            + 10 * self.all_walls_choices[:, 1]
            + self.all_walls_choices[:, 2]
        )

    def coup(self, choice=None, player_number=1, get_back=False, score_=True):
        """
        update board_state
        if score return relative diff length between paths
        if get back is True, don't change the board state
        :param choice:
        :param player_number: int
        :param get_back: bool
        :param score_: bool
        :return:
        """
        self.coup_joues.append(choice)
        if choice[2] == -1:
            self.board_state.update_player_positions(choice[:2], player_number)
        else:
            self.board_state.add_new_wall(choice, player_number)
        if score_:
            dist = score_with_relative_path_length_dif(board_state, player_number)
            if get_back:
                self.get_back(1)
            return dist

        if get_back:
            self.get_back(1)

    def get_back(self, n):
        for i_ in range(n):
            player_number = (len(self.coup_joues) - 1) % 2

            choice = self.coup_joues.pop()

            if (len(choice) == 2) or (choice[2] == -1):
                pos = self.last_pos
                self.board_state.update_player_positions(pos, player_number)
                self.board_state.winner = -1
            elif len(choice) == 3:
                self.board_state.remove_wall(choice, player_number)

    @property
    def last_pos(self):
        for i_ in range(len(self.coup_joues) - 2, -1, -2):
            if len(self.coup_joues[i_]) == 2 or self.coup_joues[i_][2] == -1:
                return self.coup_joues[i_]

    def _all_moves(self, player_number: int):
        all_moves = []
        for _, k in self.board_state.free_paths[
            self.board_state.player[player_number].k_pos, :
        ].keys():
            new_coup = (k // 10, k % 10, -1)
            new_position = new_coup[:2]
            # In this case, both players are next one another
            if new_position == self.board_state.player[1 - player_number].position:
                old_pos = np.array(self.board_state.player[player_number].position)
                new_position = np.array(new_position)
                new_coup = tuple(np.r_[2 * new_position - old_pos, -1])
                if (0 < new_coup[0] < 9) & (0 < new_coup[1] < 9):
                    if self.board_state.free_paths[
                        10 * new_coup[0] + new_coup[1],
                        new_position[0] * 10 + new_position[1],
                    ]:
                        all_moves.append(new_coup)
            else:
                all_moves.append(new_coup)
        return all_moves

    def all_coups(self, player_number: int):
        all_moves = self._all_moves(player_number)
        if self.board_state.player[player_number].n_tuiles > 0:
            all_walls = np.transpose(
                np.nonzero(self.board_state.wall_possibilities > 0)
            )
            all_coups = np.concatenate((all_moves, all_walls))
        else:
            all_coups = np.array(all_moves)
        return all_coups

    def moves_allowed(self, player_number: int):
        all_moves = self._all_moves(player_number)
        pos = self.board_state.player[player_number].position
        is_move_allowed = np.zeros((4,))
        for move in all_moves:
            if (move[0] == pos[0]) & (move[1] > pos[1]):
                is_move_allowed[0] = 1
            if (move[0] == pos[0]) & (move[1] < pos[1]):
                is_move_allowed[1] = 1
            if (move[1] == pos[1]) & (move[0] < pos[0]):
                is_move_allowed[2] = 1
            if (move[1] == pos[1]) & (move[0] > pos[0]):
                is_move_allowed[3] = 1
        return is_move_allowed

    def evaluate_all_choices(self, player_number: int):
        """
        for a given board_state, test all possibilities and returns a vector
        where the place of the score always match the same choice. If the
        choice is not available, score is 0
        :param player_number:
        :return: score vector of length 8*8*2 + 4
        """
        # d'abord pour les murs, je veux la liste des indices qui
        # correspondent aux coups, et les coups associés comme ça je peux
        # remplir un tableau de zéros aux bons indices
        if self.board_state.player[player_number].n_tuiles > 0:
            all_walls_available = np.transpose(
                np.nonzero(self.board_state.wall_possibilities > 0)
            )
            test_set = (
                100 * all_walls_available[:, 0]
                + 10 * all_walls_available[:, 1]
                + all_walls_available[:, 2]
            )
            isin = np.isin(self.set, test_set, assume_unique=True)
            indices = np.nonzero(isin)
            scores = -1000 * np.ones(len(isin))
            scores[indices] = np.apply_along_axis(
                lambda x: self.coup(x, player_number, get_back=True),
                1,
                all_walls_available,
            )
        else:
            scores = -1000 * np.ones(len(self.set))

        # reste a faire les 4 mouvements possibles, +/-/gauche/droite
        moves_scores = [-1000.0] * 4
        all_moves = self._all_moves(player_number)
        pos = self.board_state.player[player_number].position
        for move in all_moves:
            if (move[0] == pos[0]) & (move[1] > pos[1]):
                moves_scores[0] = self.coup(move, player_number, get_back=True)
            if (move[0] == pos[0]) & (move[1] < pos[1]):
                moves_scores[1] = self.coup(move, player_number, get_back=True)
            if (move[1] == pos[1]) & (move[0] < pos[0]):
                moves_scores[2] = self.coup(move, player_number, get_back=True)
            if (move[1] == pos[1]) & (move[0] > pos[0]):
                moves_scores[3] = self.coup(move, player_number, get_back=True)
        scores = np.r_[scores, moves_scores]
        scores[scores != -1000] -= np.mean(scores[scores != -1000])
        return scores

    def evaluate_all_possibilities(self, player_number):
        """
        for a given board state, test all possibilities,
        score them for player i
        sort them in ascending order
        :param player_number: int
        :return: best possibilities, cost (increasing)
        """
        # mask = self.board_state.wall_possibilities > 0  # type: ndarray
        # all_coups = np.transpose(np.nonzero(mask))
        all_coups = self.all_coups(player_number)
        # attention de temps en temps all_coups est de dimension 1 et ça plante
        all_scores = np.apply_along_axis(
            lambda x: self.coup(x, player_number, get_back=True), 1, all_coups
        )

        tri = all_scores.argsort()
        all_scores = all_scores[tri]
        all_coups = all_coups[tri, :]

        # best_scores = all_scores[-min(len(all_coups),
        # self.n_coups_simultanes):]
        # mask = best_scores != -1000
        # best_scores = best_scores[mask]
        # best_coups = all_coups[-min(len(all_coups),
        #                             self.n_coups_simultanes):, :][mask, :]
        # return best_coups, best_scores
        return all_coups, all_scores