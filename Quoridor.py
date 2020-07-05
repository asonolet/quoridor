from typing import Tuple

import numpy as np
import time

from numpy.core.multiarray import ndarray
from scipy import sparse as sp


class Game:
    def __init__(self, game_name):

        self.game_name = game_name
        self.board_state = BoardState()

        self.coup_joues = []
        self.coup_joues.append((4, 0, -1))
        self.coup_joues.append((4, 8, -1))

        self.all_walls_choices = np.transpose(np.nonzero(
            self.board_state.wall_possibilities > 0))
        self.set = 100 * self.all_walls_choices[:, 0] \
            + 10 * self.all_walls_choices[:, 1] + self.all_walls_choices[:, 2]

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
            dist = self.board_state.score_with_relative_path_length_dif(
                player_number)
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
        for _, k in self.board_state.free_paths[self.board_state.player[
                player_number].k_pos, :].keys():
            new_coup = (k // 10, k % 10, -1)
            new_position = new_coup[:2]
            # In this case, both players are next one another
            if new_position == self.board_state.player[
                    1 - player_number].position:
                old_pos = np.array(self.board_state.player[
                                       player_number].position)
                new_position = np.array(new_position)
                new_coup = tuple(np.r_[2 * new_position - old_pos, -1])
                if (0 < new_coup[0] < 9) & (0 < new_coup[1] < 9):
                    if self.board_state.free_paths[
                        10 * new_coup[0] + new_coup[1], new_position[0] * 10 +
                            new_position[1]]:
                        all_moves.append(new_coup)
            else:
                all_moves.append(new_coup)
        return all_moves

    def all_coups(self, player_number: int):
        all_moves = self._all_moves(player_number)
        if self.board_state.player[player_number].n_tuiles > 0:
            all_walls = np.transpose(np.nonzero(
                self.board_state.wall_possibilities > 0))
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
            all_walls_available = np.transpose(np.nonzero(
                self.board_state.wall_possibilities > 0))
            test_set = 100 * all_walls_available[:, 0] + \
                10 * all_walls_available[:, 1] + \
                all_walls_available[:, 2]
            isin = np.isin(self.set, test_set, assume_unique=True)
            indices = np.nonzero(isin)
            scores = -1000 * np.ones(len(isin))
            scores[indices] = \
                np.apply_along_axis(lambda x: self.coup(x, player_number,
                                                        get_back=True), 1,
                                    all_walls_available)
        else:
            scores = -1000 * np.ones(len(self.set))

        # reste a faire les 4 mouvements possibles, +/-/gauche/droite
        moves_scores = [-1000.] * 4
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

    def evaluate_all_possibilities(self, player_number: int):
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
        all_scores = np.apply_along_axis(lambda x: self.coup(x, player_number,
                                                             get_back=True), 1,
                                         all_coups)

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


class Player:
    def __init__(self, player_number: int):
        """
        each player gets a name, a number of wall, a position (i,j) and a
        position k
        :param player_number:
        """
        self.name = player_number
        self.n_tuiles = 10
        if player_number == 0:
            self.position = (4, 0)
        else:
            self.position = (4, 8)

    @property
    def k_pos(self):
        return 10 * self.position[0] + self.position[1]


class BoardState:
    def __init__(self):

        self.wall_possibilities = np.ones((8, 8, 2))
        self.player = [Player(0), Player(1)]

        def init_free_paths():
            node_links = np.zeros((89, 89), dtype=bool)
            for x in range(9):
                for y in range(9):
                    if y < 8:
                        node_links[10 * x + y, 10 * x + y + 1] = 1
                    if y > 0:
                        node_links[10 * x + y, 10 * x + y - 1] = 1
                    if x < 8:
                        node_links[10 * x + y, 10 * (x + 1) + y] = 1
                    if x > 0:
                        node_links[10 * x + y, 10 * (x - 1) + y] = 1
            links_graph = sp.dok_matrix(node_links)
            return links_graph

        self.free_paths = init_free_paths()
        self.winner = -1

    def update_player_positions(self, new_position, player_number):
        """
        update player position and actualize the winner
        :param new_position: [i,j,-1]
        :param player_number: int
        :return:
        """
        self.player[player_number].position = tuple(new_position[:2])
        # self.player[player_number].k_pos = 10 * new_position[0] +
        # new_position[
        #     1]

        if player_number:
            if new_position[1] == 0:
                self.winner = player_number
        else:
            if new_position[1] == 8:
                self.winner = player_number

    def add_new_wall(self, new_position, player_number):
        """
        add wall by modifying wall possibilities,
        update free_paths,
        update player remaining walls number

        :param new_position: [i,j,0] or [i,j,1]
        :param player_number: int
        :return:
        """
        self.player[player_number].n_tuiles -= 1
        x, y = new_position[0:2]
        k = 10 * x + y
        self.wall_possibilities[x, y, 0] -= 1
        self.wall_possibilities[x, y, 1] -= 1
        if new_position[2] == 0:
            if x < 7:
                self.wall_possibilities[x + 1, y, 0] -= 1
            if x > 0:
                self.wall_possibilities[x - 1, y, 0] -= 1
        if new_position[2] == 1:
            if y < 7:
                self.wall_possibilities[x, y + 1, 1] -= 1
            if y > 0:
                self.wall_possibilities[x, y - 1, 1] -= 1

        # removing blocked path from free_paths
        if new_position[2] == 0:
            self.free_paths.pop((k, k + 1))
            self.free_paths.pop((k + 1, k))
            self.free_paths.pop((k + 10, k + 11))
            self.free_paths.pop((k + 11, k + 10))
        if new_position[2] == 1:
            self.free_paths.pop((k, k + 10))
            self.free_paths.pop((k + 10, k))
            self.free_paths.pop((k + 1, k + 11))
            self.free_paths.pop((k + 11, k + 1))

    def remove_wall(self, new_position, player_number):
        """
        add one wall in remaining player walls,
        update wall possibilities,
        update free_paths

        :param new_position:
        :param player_number:
        :return:
        """
        self.player[player_number].n_tuiles += 1
        x, y = new_position[0], new_position[1]
        k = 10 * x + y
        self.wall_possibilities[x, y, 0] = min(1, self.wall_possibilities[x,
                                                                          y,
                                                                          0]
                                               + 1)
        self.wall_possibilities[x, y, 1] = min(1, self.wall_possibilities[x,
                                                                          y,
                                                                          1]
                                               + 1)

        if new_position[2] == 0:
            if x < 7:
                self.wall_possibilities[x + 1, y, 0] = \
                    min(1, self.wall_possibilities[x + 1, y, 0] + 1)
            if x > 0:
                self.wall_possibilities[x - 1, y, 0] = \
                    min(1, self.wall_possibilities[x - 1, y, 0] + 1)

        if new_position[2] == 1:
            if y < 7:
                self.wall_possibilities[x, y + 1, 1] = \
                    min(1, self.wall_possibilities[x, y + 1, 1] + 1)
            if y > 0:
                self.wall_possibilities[x, y - 1, 1] = \
                    min(1, self.wall_possibilities[x, y - 1, 1] + 1)

        # adding the path that was blocked
        if new_position[2] == 0:
            self.free_paths[k, k + 1] = 1
            self.free_paths[k + 1, k] = 1
            self.free_paths[k + 10, k + 11] = 1
            self.free_paths[k + 11, k + 10] = 1
        if new_position[2] == 1:
            self.free_paths[k, k + 10] = 1
            self.free_paths[k + 10, k] = 1
            self.free_paths[k + 1, k + 11] = 1
            self.free_paths[k + 11, k + 1] = 1

    def score_with_relative_path_length_dif(self, player_number: int) -> float:
        """
        calculate the actual relative path length difference between players
        :param player_number: the player for who the reward is calculated
        :return: if one way is blocked -1000, if player won inf, otherwise (
        l2-l1)/l1
        """
        dist_graph = sp.csgraph.shortest_path(self.free_paths.tocsr(),
                                              unweighted=True,
                                              directed=False)  # type: ndarray
        l1 = np.min([dist_graph[self.player[player_number].k_pos,
                                8 * (1 - player_number) + 10 * i_] for i_ in
                     range(9)])
        l2 = np.min([dist_graph[self.player[1 - player_number].k_pos,
                                8 * player_number + 10 * i_] for i_ in range(
            9)])
        if (l1 == np.inf) or (l2 == np.inf):
            return -1000
        # if l1 == 0:
        #     return np.inf
        return (l2 - l1) / (l1 + 1)

    def to_universal_state(self, i_):
        return np.r_[np.ravel(self.wall_possibilities), np.array(self.player[
            i_ % 2].position)/8, np.array(self.player[(i_ + 1) % 2].position)/8,
                     self.player[i_ % 2].n_tuiles/10, self.player[(i_ + 1) %
                                                               2].n_tuiles/10]


def play_greedy(game_, player_number):
    return game_.evaluate_all_possibilities(player_number)[0][-1]


def play_random(game_, player_number):
    res = game_.evaluate_all_possibilities(player_number)[0]
    return res[np.random.randint(len(res))]


def play_with_proba(game_, player_number):
    res, cost = game_.evaluate_all_possibilities(player_number)
    maxi = np.max(cost)
    cost = cost + 1 - maxi
    cost_ = np.where(cost > -500, np.exp(100 * cost), 0)
    cost_ = cost_ / np.sum(cost_)
    return res[np.random.choice(len(cost), p=cost_)]


def play_seeing_future_rec(game_, player_number, n_sim=3, n_future=3,
                           counter=0, returned_scores=None):
    if game_.board_state.winner != -1:
        if player_number == game_.board_state.winner:
            return 1000
        else:
            return -1000
    else:
        if 2 * n_future == counter:
            choices, scores = game_.evaluate_all_possibilities(player_number)
            return [scores[-1]]
        else:
            choices, _ = game_.evaluate_all_possibilities((player_number +
                                                           counter) % 2)
            n_sim_possible = min(n_sim, len(choices))
            choices = choices[-n_sim_possible:]
            if returned_scores is None:
                returned_scores = []
            for i_sim in range(n_sim_possible):
                game_.coup(choices[i_sim], (player_number + counter) % 2,
                           score_=False)
                scores = play_seeing_future_rec(game_, player_number,
                                                n_sim, n_future,
                                                counter + 1)
                game_.get_back(1)
                if counter % 2 == 0:
                    # returned_scores.append(np.mean(scores))
                    returned_scores.append(np.max(scores))
                else:
                    returned_scores.append(np.max(scores))
            if counter == 0:
                return choices[np.argmax(returned_scores)], np.max(
                    returned_scores)
            else:
                return returned_scores


if __name__ == '__main__':
    from graphic_quoridor import Plotter

    game = Game('partie 1')
    pt = Plotter()
    debut = time.time()
    i = 0
    # print(game.evaluate_all_possibilities(0))
    while game.board_state.winner == -1:
        a = np.random.uniform(0, 1)
        if i % 2 == 0:
            coup = play_greedy(game, i % 2)
        else:
            # coup, _ = play_seeing_future_rec(game, i % 2, n_sim=1, n_future=4)
            coup = play_with_proba(game, i % 2)
            # coup, _ = play_seeing_future_rec(game, i % 2, 2, 1)
            # coup = play_greedy(game, i % 2)
        pt.play(game, i % 2, coup)
        score = game.coup(coup, i % 2, False, True)
        print('Player %1d ' % (i % 2), coup, score)
        i = i + 1
    print("And the winner is ... Player %.1d" % game.board_state.winner)
    fin = time.time()
    temps = fin - debut
    print('Temps moyen par coup : ', temps / i)
    print('Nombre de coup total :', i)
