"""Game module."""

from __future__ import annotations

import numpy as np

from quoridor.board_state import BOARD_SIZE, BoardState
from quoridor.scorer import (
    SCORE_MIN,
    score_with_relative_path_length_dif,
)


class Game:
    """Game object contains anything needed to play."""

    def __init__(self, game_name: str) -> None:
        """Initialise a game.

        :param game_name: The name of the game.
        """
        self.game_name = game_name
        self.board_state = BoardState()

        self.all_walls_choices = np.transpose(
            np.nonzero(self.board_state.wall_possibilities > 0),
        )
        self.set = (
            100 * self.all_walls_choices[:, 0]
            + 10 * self.all_walls_choices[:, 1]
            + self.all_walls_choices[:, 2]
        )

    def play(self, choice: tuple[int, int, int]) -> None:
        """Make a move.

        Update board_state
        if score return relative diff length between paths
        if get back is True, don't change the board state.

        :param choice:
        :param player_number: int
        :param get_back: bool
        :param score_: bool
        :return:
        """
        if choice[2] == -1:
            self.board_state.update_player_positions(choice)
        else:
            self.board_state.add_new_wall(choice)

    def evaluate(self, choice: tuple[int, int, int]):
        """Coup is played, scored and unplayed."""
        self.play(choice)
        score = self.score()
        self.board_state.get_back()
        return score

    def score(self):
        """Score for last player."""
        return score_with_relative_path_length_dif(
            self.board_state
        )

    def _all_moves(self):
        all_moves = []
        for _, k in self.board_state.free_paths[  # noqa:SIM118
            self.board_state.player.k_pos, :
        ].keys():
            new_coup = (k // 10, k % 10, -1)
            new_position = new_coup[:2]
            # In this case, both players are next one another
            if (
                new_position
                == self.board_state.last_player.position
            ):
                old_pos = np.array(
                    self.board_state.player.position
                )
                other_player_position = np.array(new_position)
                new_coup = tuple(
                    np.r_[
                        2 * other_player_position - old_pos, -1
                    ]
                )
                if (
                    (0 < new_coup[0] < BOARD_SIZE)
                    and (0 < new_coup[1] < BOARD_SIZE)
                    and self.board_state.free_paths[
                        10 * new_coup[0] + new_coup[1],
                        10 * other_player_position[0]
                        + other_player_position[1],
                    ]
                ):
                    all_moves.append(new_coup)
            else:
                all_moves.append(new_coup)
        return all_moves

    def _all_walls(self):
        return np.transpose(
            np.nonzero(self.board_state.wall_possibilities > 0),
        )

    def _all_possibilities(self):
        """Return current possibilities.

        List of possible moves and possible wall placements.
        Does not take into account the rule about not enclosing
        someone.
        """
        all_moves = self._all_moves()
        if self.board_state.player.n_tuiles > 0:
            all_walls = self._all_walls()
            all_coups = np.concatenate((all_moves, all_walls))
        else:
            all_coups = np.array(all_moves)
        return all_coups

    def _moves_allowed(self):
        all_moves = self._all_moves()
        pos = self.board_state.player.position
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

    def evaluate_all_choices(self):
        """Score all choices with consistent order.

        If the choice is not available, score is 0
        :return: score vector of length 8*8*2 + 4.
        """
        # d'abord pour les murs, je veux la liste des indices qui
        # correspondent aux coups, et les coups associés comme
        # ça je peux
        # remplir un tableau de zéros aux bons indices
        if self.board_state.player.n_tuiles > 0:
            all_walls_available = np.transpose(
                np.nonzero(
                    self.board_state.wall_possibilities > 0
                ),
            )
            test_set = (
                100 * all_walls_available[:, 0]
                + 10 * all_walls_available[:, 1]
                + all_walls_available[:, 2]
            )
            isin = np.isin(
                self.set, test_set, assume_unique=True
            )
            indices = np.nonzero(isin)
            scores = SCORE_MIN * np.ones(len(isin))
            scores[indices] = np.apply_along_axis(
                lambda x: self.evaluate(tuple(x)),
                1,
                all_walls_available,
            )
        else:
            scores = SCORE_MIN * np.ones(len(self.set))

        # reste a faire les 4 mouvements possibles,
        # +/-/gauche/droite
        moves_scores = [SCORE_MIN] * 4
        all_moves = self._all_moves()
        pos = self.board_state.player.position
        for move in all_moves:
            if (move[0] == pos[0]) & (move[1] > pos[1]):
                moves_scores[0] = self.evaluate(move)
            if (move[0] == pos[0]) & (move[1] < pos[1]):
                moves_scores[1] = self.evaluate(move)
            if (move[1] == pos[1]) & (move[0] < pos[0]):
                moves_scores[2] = self.evaluate(move)
            if (move[1] == pos[1]) & (move[0] > pos[0]):
                moves_scores[3] = self.evaluate(move)
        scores = np.r_[scores, moves_scores]
        scores[scores != SCORE_MIN] -= np.mean(
            scores[scores != SCORE_MIN]
        )
        return scores

    def evaluate_all_possibilities(self):
        """For a given board state, test all possibilities.

        Score them for player i
        Sort them in ascending order

        :param player_number: int
        :return: best possibilities, cost (increasing).
        """
        all_coups = self._all_possibilities()
        # attention de temps en temps all_coups est de dimension
        # 1 et ça plante
        all_scores = np.apply_along_axis(
            lambda x: self.evaluate(tuple(x)),
            1,
            all_coups,
        )

        tri = all_scores.argsort()
        all_scores = all_scores[tri]
        all_coups = all_coups[tri, :]

        # return best_coups, best_scores
        return all_coups, all_scores
