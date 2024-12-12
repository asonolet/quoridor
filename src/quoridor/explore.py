"""Exploration related objects.

Those objects are used to explore different path while
remembering the score and updating a global score.
"""

from __future__ import annotations

import numpy as np

import quoridor.policy as po
from quoridor.game import Game
from quoridor.terminal_plotter import TermPlotter


class PossibleMove:
    """Stores the known future score for this move.

    Be careful, the adv moves are stored but not acounted for
    scoring.
    """

    def __init__(
        self, player_id, move, parent=None, score=np.nan
    ):
        """PossibleMove initialization."""
        self._player_id: int = player_id
        self._move: tuple[int, int, int] = move
        self._score: np.float64 = score
        self.childs: list[PossibleMove] = []

        if parent is not None:
            parent.addChildren(self)
        else:
            self.parent = None

    def add_children(self, child):
        """Add a new children.

        And ensure this children parent is set.
        """
        self.childs.append(child)
        child.parent = self

    @property
    def score(self) -> np.float64:
        """Returns score computed from childrens.

        Does not look at adversary scores.
        """
        if len(self.childs) == 0:
            return self._score

        my_future_possible_moves = []
        for adv_move in self.childs:
            my_future_possible_moves += adv_move.childs

        if len(my_future_possible_moves) == 0:
            return self._score

        return max(
            move.score for move in my_future_possible_moves
        )


class TreeOperations:
    """Static methods to go through a tree."""

    @staticmethod
    def get_path_from_state1_to_state2(state1, state2):
        """As the name implies.

        Returns
        -------
        (int, Path): The number of move to unplay and the
            remaining path to state2.

        """
        state1_path = []
        state2_path = []
        state = state1
        while state.parent is not None:
            state1_path.append(state)
            state = state.parent
        state1_path.reverse()
        state = state2
        while state.parent is not None:
            state2_path.append(state)
            state = state.parent
        state2_path.reverse()
        i = 0
        while state1_path[i] == state2_path[i]:
            i += 1
        return len(state1_path) - i, state2_path[i:]


class BestMoves:
    """Store best moves."""

    def __init__(self, length=4):
        """Initialise the list of best moves."""

        self._best_moves = np.empty(
            (length,), dtype=PossibleMove
        )
        self._scores = np.empty((length,), dtype=float)
        self._min_score = 0.0

    def __getitem__(self, i):
        """Get the i-th best move. 0 is the poorest stored."""
        return self._best_moves[np.argsort(self._scores)][i]

    def store_if_needed(self, move: PossibleMove):
        """If score is higher than min score replace it."""
        score = move.score
        if score > self._min_score:
            self._min_score = score
            self._best_moves[np.argsort(self._scores)][0] = move
            self._scores[np.argsort(self._scores)][0] = score


class Path:
    """List of moves which forms a part of a play."""

    def __init__(self, init_path=None):
        """Path initialization with the start of a path."""
        if init_path is None:
            init_path = []
        self.moves: list[PossibleMove] = init_path

    def __getitem__(self, ind) -> PossibleMove:
        """Get ind th move."""
        return self.moves[ind]

    def score(self):
        """Path score, based on last move."""
        return self.moves[-1].score


class Exploration:
    """Allows for clever space exploration.

    Stores tested moves and associated scores.
    """

    def __init__(self):
        """Init an exploration."""
        self.game: Game = Game("partie 1")
        self.best_moves = BestMoves()

        self.bs = self.game.board_state
        self.pt = TermPlotter()

        print("Player 0 pos, ", self.bs.player.position)
        print("Player 1 pos, ", self.bs.last_player.position)

    def __call__(self):
        """Explore."""
        self.pt.plot(self.bs)
        i = 0
        last_possible_move = None
        while self.bs.winner == -1:
            coup = tuple(
                po.play_greedy(self.game)
                if i % 2 == 0
                else po.play_with_proba(self.game, rng=None)
            )
            self.game.play(coup)
            score = self.game.score()
            last_possible_move = PossibleMove(
                i % 2,
                coup,
                parent=last_possible_move,
                score=score,
            )
            self.best_moves.store_if_needed(last_possible_move)
            print(
                "Player %1d " % (i % 2),
                coup,
                score,
                "position",
                self.bs.last_player.position,
            )
            self.pt.plot(self.bs)
            i = i + 1
        print(
            "And the winner is ... Player %.1d" % self.bs.winner
        )
        print("Nombre de coup total :", i)
