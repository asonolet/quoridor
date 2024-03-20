"""Exploration related objects.

Those objects are used to explore different path while
remembering the score and updating a global score.
"""

from __future__ import annotations

import numpy as np

from quoridor.game import Game
from quoridor.terminal_plotter import TermPlotter


class PossibleMove:
    """Stores the known future score for this move."""

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

    def add_children(self, child):
        """Add a new children.

        And ensure this children parent is set.
        """
        self.childs.append(child)
        child.parent = self

    @property
    def score(self) -> np.float64:
        """Returns score computed from childrens."""
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
        self.bs = self.game.board_state
        print("Player 0 pos, ", self.bs.player.position)
        print(
            "Player 1 pos, ",
            self.bs.players[bs.last_player_nb].position,
        )
        self.pt = TermPlotter()

    def __call__(self):
        """Explore."""
        self.pt.plot(self.bs)
        i = 0
        # print(game.evaluate_all_possibilities(0))
        while self.bs.winner == -1:
            coup = (
                po.play_seeing_future_rec(self.game)
                if i % 2 == 0
                else po.play_with_proba(self.game, rng=None)
            )
            self.game.play(tuple(coup))
            score = self.game.score()
            print(
                "Player %1d " % (i % 2),
                coup,
                score,
                "position",
                self.bs.last_player.position,
            )
            self.pt.plot(self.bs)
            # time.sleep(0.35)
            i = i + 1
        print(
            "And the winner is ... Player %.1d" % self.bs.winner
        )
        print("Nombre de coup total :", i)
