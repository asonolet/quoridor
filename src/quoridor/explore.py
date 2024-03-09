"""Exploration related objects.

Those objects are used to explore different path while remembering the the
score and updating a global score.
"""

from __future__ import annotations

import numpy as np


class PossibleMove:
    """Stores the known future score for this move."""

    def __init__(self, player_id, move, parent=None, score=np.nan):
        """PossibleMove initialization."""
        self._player_id: int = player_id
        self._move: tuple[int, int, int] = move
        self._score: np.float64 = score
        self.childs: list = []

        if parent is not None:
            parent.addChildren(self)

    def add_children(self, child):
        """Add a new children and ensure this children parent is set."""
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
        return max(move.score for move in my_future_possible_moves)


class Path:
    """List of moves which forms a part of a play."""

    def __init__(self, init_path=None):
        """Path initialization with eventually the start of a path."""
        if init_path is None:
            init_path = []
        self.moves: list[PossibleMove] = init_path

    def __getitem__(self, ind) -> PossibleMove:
        """Get ind th move."""
        return self.moves[ind]

    def score(self):
        """Path score, based on last move."""
        return self.moves[-1].score
