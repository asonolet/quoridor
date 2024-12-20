"""Scorers for board state."""

import numpy as np
from scipy import sparse as sp

from quoridor.board_state import BoardState

SCORE_MIN = -1000
SCORE_MAX = 1000


def score_with_relative_path_length_dif(bs: BoardState) -> float:
    """Calculate a score.

    The score is the actual relative path length difference
    between players.

    :return: if one way is blocked -1000, if player won inf,
       otherwise (l2-l1)/l1.

    """
    if bs.winner == bs.last_player_nb:
        return SCORE_MAX
    if bs.winner == bs.next_player_nb:
        return SCORE_MIN

    dist_graph = sp.csgraph.shortest_path(
        bs.free_paths.tocsr(),
        unweighted=True,
        directed=False,
        indices=[bs.player.k_pos, bs.last_player.k_pos],
    )  # type: np.ndarray
    l2 = np.min(
        [
            dist_graph[0, 8 * bs.last_player_nb + 10 * i_]
            for i_ in range(9)
        ],
    )
    l1 = np.min(
        [
            dist_graph[1, 8 * bs.next_player_nb + 10 * i_]
            for i_ in range(9)
        ],
    )
    if np.inf in (l1, l2):
        return SCORE_MIN

    return (l2 - l1) / (l1 + l2)
