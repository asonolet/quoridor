import numpy as np
from scipy import sparse as sp

from quoridor.board_state import BoardState


def score_with_relative_path_length_dif(bs: BoardState) -> float:
    """Calculate the actual relative path length difference between players
    :param player_number: the player for who the reward is calculated
    :return: if one way is blocked -1000, if player won inf, otherwise (
    l2-l1)/l1.
    """
    dist_graph = sp.csgraph.shortest_path(
        bs.free_paths.tocsr(),
        unweighted=True,
        directed=False,
        indices=[bs.player.k_pos, bs.players[bs.last_player].k_pos]
    )  # type: ndarray
    l1 = np.min(
        [dist_graph[0, 8 * bs.last_player + 10 * i_] for i_ in range(9)],
    )
    l2 = np.min(
        [
            dist_graph[1, 8 * bs.next_player + 10 * i_]
            for i_ in range(9)
        ],
    )
    if np.inf in (l1, l2):
        return -1000

    return (l2 - l1) / (l1 + 1)
