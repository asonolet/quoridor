import numpy as np
from scipy import sparse as sp

from quoridor.board_state import BoardState


def score_with_relative_path_length_dif(bs: BoardState, player_number: int) -> float:
    """
    calculate the actual relative path length difference between players
    :param player_number: the player for who the reward is calculated
    :return: if one way is blocked -1000, if player won inf, otherwise (
    l2-l1)/l1
    """
    dist_graph = sp.csgraph.shortest_path(
        bs.free_paths.tocsr(), unweighted=True, directed=False
    )  # type: ndarray
    l1 = np.min(
        [
            dist_graph[
                bs.player[player_number].k_pos, 8 * (1 - player_number) + 10 * i_
            ]
            for i_ in range(9)
        ]
    )
    l2 = np.min(
        [
            dist_graph[bs.player[1 - player_number].k_pos, 8 * player_number + 10 * i_]
            for i_ in range(9)
        ]
    )
    if (l1 == np.inf) or (l2 == np.inf):
        return -1000
    # if l1 == 0:
    #     return np.inf
    return (l2 - l1) / (l1 + 1)
