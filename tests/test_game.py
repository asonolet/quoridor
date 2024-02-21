"""Test game module."""

from quoridor.game import Game


def test_game_init():
    """Game init should set board state, coup counter and coup memory."""
    g = Game("Test game")
    assert g.coup_joues == [(4, 0, -1), (4, 8, -1)]
    bs = g.board_state
    assert bs.next_player == 0
