"""Test game module."""

from quoridor.game import Game


def test_game_init():
    """Game init should set board state, coup counter and coup memory."""
    g = Game("Test game")
    bs = g.board_state
    assert bs.coup_joues == [(4, 0, -1), (4, 8, -1)]
    assert bs.next_player_nb == 0


def test_player_positions_with_coup() -> None:
    """Testing move from game init."""
    g = Game("test")
    bs = g.board_state
    g.play((4, 1, -1))
    assert bs.last_player.position == (4, 1)
    assert bs.players[bs.next_player_nb].position == (4, 8)
    g.play((4, 7, -1))
    assert bs.last_player.position == (4, 7)
    assert bs.players[bs.next_player_nb].position == (4, 1)
    g.play((4, 6, 0))
    assert bs.last_player.position == (4, 1)
    assert bs.players[bs.next_player_nb].position == (4, 7)
    bs.get_back()
    assert bs.last_player.position == (4, 7)
    assert bs.players[bs.next_player_nb].position == (4, 1)
    g.play((4, 2, -1))
    assert bs.last_player.position == (4, 2)
    assert bs.players[bs.next_player_nb].position == (4, 7)
    bs.get_back()
    assert bs.last_player.position == (4, 7)
    assert bs.players[bs.next_player_nb].position == (4, 1)
