"""Tests for the board_state module."""

from quoridor import board_state as bs

PLAYERS_NUMBER = 2


def test_bs_players_init() -> None:
    """Initial player positions."""
    board = bs.BoardState()
    assert len(board.players) == PLAYERS_NUMBER
    assert board.players[0].position == (4, 0)
    assert board.players[1].position == (4, 8)


def test_next_player_changes() -> None:
    """next_player switches when position is updated, a wall is played or removed."""
    board = bs.BoardState()
    first_player = board.next_player
    board.update_player_positions((4, 1, -1))
    assert (first_player + board.next_player) % 2 == 1
    board.update_player_positions((4, 7, -1))
    assert (first_player + board.next_player) % 2 == 0
    board.add_new_wall((3, 6, 0))
    assert (first_player + board.next_player) % 2 == 1
    board.remove_wall((3, 6, 0))
    assert (first_player + board.next_player) % 2 == 0


def test_add_remove_wall() -> None:
    """Add wall changes free_paths, same with remove wall."""
    board = bs.BoardState()
    board.add_new_wall((4, 1, 0))
    assert not board.free_paths[41, 42]
    board.remove_wall((4, 1, 0))
    assert board.free_paths[41, 42]


def test_change_position() -> None:
    """Positions are set as expected."""
    board = bs.BoardState()
    board.update_player_positions((4, 1, -1))
    assert board.players[0].position == (4, 1)
    assert board.players[1].position == (4, 8)
    board.update_player_positions((4, 7, -1))
    assert board.players[0].position == (4, 1)
    assert board.players[1].position == (4, 7)
