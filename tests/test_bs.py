from quoridor import board_state as bs


def test_bs_players_init() -> None:
    board = bs.BoardState()
    assert len(board.players) == 2
    assert board.players[0].position == (4, 0)
    assert board.players[1].position == (4, 8)


def test_next_player_changes() -> None:
    board = bs.BoardState()
    first_player = board.next_player
    board.update_player_positions((4, 1, -1))
    assert (first_player + board.next_player) % 2 == 1
    board.update_player_positions((4, 7, -1))
    assert (first_player + board.next_player) % 2 == 0
    print(board.free_paths.keys())
    board.add_new_wall((3, 6, 0))
    assert (first_player + board.next_player) % 2 == 1
    board.remove_wall((3, 6, 0))
    assert (first_player + board.next_player) % 2 == 0
