from quoridor import board_state as bs


def test_bs_players_init():
    board = bs.BoardState()
    assert len(board.players) == 2
    assert board.players[0].position == (0, 5, -1)
    assert board.players[1].position == (8, 5, -1)

def test_next_player_changes():
    board = bs.BoardState()
    first_player = board.next_player
    board.update_player_positions((1, 5, -1))
    assert (first_player + board.next_player) % 2 == 1
    board.update_player_positions((7, 5, -1))
    assert (first_player + board.next_player) % 2 == 0
    board.add_new_wall((6, 5, 0))
    assert (first_player + board.next_player) % 2 == 1
    board.remove_wall((6,5,0))
    assert (first_player + board.next_player) % 2 == 0