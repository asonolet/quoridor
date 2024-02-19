from quoridor import board_state as bs


def test_bs_players_init():
    board = bs.BoardState()
    assert len(board.players) == 2

def test_next_player_changes():
    board = bs.BoardState()
    first_player = board.next_player
    board.update_player_positions((1, 5, -1))
    assert (first_player + board.next_player) % 2 == 1
