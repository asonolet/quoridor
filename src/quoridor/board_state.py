"""Board state module."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

BOARD_SIZE = 9


class Player:
    """Player data holder."""

    name: int
    n_tuiles: int
    position: tuple[int, int]

    def __init__(self, player_number: int) -> None:
        """Player class initialisation.

        Each player gets a name, a number of wall, a position (i,j) and a position k.

        :param player_number:
        """
        self.name = player_number
        self.n_tuiles = 10
        if player_number == 0:
            self.position = (4, 0)
        else:
            self.position = (4, BOARD_SIZE - 1)

    @property
    def k_pos(self) -> int:
        """Transforms the position to a signle int."""
        return 10 * self.position[0] + self.position[1]


def _init_free_paths() -> sp.dok_matrix:
    node_links = np.zeros((89, 89), dtype=bool)
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if y < BOARD_SIZE - 1:
                node_links[10 * x + y, 10 * x + y + 1] = 1
            if y > 0:
                node_links[10 * x + y, 10 * x + y - 1] = 1
            if x < BOARD_SIZE - 1:
                node_links[10 * x + y, 10 * (x + 1) + y] = 1
            if x > 0:
                node_links[10 * x + y, 10 * (x - 1) + y] = 1
    return sp.dok_matrix(node_links)


class BoardState:
    """BoardState object is used to play at one instant.

    A game is a succession of BoardState.
    Methods to pass from one BoardState to an other are declared here.
    """

    walls: set[tuple[int, int, int]]

    def __init__(self, first_player: int = 0) -> None:
        """Initialize the board state as the begining of a game.

        :param first_player: the first player to play. Player 0 start
        from bottom and player 1 at top (j=BOARD_SIZE - 1).
        """
        self.wall_possibilities = np.ones((BOARD_SIZE - 1, BOARD_SIZE - 1, 2))
        self.players = [Player(0), Player(1)]
        self.played_coup = 0
        self._first_player = first_player
        self.free_paths = _init_free_paths()
        self.winner = -1
        self.walls = set()

    @property
    def next_player_nb(self) -> int:
        """The id of the player currently playing."""
        return (self._first_player + self.played_coup) % 2

    @property
    def last_player_nb(self) -> int:
        """The id of the player which just played."""
        return (self._first_player + self.played_coup + 1) % 2

    @property
    def player(self) -> Player:
        """Current player."""
        return self.players[self.next_player_nb]

    @property
    def last_player(self) -> Player:
        """The player who just played."""
        return self.players[self.last_player_nb]

    def update_player_positions(self, new_position: tuple[int, int, int]) -> None:
        """Update player position and actualize the winner.

        :param new_position: [i,j,-1]
        :param player_number: int
        :return:
        """
        if (self.next_player_nb and new_position[1] == 0) or (
            not self.next_player_nb and new_position[1] == BOARD_SIZE - 1
        ):
            self.winner = self.next_player_nb
        self.player.position = (new_position[0], new_position[1])
        self.played_coup += 1

    def reset_player_position(self, last_pos: tuple[int, int, int]) -> None:
        """Reset player position to last position.

        :param new_position: [i,j,-1]
        :param player_number: int
        :return:
        """
        self.last_player.position = (last_pos[0], last_pos[1])
        self.played_coup -= 1

    def add_new_wall(self, new_position: tuple[int, int, int]) -> None:
        """Add wall.

        Modify wall possibilities,
        update free_paths,
        update player remaining walls number.

        :param new_position: (i,j,0) or (i,j,1)
        :param player_number: int
        :return:
        """
        self.player.n_tuiles -= 1
        self.walls.add(new_position)
        x, y = new_position[0:2]
        k = 10 * x + y
        self.wall_possibilities[x, y, 0] -= 1
        self.wall_possibilities[x, y, 1] -= 1
        if new_position[2] == 0:
            if x < BOARD_SIZE - 2:
                self.wall_possibilities[x + 1, y, 0] -= 1
            if x > 0:
                self.wall_possibilities[x - 1, y, 0] -= 1
        if new_position[2] == 1:
            if y < BOARD_SIZE - 2:
                self.wall_possibilities[x, y + 1, 1] -= 1
            if y > 0:
                self.wall_possibilities[x, y - 1, 1] -= 1

        # removing blocked path from free_paths
        if new_position[2] == 0:
            self.free_paths[k, k + 1] = False
            self.free_paths[k + 1, k] = False
            self.free_paths[k + 10, k + 11] = False
            self.free_paths[k + 11, k + 10] = False
        if new_position[2] == 1:
            self.free_paths[k, k + 10] = False
            self.free_paths[k + 10, k] = False
            self.free_paths[k + 1, k + 11] = False
            self.free_paths[k + 11, k + 1] = False
        self.played_coup += 1

    def remove_wall(self, new_position: tuple[int, int, int]) -> None:
        """Undo an add wall operation.

        Add one wall in remaining player walls,
        update wall possibilities,
        update free_paths.
        update walls.
        If last int is 9, wall is horizontal, otherwise it is
        vertical.

        :param new_position:
        :param player_number:
        :return:
        """
        self.player.n_tuiles += 1
        self.walls.remove(new_position)
        x, y = new_position[0], new_position[1]
        k = 10 * x + y
        self.wall_possibilities[x, y, 0] = min(1, self.wall_possibilities[x, y, 0] + 1)
        self.wall_possibilities[x, y, 1] = min(1, self.wall_possibilities[x, y, 1] + 1)

        if new_position[2] == 0:
            if x < BOARD_SIZE - 2:
                self.wall_possibilities[x + 1, y, 0] = min(
                    1,
                    self.wall_possibilities[x + 1, y, 0] + 1,
                )
            if x > 0:
                self.wall_possibilities[x - 1, y, 0] = min(
                    1,
                    self.wall_possibilities[x - 1, y, 0] + 1,
                )

        if new_position[2] == 1:
            if y < BOARD_SIZE - 2:
                self.wall_possibilities[x, y + 1, 1] = min(
                    1,
                    self.wall_possibilities[x, y + 1, 1] + 1,
                )
            if y > 0:
                self.wall_possibilities[x, y - 1, 1] = min(
                    1,
                    self.wall_possibilities[x, y - 1, 1] + 1,
                )

        # adding the path that was blocked
        if new_position[2] == 0:
            self.free_paths[k, k + 1] = 1
            self.free_paths[k + 1, k] = 1
            self.free_paths[k + 10, k + 11] = 1
            self.free_paths[k + 11, k + 10] = 1
        if new_position[2] == 1:
            self.free_paths[k, k + 10] = 1
            self.free_paths[k + 10, k] = 1
            self.free_paths[k + 1, k + 11] = 1
            self.free_paths[k + 11, k + 1] = 1
        self.played_coup -= 1

    def to_universal_state(self, i_):
        """Supposed to be a universal representation."""
        return np.r_[
            np.ravel(self.wall_possibilities),
            np.array(self.players[i_ % 2].position) / (BOARD_SIZE - 1),
            np.array(self.players[(i_ + 1) % 2].position) / (BOARD_SIZE - 1),
            self.players[i_ % 2].n_tuiles / 10,
            self.players[(i_ + 1) % 2].n_tuiles / 10,
        ]
