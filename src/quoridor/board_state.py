import numpy as np
from scipy import sparse as sp


class Player:
    """each player gets a name, a number of wall, a position (i,j) and a
    position k
    :param player_number:
    """

    def __init__(self, player_number: int):
        self.name = player_number
        self.n_tuiles = 10
        if player_number == 0:
            self.position = (4, 0)
        else:
            self.position = (4, 8)

    @property
    def k_pos(self):
        return 10 * self.position[0] + self.position[1]


def _init_free_paths():
    node_links = np.zeros((89, 89), dtype=bool)
    for x in range(9):
        for y in range(9):
            if y < 8:
                node_links[10 * x + y, 10 * x + y + 1] = 1
            if y > 0:
                node_links[10 * x + y, 10 * x + y - 1] = 1
            if x < 8:
                node_links[10 * x + y, 10 * (x + 1) + y] = 1
            if x > 0:
                node_links[10 * x + y, 10 * (x - 1) + y] = 1
    links_graph = sp.dok_matrix(node_links)
    return links_graph


class BoardState:
    """This object is used to play at one instant. A game is a succession of BoardState.
    Methods to pass from one BoardState to an other are declared here.
    """

    def __init__(self, first_player=0):
        self.wall_possibilities = np.ones((8, 8, 2))
        self.players = [Player(0), Player(1)]
        self.played_coup = 0
        self._first_player = first_player
        self.free_paths = _init_free_paths()
        self.winner = -1

    @property
    def next_player(self):
        return (self._first_player + self.played_coup) % 2

    @property
    def last_player(self):
        return (self._first_player + self.played_coup + 1) % 2

    @property
    def player(self):
        return self.players[self.next_player]

    def update_player_positions(self, new_position):
        """Update player position and actualize the winner
        :param new_position: [i,j,-1]
        :param player_number: int
        :return:
        """
        self.player.position = tuple(new_position[:2])

        if (self.next_player and new_position[1] == 0) or (
            not self.next_player and new_position[1] == 8
        ):
            self.winner = self.next_player
        self.played_coup += 1

    def add_new_wall(self, new_position):
        """Add wall by modifying wall possibilities,
        update free_paths,
        update player remaining walls number

        :param new_position: [i,j,0] or [i,j,1]
        :param player_number: int
        :return:
        """
        self.player.n_tuiles -= 1
        x, y = new_position[0:2]
        k = 10 * x + y
        self.wall_possibilities[x, y, 0] -= 1
        self.wall_possibilities[x, y, 1] -= 1
        if new_position[2] == 0:
            if x < 7:
                self.wall_possibilities[x + 1, y, 0] -= 1
            if x > 0:
                self.wall_possibilities[x - 1, y, 0] -= 1
        if new_position[2] == 1:
            if y < 7:
                self.wall_possibilities[x, y + 1, 1] -= 1
            if y > 0:
                self.wall_possibilities[x, y - 1, 1] -= 1

        # removing blocked path from free_paths
        if new_position[2] == 0:
            self.free_paths.pop((k, k + 1))
            self.free_paths.pop((k + 1, k))
            self.free_paths.pop((k + 10, k + 11))
            self.free_paths.pop((k + 11, k + 10))
        if new_position[2] == 1:
            self.free_paths.pop((k, k + 10))
            self.free_paths.pop((k + 10, k))
            self.free_paths.pop((k + 1, k + 11))
            self.free_paths.pop((k + 11, k + 1))
        self.played_coup += 1

    def remove_wall(self, new_position):
        """Add one wall in remaining player walls,
        update wall possibilities,
        update free_paths

        :param new_position:
        :param player_number:
        :return:
        """
        self.player.n_tuiles += 1
        x, y = new_position[0], new_position[1]
        k = 10 * x + y
        self.wall_possibilities[x, y, 0] = min(1, self.wall_possibilities[x, y, 0] + 1)
        self.wall_possibilities[x, y, 1] = min(1, self.wall_possibilities[x, y, 1] + 1)

        if new_position[2] == 0:
            if x < 7:
                self.wall_possibilities[x + 1, y, 0] = min(
                    1, self.wall_possibilities[x + 1, y, 0] + 1,
                )
            if x > 0:
                self.wall_possibilities[x - 1, y, 0] = min(
                    1, self.wall_possibilities[x - 1, y, 0] + 1,
                )

        if new_position[2] == 1:
            if y < 7:
                self.wall_possibilities[x, y + 1, 1] = min(
                    1, self.wall_possibilities[x, y + 1, 1] + 1,
                )
            if y > 0:
                self.wall_possibilities[x, y - 1, 1] = min(
                    1, self.wall_possibilities[x, y - 1, 1] + 1,
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
        return np.r_[
            np.ravel(self.wall_possibilities),
            np.array(self.players[i_ % 2].position) / 8,
            np.array(self.players[(i_ + 1) % 2].position) / 8,
            self.players[i_ % 2].n_tuiles / 10,
            self.players[(i_ + 1) % 2].n_tuiles / 10,
        ]
