"""
This module allaws to plot a game in order to better understanding how is the agent learning
"""

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    The ``Plotter`` object is used to plot a game. It can only work if it plots the game from the begining, it doesn't have yet any feature to plot a game from any state it is in.
    """

    def __init__(self):
        plt.figure()
        columns = [i for i in range(0, 10)]
        rows = [i for i in range(0, 10)]
        grid_x, grid_y = np.meshgrid(columns, rows)
        for i in range(len(grid_x)):
            plt.plot(grid_x[i], grid_y[i], "k", linewidth=1)
            plt.plot(grid_y[i], grid_x[i], "k", linewidth=1)
        plt.scatter([4.5], [0.5], c="r", s=80)
        plt.scatter([4.5], [8.5], c="g", s=80)
        plt.pause(0.5)

    def _add_wall(self, player_number, move):
        if player_number % 2 == 0:
            color = "r"
        else:
            color = "g"
        if move[2] == 0:
            y = [move[1] + 1, move[1] + 1]
            x = [move[0], move[0] + 2]
        elif move[2] == 1:
            y = [move[1], move[1] + 2]
            x = [move[0] + 1, move[0] + 1]
        else:
            print("this is not a wall!!!")
            return
        plt.plot(x, y, color, linewidth=10)

    def _move(self, game, player_number, move):
        player_number = player_number % 2
        if player_number == 0:
            color = "r"
        else:
            color = "g"
        a = plt.gca()
        if move[2] == -1:
            for i, point in enumerate(a.collections):
                if np.all(
                    point._offsets
                    == np.array([game.board_state.player[player_number].position])
                    + np.array([[0.5, 0.5]])
                ):
                    a.collections.pop(i)
            plt.pause(0.2)
            plt.scatter([move[0] + 0.5], [move[1] + 0.5], c=color, s=80)

        else:
            print("ce n'est pas un mouvement !!!")
            return

    def play(self, game, player_number, move):
        """
        Used to update the plot of the game.

        Args:
           game (Game): the game to read
           player_number (int): the id of the player whose turn it is
           move (tuple): the move played

        """
        if move[2] == -1:
            self._move(game, player_number, move)
        else:
            self._add_wall(player_number, move)
        plt.pause(0.1)

    def load_board(self, universal_board):
        """
        .. todo::
           write a method to plot a game state from the ``universal_board_state`` variable given.
        """
        return


class TermPlotter:
    """
    The ``TermPlotter`` object is used to plot a game on the terminal. It can only work if it plots the game from the begining, it doesn't have yet any feature to plot a game from any state it is in.
    """

    wall_line = "+  " * 10 + "+"
    pos_line = "| " + "  " * 9 + "|"

    def __init__(self):
        columns = [i for i in range(0, 10)]
        rows = [i for i in range(0, 10)]
        grid_x, grid_y = np.meshgrid(columns, rows)
        for i in range(len(grid_x)):
            plt.plot(grid_x[i], grid_y[i], "k", linewidth=1)
            plt.plot(grid_y[i], grid_x[i], "k", linewidth=1)
        plt.scatter([4.5], [0.5], c="r", s=80)
        plt.scatter([4.5], [8.5], c="g", s=80)

    def _add_wall(self, player_number, move):
        raise NotImplementedError

    def _move(self, game, player_number, move):
        player_number = player_number % 2
        if player_number == 0:
            color = "r"
        else:
            color = "g"
        a = plt.gca()
        if move[2] == -1:
            for i, point in enumerate(a.collections):
                if np.all(
                    point._offsets
                    == np.array([game.board_state.player[player_number].position])
                    + np.array([[0.5, 0.5]])
                ):
                    a.collections.pop(i)
            plt.pause(0.2)
            plt.scatter([move[0] + 0.5], [move[1] + 0.5], c=color, s=80)

        else:
            print("ce n'est pas un mouvement !!!")
            return

    def play(self, game, player_number, move):
        """
        Used to update the plot of the game.

        Args:
           game (Game): the game to read
           player_number (int): the id of the player whose turn it is
           move (tuple): the move played

        """
        if move[2] == -1:
            self._move(game, player_number, move)
        else:
            self._add_wall(player_number, move)
        plt.pause(0.1)

    def load_board(self, universal_board):
        """
        .. todo::
           write a method to plot a game state from the ``universal_board_state`` variable given.
        """
        raise NotImplementedError