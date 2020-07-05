import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        plt.figure()
        columns = [i for i in range(0, 10)]
        rows = [i for i in range(0, 10)]
        grid_x, grid_y = np.meshgrid(columns, rows)
        for i in range(len(grid_x)):
            plt.plot(grid_x[i], grid_y[i], 'k', linewidth=1)
            plt.plot(grid_y[i], grid_x[i], 'k', linewidth=1)
        plt.scatter([4.5], [0.5], c='r', s=80)
        plt.scatter([4.5], [8.5], c='g', s=80)
        plt.pause(0.5)

    def add_wall(self, player_number, coup):
        if player_number % 2 == 0:
            color = 'r'
        else:
            color = 'g'
        if coup[2] == 0:
            y = [coup[1]+1, coup[1]+1]
            x = [coup[0], coup[0] + 2]
        elif coup[2] == 1:
            y = [coup[1], coup[1] + 2]
            x = [coup[0]+1, coup[0]+1]
        else:
            print('this is not a wall!!!')
            return
        plt.plot(x, y, color, linewidth=10)

    def move(self, game, player_number, coup):
        player_number = player_number % 2
        if player_number == 0:
            color = 'r'
        else:
            color = 'g'
        a = plt.gca()
        if coup[2] == -1:
            for i, point in enumerate(a.collections):
                if np.all(point._offsets == np.array([
                    game.board_state.player[player_number].position]) +
                          np.array([[0.5, 0.5]])):
                    a.collections.pop(i)
            plt.pause(0.2)
            plt.scatter([coup[0]+0.5], [coup[1]+0.5], c=color, s=80)

        else:
            print("ce n'est pas un mouvement !!!")
            return

    def play(self, game, player_number, coup):
        if coup[2] == -1:
            self.move(game, player_number, coup)
        else:
            self.add_wall(player_number, coup)
        plt.pause(0.1)

    def load_board(self, universal_board):
        return
# ginput


if __name__ == '__main__':
    from Quoridor2 import Game
    game = Game('Partie 1')
    plotter = Plotter()
    plt.pause(0.5)
    plotter.add_wall(1, [0,0,1])
    plt.pause(0.5)
    plotter.move(game, 0, [4,0,-1])
    plt.pause(0.5)
    plotter.move(game, 0, [4,1,-1])
    plt.pause(0.5)
