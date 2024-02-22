from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quoridor.board_state import BoardState


class TermPlotter:
    """Terminal plotter.

    The ``TermPlotter`` object is used to plot a game on the
    terminal. It can only work if it plots the game from the begining,
    it doesn't have yet any feature to plot a game from any state it
    is in.
    """

    wall_line = "+   " * 9 + "+"
    pos_line = "|   " + "    " * 8 + "|"
    init_wall = "+---" * 9 + "+"

    def __init__(self) -> None:
        pass

    def plot(self, bs: BoardState) -> None:
        """Plot a board state."""
        lines = []
        lines.append(self.init_wall)
        for line in range(8):
            lines.append(self._fill_position_line(line, bs))
            lines.append(self.wall_line)
        lines.append(self._fill_position_line(8, bs))
        lines.append(self.init_wall)
        lines = self._fill_walls(lines, bs.walls)
        print("\n".join(lines))

    def _fill_position_line(self, line: int, bs: BoardState):
        pos = list(self.pos_line)
        if line == 8 - bs.player.position[1]:
            pos[4 * bs.player.position[0] + 2] = str(bs.next_player_nb)
        if line == 8 - bs.last_player.position[1]:
            pos[4 * bs.last_player.position[0] + 2] = str(bs.last_player_nb)
        return "".join(pos)

    def _fill_walls(self, lines: list, walls: set[tuple[int, int, int]]):
        lines = [list(line) for line in lines]
        for wall in walls:
            if wall[2] == 0:
                y = (8 - wall[1]) * 2
                x = wall[0] * 4 + 1
                lines[y][x : x + 3] = "---"
                lines[y][x + 4 : x + 7] = "---"
            if wall[2] == 1:
                y = (8 - wall[1]) * 2 + 1
                x = (wall[0] + 1) * 4
                lines[y][x] = "|"
                lines[y - 2][x] = "|"
        return ["".join(line) for line in lines]
