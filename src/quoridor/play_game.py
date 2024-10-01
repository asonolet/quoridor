"""Define the command line quoridor application."""

import time

import click
import numpy as np

from quoridor import policy as po
from quoridor.game import Game
from quoridor.terminal_plotter import TermPlotter


def play_fn(*, pause=0.0, verbose=True):
    """Simulate a game with two players."""
    # rng = np.random.default_rng(654654)
    game = Game("partie 1")
    sleep_time = pause
    bs = game.board_state
    if verbose:
        pt = TermPlotter()
        print("Player 0 pos, ", bs.player.position)
        print(
            "Player 1 pos, ",
            bs.players[bs.last_player_nb].position,
        )
        pt.plot(bs)

    debut = time.time()
    i = 0
    while bs.winner == -1:
        coup = (
            po.play_with_proba(game)
            if i % 2 == 0
            else po.play_with_proba(game, rng=None)
        )
        game.play(tuple(coup))
        score = game.score()
        if verbose:
            print(
                "Player %1d " % (i % 2),
                coup,
                score,
            )
            print(
                "position",
                bs.last_player.position,
            )
            pt.plot(bs)

        time.sleep(sleep_time)
        i = i + 1
    fin = time.time()
    temps = fin - debut
    if verbose:
        print("And the winner is ... Player %.1d" % bs.winner)
        print("Temps moyen par coup : ", temps / i - sleep_time)
        print("Nombre de coup total :", i)
    return bs.winner, i


@click.group()
def cli():
    """Launch the main application for the quoridor library."""


@click.command()
@click.option("-p", "--pause", default=0.0)
def play(pause):
    """Simulate a game with two players."""
    play_fn(pause=pause, verbose=True)


@click.command()
@click.option("-n", default=5)
def stat(n: int):
    """Play n games and print the statistics."""
    winner_count = [0, 0]
    player0_coups = []
    player1_coups = []
    debut = time.time()
    for _ in range(n):
        w, c = play_fn(pause=0.0, verbose=False)
        winner_count[w] += 1
        if w == 1:
            player1_coups.append(c)
        else:
            player0_coups.append(c)
    fin = time.time()
    print(f"Player0 won {winner_count[0]} / {n} times")
    print(f"    using {np.mean(player0_coups)} coup in mean")
    print(f"Player1 won {winner_count[1]} / {n} times")
    print(f"    using {np.mean(player1_coups)} coup in mean")
    print(f"Time per game is {(fin-debut)/n}s")


cli.add_command(play)
cli.add_command(stat)


if __name__ == "__main__":
    cli()
