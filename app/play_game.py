import time

import numpy as np
from quoridor import policy as po
from quoridor.game import Game
from quoridor.terminal_plotter import TermPlotter

if __name__ == "__main__":
    # rng = np.random.default_rng(654654)
    pt = TermPlotter()
    game = Game("partie 1")
    bs = game.board_state
    print("Player 0 pos, ", bs.player.position)
    print("Player 1 pos, ", bs.players[bs.last_player_nb].position)
    pt.plot(bs)
    debut = time.time()
    i = 0
    # print(game.evaluate_all_possibilities(0))
    while bs.winner == -1:
        coup = (
            po.play_seeing_future_rec(game)
            if i % 2 == 0
            else po.play_with_proba(game, rng=None)
        )
        game.play(tuple(coup))
        score = game.score()
        print(
            "Player %1d " % (i % 2),
            coup,
            score,
            "position",
            bs.last_player.position,
        )
        pt.plot(bs)
        # time.sleep(0.35)
        i = i + 1
    print("And the winner is ... Player %.1d" % bs.winner)
    fin = time.time()
    temps = fin - debut
    print("Temps moyen par coup : ", temps / i)
    print("Nombre de coup total :", i)
