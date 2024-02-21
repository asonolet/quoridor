import time

import numpy as np
from quoridor import policy as po
from quoridor.game import Game

if __name__ == "__main__":
    game = Game("partie 1")
    bs = game.board_state
    print("Player 0 pos, ", bs.player.position)
    print("Player 1 pos, ", bs.players[bs.last_player_nb].position)
    debut = time.time()
    i = 0
    # print(game.evaluate_all_possibilities(0))
    while bs.winner == -1:
        a = np.random.uniform(0, 1)

        coup = po.play_greedy(game) if i % 2 == 0 else po.play_with_proba(game)
        game.play(coup)
        score = game.score()
        print(
            "Player %1d " % (i % 2),
            coup,
            score,
            "position",
            bs.players[bs.last_player_nb].position,
        )
        i = i + 1
    print("And the winner is ... Player %.1d" % bs.winner)
    fin = time.time()
    temps = fin - debut
    print("Temps moyen par coup : ", temps / i)
    print("Nombre de coup total :", i)
