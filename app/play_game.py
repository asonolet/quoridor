import time

import numpy as np

from quoridor.game import Game
from quoridor import policy as po

if __name__ == "__main__":
    game = Game("partie 1")
    debut = time.time()
    i = 0
    # print(game.evaluate_all_possibilities(0))
    while game.board_state.winner == -1:
        a = np.random.uniform(0, 1)
        if i % 2 == 0:
            coup = po.play_greedy(game, i % 2)
        else:
            coup = po.play_with_proba(game, i % 2)
        score = game.coup(coup, i % 2, False, True)
        print("Player %1d " % (i % 2), coup, score)
        i = i + 1
    print("And the winner is ... Player %.1d" % game.board_state.winner)
    fin = time.time()
    temps = fin - debut
    print("Temps moyen par coup : ", temps / i)
    print("Nombre de coup total :", i)
