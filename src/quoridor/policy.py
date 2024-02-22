"""Different functions to evaluate the coup to play."""

import numpy as np

RNG = np.random.default_rng()
LOWEST_SCORE = -1000
HIGHEST_SCORE = 1000


def play_greedy(game):
    """Evaluate all possibilities and choose best one."""
    return tuple(game.evaluate_all_possibilities()[0][-1])


def play_random(game):
    """Evaluate all possibilities and choose any of them."""
    res = game.evaluate_all_possibilities()[0]
    return int(res[RNG.randint(len(res))])


def play_with_proba(game):
    """Evaluate all possibilities and choose one of them based on a exp formula."""
    res, cost = game.evaluate_all_possibilities()
    maxi = np.max(cost)
    cost = cost + 1 - maxi
    cost_ = np.where(cost > LOWEST_SCORE / 2, np.exp(100 * cost), 0)
    cost_ = cost_ / np.sum(cost_)
    return tuple(res[RNG.choice(len(cost), p=cost_)])


def play_seeing_future_rec(game, n_future=3, counter=0):
    """Evaluate all possibilities and simulate the score after playing n coups."""
    if game.board_state.winner != -1:
        if game.board_state.next_player == game.board_state.winner:
            return HIGHEST_SCORE
        return LOWEST_SCORE

    if 2 * n_future == counter:
        choices, scores = game.evaluate_all_possibilities()
        return [scores[-1]]
    return None


#     choices, _ = game.evaluate_all_possibilities((player_number + counter) % 2)
#     n_sim_possible = min(n_sim, len(choices))
#     choices = choices[-n_sim_possible:]
#     if returned_scores is None:
#         returned_scores = []
#     for i_sim in range(n_sim_possible):
#         game_.coup(choices[i_sim], (player_number + counter) % 2, score_=False)
#         scores = play_seeing_future_rec(
#             game_, player_number, n_sim, n_future, counter + 1
#         )
#         game_.get_back(1)
#         if counter % 2 == 0:
#             # returned_scores.append(np.mean(scores))
#             returned_scores.append(np.max(scores))
#         else:
#             returned_scores.append(np.max(scores))
#     if counter == 0:
#         return choices[np.argmax(returned_scores)], np.max(returned_scores)
#     else:
#         return returned_scores
