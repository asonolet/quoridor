import numpy as np


def play_greedy(game_, player_number):
    return game_.evaluate_all_possibilities(player_number)[0][-1]


def play_random(game_, player_number):
    res = game_.evaluate_all_possibilities(player_number)[0]
    return res[np.random.randint(len(res))]


def play_with_proba(game_, player_number):
    res, cost = game_.evaluate_all_possibilities(player_number)
    maxi = np.max(cost)
    cost = cost + 1 - maxi
    cost_ = np.where(cost > -500, np.exp(100 * cost), 0)
    cost_ = cost_ / np.sum(cost_)
    return res[np.random.choice(len(cost), p=cost_)]


def play_seeing_future_rec(
    game_, player_number, n_sim=3, n_future=3, counter=0, returned_scores=None
):
    if game_.board_state.winner != -1:
        if player_number == game_.board_state.winner:
            return 1000
        else:
            return -1000
    else:
        if 2 * n_future == counter:
            choices, scores = game_.evaluate_all_possibilities(player_number)
            return [scores[-1]]
        else:
            choices, _ = game_.evaluate_all_possibilities((player_number + counter) % 2)
            n_sim_possible = min(n_sim, len(choices))
            choices = choices[-n_sim_possible:]
            if returned_scores is None:
                returned_scores = []
            for i_sim in range(n_sim_possible):
                game_.coup(choices[i_sim], (player_number + counter) % 2, score_=False)
                scores = play_seeing_future_rec(
                    game_, player_number, n_sim, n_future, counter + 1
                )
                game_.get_back(1)
                if counter % 2 == 0:
                    # returned_scores.append(np.mean(scores))
                    returned_scores.append(np.max(scores))
                else:
                    returned_scores.append(np.max(scores))
            if counter == 0:
                return choices[np.argmax(returned_scores)], np.max(returned_scores)
            else:
                return returned_scores