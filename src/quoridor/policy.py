"""Different functions to evaluate the coup to play."""

import numpy as np

RNG = np.random.default_rng()
LOWEST_SCORE = -1000
HIGHEST_SCORE = 1000


def play_greedy(game):
    """Evaluate all possibilities and choose best one."""
    choices, scores = game.evaluate_all_possibilities()
    return tuple(choices[-1])


def play_random(game, rng=None):
    """Evaluate all possibilities and choose any of them."""
    res = game.evaluate_all_possibilities()[0]
    if rng is None:
        rng = RNG
    return int(res[rng.randint(len(res))])


def play_with_proba(game, rng=None):
    """Evaluate all possibilities and choose one of them based on a exp formula."""
    res, cost = game.evaluate_all_possibilities()
    maxi = np.max(cost)
    cost = cost + 1 - maxi
    cost_ = np.where(cost > LOWEST_SCORE / 2, np.exp(100 * cost), 0)
    cost_ = cost_ / np.sum(cost_)
    if rng is None:
        rng = RNG
    return tuple(res[rng.choice(len(cost), p=cost_)])


def _best_move_seeing_future(game, n_future, n_sim, counter):
    # print("  " * counter + f"{counter=}")
    choices, scores = game.evaluate_all_possibilities()
    if counter >= n_future:
        return tuple(choices[-1]), scores[-1]

    worse_case_scores = []
    for move in choices[-n_sim:]:
        # print("  " * counter + f"Testing {move=}")
        game.play(tuple(move))
        adv_choices, _ = game.evaluate_all_possibilities()
        scores = []
        for adv_choice in adv_choices[-1:]:
            # print("  " * counter + f"  Testing {adv_choice=}")
            game.play(tuple(adv_choice))
            _, score = _best_move_seeing_future(
                game,
                n_future=n_future,
                n_sim=1,
                counter=counter + 1,
            )
            game.get_back()
            scores.append(score)
        game.get_back()
        worse_case_scores.append(min(scores))
    return (
        tuple(choices[-n_sim:][np.argmax(worse_case_scores)]),
        np.max(worse_case_scores),
    )


def play_seeing_future_rec(game, n_future=2, n_sim=4):
    """Evaluate all possibilities and simulate the score after playing n coups."""
    coup, _ = _best_move_seeing_future(game, n_future=n_future, n_sim=n_sim, counter=0)
    return coup
