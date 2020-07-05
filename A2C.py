import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import logging
from graphic_quoridor import Plotter


class ActionSpace:
    def __init__(self, wall_possibilities):
        self.n = len(wall_possibilities) + 4


class Env:
    def __init__(self, g, adverse_policy):
        self.G = g
        self.g = self.G('')
        self.adverser = adverse_policy
        self.previous_reward = 0
        self.observation_space = self.obs()
        self.action_space = ActionSpace(self.g.all_walls_choices)

    def reset(self):
        self.g = self.G('')
        self.previous_reward = 0
        return self.obs()

    def obs(self):
        return self.g.board_state.to_universal_state(0), self.g.moves_allowed(0)

    def step(self, categorical_action_, plotter=None):
        # le + 128 est un hack pour jouer seulement des mouvements
        categorical_move_ = categorical_action_ + 128

        action_, is_correct = self._from_categorical_to_action_no_check(
            categorical_move_)
        if plotter is not None:
            print(action_, is_correct)
        if is_correct:
            if plotter is not None:
                plotter.play(self.g, 0, action_)
            self.g.coup(action_, 0)
            coup_adv = self.adverser(self.g, 1)
            if plotter is not None:
                plotter.play(self.g, 1, coup_adv)
            self.g.coup(coup_adv, 1)
            reward = self.g.board_state.score_with_relative_path_length_dif(
                0) - \
                     self.previous_reward
            self.previous_reward = reward
            return self.obs(), \
                   reward, \
                   self.g.board_state.winner != -1, \
                   plotter
        else:
            return None, -10, 1, plotter

    def _from_categorical_to_action_no_check(self, categorical_action):
        ind = int(categorical_action)
        # si le coup que veux jouer le réseau correspond à un mur :
        if ind < self.action_space.n - 4:
            action_ = self.g.all_walls_choices[ind, :]
        else:
            # Si le réseau veut jouer un mouvement :
            all_moves = self.g._all_moves(0)
            pos = self.g.board_state.player[0].position
            # Ordre : +/-/g/d
            action_ = None
            if ind - self.action_space.n + 4 == 0:
                for move in all_moves:
                    if (move[0] == pos[0]) & (move[1] > pos[1]):
                        action_ = move
            elif ind - self.action_space.n + 4 == 1:
                for move in all_moves:
                    if (move[0] == pos[0]) & (move[1] < pos[1]):
                        action_ = move
            elif ind - self.action_space.n + 4 == 2:
                for move in all_moves:
                    if (move[1] == pos[1]) & (move[0] < pos[0]):
                        action_ = move
            elif ind - self.action_space.n + 4 == 3:
                for move in all_moves:
                    if (move[1] == pos[1]) & (move[0] > pos[0]):
                        action_ = move
        if action_ is None:
            Raise()
        return action_, True

    def _from_categorical_to_action(self, categorical_action):
        ind = int(categorical_action)
        # si le couop que veux jouer le résezau correspond à un mur :
        if ind < self.action_space.n - 4:
            # S'il lui reste encore des murs à jouer :
            if self.g.board_state.player[0].n_tuiles > 0:
                action_ = self.g.all_walls_choices[ind, :]
                scalar_action = 100*action_[0] + 10*action_[1] + action_[2]
                all_walls_available = np.transpose(np.nonzero(
                    self.g.board_state.wall_possibilities > 0))
                test_set = 100 * all_walls_available[:, 0] + \
                           10 * all_walls_available[:, 1] + \
                           all_walls_available[:, 2]
                is_correct = np.isin(scalar_action, test_set, assume_unique=True)
            else:
                action_ = None
                is_correct = False
        else:
            # Si le réseau veut jouer un mouvement :
            all_moves = self.g._all_moves(0)
            pos = self.g.board_state.player[0].position
            # Ordre : +/-/g/d
            is_correct = False
            action_ = None
            if ind - self.action_space.n + 4 == 0:
                for move in all_moves:
                    if (move[0] == pos[0]) & (move[1] > pos[1]):
                        action_ = move
                        is_correct = True
            elif ind - self.action_space.n + 4 == 1:
                for move in all_moves:
                    if (move[0] == pos[0]) & (move[1] < pos[1]):
                        action_ = move
                        is_correct = True
            elif ind - self.action_space.n + 4 == 2:
                for move in all_moves:
                    if (move[1] == pos[1]) & (move[0] < pos[0]):
                        action_ = move
                        is_correct = True
            elif ind - self.action_space.n + 4 == 3:
                for move in all_moves:
                    if (move[1] == pos[1]) & (move[0] > pos[0]):
                        action_ = move
                        is_correct = True
        return action_, is_correct


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        # c'est peut etre ici qu'il faut corriger les proba pour mettre à 0
        # toutes celles qui correspondent à des coups interdits. A mois qu'on
        # ne mette des scores horribles aux coups interdits joués. Je vais
        # plutôt faire ça.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # Note: no tf.get_variable(), just simple Keras API!
        self.conv1 = kl.Conv2D(10, (3, 3), activation='relu')
        self.conv2 = kl.Conv2D(10, (3, 3), activation='relu')
        self.conv3 = kl.Conv2D(10, (4, 4), activation='relu')
        self.flatten = kl.Flatten()
        self.hidden_other = kl.Dense(6, activation='relu', name='hidden_other')
        # self.hidden1 = kl.Dense(128, activation='relu')
        # self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        # self.wall_choice = kl.Dense(num_actions - 4, name='policy_walls_logits')
        self.move_choice = kl.Dense(4, name='policy_move_logits')
        self.dist = ProbabilityDistribution()
        self.num_actions = num_actions

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = inputs[0]
        poss = inputs[1]
        x = tf.convert_to_tensor(x)
        poss = tf.convert_to_tensor(poss)
        # Seperate the wall part from the other infos
        walls = x[..., :-6]
        other = x[..., -6:]
        walls = tf.reshape(walls, [-1, 8, 8, 2])
        # Separate hidden layers from the same input tensor.
        walls = self.conv1(walls)
        walls = self.conv2(walls)
        walls = self.conv3(walls)
        walls = self.flatten(walls)
        other = self.hidden_other(other)
        hidden = tf.concat((walls, other), axis=-1)
        # hidden_vals = self.hidden2(x)
        move_choice = self.move_choice(hidden)
        # move_choice_allowed = np.where(poss, move_choice, np.inf)
        move_choice_allowed = tf.where(poss > 0., move_choice, - np.inf *
                                       tf.ones_like(move_choice))
        return move_choice_allowed, self.value(
            hidden), move_choice, hidden
        # self.logits(hidden),

    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value, _, __ = self.predict_on_batch(obs)  #
        action = self.dist.predict_on_batch(logits)
        if np.all(action == 4):
            raise Exception
        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)  # , \
               # np.squeeze(naive_logits, axis=-1)


class A2CAgent:
    def __init__(self, model, lr=7e-4, gamma=0.2, value_c=0.5,
                 entropy_c=1e-4, reprise=False):
        # `gamma` is the discount factor
        self.gamma = gamma
        # Coefficients are used for the loss terms.
        self.value_c = value_c
        self.entropy_c = entropy_c

        self.model = model
        if not reprise:
            self.model.compile(
                optimizer=ko.RMSprop(lr=lr),
                # Define separate losses for policy logits and value estimate.
                loss=[self._logits_loss, self._value_loss, self._rules_loss])

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), 0, 0
        if render:
            plotter = Plotter()
        else:
            plotter = None

        while not done:
            action, _ = self.model.action_value((obs[0][None, :], obs[1][
                None, :]))

            allowed_choices_proba, _, choices_proba, __ = self.model((obs[0][
                                                                   None, :],
                                                      obs[1][None, :]))  # , \
                                       #
            obs, reward, done, plotter = env.step(action, plotter)
            # reward = g.coups(action)
            #
            ep_reward += reward
            # if render:
            #     env.render()
        print(self.model.summary())
        return ep_reward

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _rules_loss(self, allowed_choices, naive_logits):
        return kls.mean_squared_error(allowed_choices, naive_logits)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on
        # `call()`.
        # `from_logits` argument ensures transformation into normalized
        # probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits,
                                         sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss

    # @tf.function
    # a priori ca ne marche pas a cause des arguments de sortie multiple mais
    # pas sûr
    def train(self, env, batch_sz=20, updates=1000):
        n = self.model.num_actions
        # TRICHE :
        n = 4
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        allowed_choices_proba = np.empty((batch_sz, n), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations1 = np.empty((batch_sz,) + env.observation_space[0].shape)
        observations2 = np.empty((batch_sz,) + env.observation_space[1].shape)

        # Training loop: collect samples, send to optimizer, repeat updates
        # times.
        ep_rewards = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations1[step] = next_obs[0].copy()
                observations2[step] = next_obs[1].copy()
                actions[step], values[step] = self.model.action_value(
                    (next_obs[0][None, :], next_obs[1][None, :]))
                allowed_choices_proba[step, :], _, _, __ = self.model((next_obs[0][
                                                                None, :],
                                                                   next_obs[1][
                                                                None, :]))
                next_obs, rewards[step], dones[step], _ = env.step(
                    actions[step])
                # mettre en place une boucle qui rapelle un step
                # d'entrainement tant que le réseau ne donne pas une action
                # autorisée. Pour le moment on met juste un score pourri et
                # on termine le jeu. Ca va peut être mener à un réseau qui
                # met beaucoup de temps a apprendre les règles.

                ep_rewards[-1] += rewards[step]
                if dones[step]:  # initialement juste
                # dones[step]!= -1
                    ep_rewards.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (
                        len(ep_rewards) - 1, ep_rewards[-2]))

            _, next_value = self.model.action_value((next_obs[0][None, :],
                                                     next_obs[1][None, :]))

            returns, advs = self._returns_advantages(rewards, dones, values,
                                                     next_value)
            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate(
                [actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch((observations1, observations2),
                                               [acts_and_advs, returns,
                                                allowed_choices_proba]) #, \
                     #

            logging.debug(
                "[%d/%d] Losses: %s" % (update + 1, updates, losses))

            print(update)
            if update % 40 == 0:
                self.test(env, render=True)
                # self.model.save('chkpt_'+str(update//40)+'.nn')

        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (
        # critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (
                    1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our
        # case).
        advantages = returns - values

        return returns, advantages


if __name__ == '__main__':
    from Quoridor2 import Game, play_with_proba

    env = Env(Game, play_with_proba)
    model = Model(num_actions=env.action_space.n)

    obs = env.reset()

    agent = A2CAgent(model)
    rewards_sum = agent.test(env, render=True)
    print("%d out of 200" % rewards_sum)  

    rewards_history = agent.train(env, updates=10000)
    print("Finished training, testing...")
    print("%d out of 200" % agent.test(env))  
    # testing version 1 of the agent
    # agent = A2CAgent(tf.keras.models.load_model('chkpt_1.nn'), reprise=True)
    # env.reset()
    # agent.test(env)
