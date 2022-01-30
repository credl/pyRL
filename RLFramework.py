import random
from collections import deque
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class RLEnvironment:
    def get_state_dim(self): return 0                               # number of state dimensions
    def get_action_dim(self): return 0                              # number of actions
    def next(self, action): return ([], 0)                          # get successor state and reward after performing action in current state; returns (get_state() after action, reward)
    def get_state(self): return []                                  # get current state; returns current state of environment
    def randomize_state(self): return self.encode_state()           # randomize state; returns get_state() after randomization
    def visualize(self, rlf): return                                # display current state (e.g. GUI or text output)
    def cont(self): return True                                     # callback to allow for aborting training

class RLTrainer:
    dqn_q = None            # deep q network
    dqn_t = None            # deep q target network
    nn_stats = None         # model evaluation
    loss_fn = None          # nn loss function
    env = None              # environment
    additional_stats = ""   # statistics output other than provided by tf

    # settings
    #    nn learning
    SETTING_nn_epochs: int = 1,
    #    q learning
    SETTING_sample_size: int = 32,
    SETTING_training_interval: int = 1,
    SETTING_replay_buffer_size: int = 2000,
    SETTING_alpha_q_learning_rate: float = 0.1,
    SETTING_gamma_discout_factor: float = 0.7,
    SETTING_accept_q_network_interval: int = 1,
    #    state space exploration
    SETTING_exploration_rate_start: float = 0.7,
    SETTING_exploration_rate_decrease: float = 0.0001,
    SETTING_exploration_rate_min: float = 0.1,
    SETTING_random_state_change_probability_start: float = 0.0,
    SETTING_random_state_change_probability_decrease: float = 0.0,
    SETTING_random_state_change_probability_min: float = 0.0,
    SETTING_random_state_change_probability_decrease: float = 0.0,
    # other
    SETTING_visualize_interval: int = 1

    def __init__(self, env,
                    nn = None,
                    # nn learning
                    nn_learning_rate: float = 0.01,
                    nn_epochs: int = 1,
                    # q learning
                    sample_size: int = 32,
                    training_interval: int = 1,
                    replay_buffer_size: int = 2000,
                    alpha_q_learning_rate: float = 0.1,
                    gamma_discout_factor: float = 0.7,
                    accept_q_network_interval: int = 1,
                    # state space exploration
                    exploration_rate_start: float = 0.7,
                    exploration_rate_decrease: float = 0.0001,
                    exploration_rate_min: float = 0.1,
                    random_state_change_probability_start: float = 0.0,
                    random_state_change_probability_decrease: float = 0.0,
                    random_state_change_probability_min: float = 0.0,
                    # other
                    visualize_interval: int = 1
                    ):
        self.env = env

        # save settings
        self.SETTING_nn_learning_rate = nn_learning_rate
        self.SETTING_nn_epochs = nn_epochs
        self.SETTING_sample_size = sample_size
        self.SETTING_training_interval = training_interval
        self.SETTING_replay_buffer_size = replay_buffer_size
        self.SETTING_alpha_q_learning_rate = alpha_q_learning_rate
        self.SETTING_gamma_discout_factor = gamma_discout_factor
        self.SETTING_accept_q_network_interval = accept_q_network_interval
        self.SETTING_exploration_rate_start = exploration_rate_start
        self.SETTING_exploration_rate_decrease = exploration_rate_decrease
        self.SETTING_exploration_rate_min = exploration_rate_min
        self.SETTING_random_state_change_probability_start = random_state_change_probability_start
        self.SETTING_random_state_change_probability_decrease = random_state_change_probability_decrease
        self.SETTING_random_state_change_probability_min = random_state_change_probability_min
        self.SETTING_visualize_interval = visualize_interval

        # construct q network
        if nn == None: # use default network if none provided by caller
            self.dqn_q = keras.models.Sequential([
                keras.layers.Dense(32, activation="elu", input_shape=(env.get_state_dim(),), kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(32, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
            ])
        else:
            self.dqn_q = nn
        self.loss_fn = keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.SETTING_nn_learning_rate)
        self.dqn_q.compile(self.opt, loss=self.loss_fn)

        # target network is a copy of the q network
        self.dqn_t = tf.keras.models.clone_model(self.dqn_q)

    def get_action(self, state):
        return np.argmax(self.dqn_q(tf.constant([state]))[0].numpy())

    def train(self, episodes: int = -1):
        # training initialization
        self.exploration_rate = self.SETTING_exploration_rate_start
        self.random_state_change_probability = self.SETTING_random_state_change_probability_start
        replay_buffer = deque(maxlen=self.SETTING_replay_buffer_size)
        state = self.env.get_state()

        # loop training episodes
        step = 0
        while not step == episodes:
            # more state space exploration
            self.__state_randomization()
            # choose and apply action to current state
            action = self.__choose_action(state)
            (succ_state, reward) = self.env.next(action)
            # store current observation in replay buffer and do training
            replay_buffer.append([state, action, succ_state, reward])
            if step % self.SETTING_training_interval == 0: self.__train_network(replay_buffer)
            if step % self.SETTING_accept_q_network_interval == 0: self.dqn_t.set_weights(self.dqn_q.get_weights())
            # stats update and visualization
            q_values = self.dqn_q(tf.constant([state]))[0].numpy()
            self.additional_stats += "- Steps simulated: " + str(step) + "\n" + "- Q values: " + str(self.__format_list(q_values, precision=3, precomma=5)) + "\n" + "- Best action: " + str(np.argmax(q_values))
            if step % self.SETTING_visualize_interval == 0: self.env.visualize(self)
            # prepare next iteration with the possibility for aborting
            state = succ_state; step += 1; self.additional_stats = ""
            if not self.env.cont(): step = periods

    def get_stats(self):
        return "Statistics:\n" + "- Loss: " + str(self.__format_float(self.nn_stats.history['loss'][0], precision=5)) + "\n" + "Other:\n" + self.additional_stats
        
    def get_network_stats(self):
        s = "Layer shapes:"
        for l in self.dqn_q.layers:
            s += " " + str(l.output_shape)
        return s

    def __state_randomization(self):
        # state randomization with some probability that decreases over time
        if np.random.rand() < self.random_state_change_probability:
            state = self.env.get_random_state()
        if self.random_state_change_probability > 0:
            self.random_state_change_probability -= self.random_state_change_probability_decrease
            if self.random_state_change_probability < random_state_change_probability_min:
                self.random_state_change_probability = random_state_change_probability_min

    def __choose_action(self, state):
        # estimate q values based on current state
        q_values = self.dqn_q(tf.constant([state]))[0].numpy()
        # choose action (possibly by random with some probability that decreases over time)
        if np.random.rand() < self.exploration_rate: action = np.random.choice(self.env.get_action_dim())
        else: action = np.argmax(q_values)
        self.exploration_rate -= self.SETTING_exploration_rate_decrease
        if self.exploration_rate < self.SETTING_exploration_rate_min:
            self.exploration_rate = self.SETTING_exploration_rate_min
        return action

    def __train_network(self, replay_buffer):
        # draw random training set from replay buffer
        (all_states, all_action_masks, all_succ_states, all_rewards) = self.__draw_random_sample(replay_buffer, self.SETTING_sample_size)
        # compute updated q values for the selected training samples (Bellman Equation)
        all_states_q_values = self.dqn_q(all_states)                                                                                                            # get q values (current network output) for whole training set
        max_q_values_succ_states = tf.reduce_max(self.dqn_t(all_succ_states), axis=1)                                                                           # get maximum q values for all successor states
        all_rewards += self.SETTING_gamma_discout_factor * max_q_values_succ_states                                                                             # add discounted future rewards
        all_rewards_matrix = all_rewards.numpy().repeat(self.env.get_action_dim()).reshape(-1, self.env.get_action_dim())                                       # transform into matrix form by adding a separate copy of the reward column vector for each action
        all_states_updated_q_values = all_states_q_values + self.SETTING_alpha_q_learning_rate * (all_rewards_matrix - all_states_q_values) * all_action_masks  # update q values *only* for chosen actions using the action mask
        # train network
        self.nn_stats = self.dqn_q.fit(all_states, tf.constant(all_states_updated_q_values), epochs=self.SETTING_nn_epochs, verbose=0)                  

    def __draw_random_sample(self, replay_buffer, sample_size):
        trainingset_indexes = random.sample(range(len(replay_buffer)), min(sample_size, len(replay_buffer)))
        # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
        all_states          = tf.constant([ replay_buffer[sample_idx][0] for sample_idx in trainingset_indexes ])
        all_action_masks    = tf.constant([ tf.one_hot(replay_buffer[sample_idx][1], self.env.get_action_dim()).numpy() for sample_idx in trainingset_indexes ])
        all_succ_states     = tf.constant([ replay_buffer[sample_idx][2] for sample_idx in trainingset_indexes ])
        all_rewards         = tf.constant([ replay_buffer[sample_idx][3] for sample_idx in trainingset_indexes ], dtype=tf.float32)
        return (all_states, all_action_masks, all_succ_states, all_rewards)

    def __format_float(self, number: float, precision: int = 3, precomma: int = -1):
        if precomma == -1: return ("{:." + str(precision) + "f}").format(number)
        else: return ("{:" + str(precomma + precision + 1) + "." + str(precision) + "f}").format(number)

    def __format_list(self, li: list, precision: int = 3, precomma: int = -1):
        nl = "["
        for f in li:
            nl += " " + self.__format_float(f, precision, precomma)
        nl += " ]"
        return nl