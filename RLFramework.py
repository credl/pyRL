import random
from collections import deque
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class RLEnvironment:
    def get_state_dim(self): return 0                               # number of state dimensions
    def get_action_dim(self): return 0                              # number of actions
    def next(self, action): return ([], 0)                          # get successor state and reward after performing action in current state; returns (get_state() after action, reward)
    def get_state(self): return []                                  # get current state; returns current state of environment
    def randomize_state(self): return self.encode_state()           # randomize state; returns get_state() after randomization
    def visualize(self, rlf, step): return                          # display current state (e.g. GUI or text output)
    def abort(self): return False                                   # callback to allow for aborting training

class RLTrainer:
    dqn_q = None            # deep q network
    dqn_t = None            # deep q target network
    nn_stats = None         # model evaluation
    loss_fn = None          # nn loss function
    env = None              # environment
    additional_stats = ""   # statistics output other than provided by tf
    start_times = dict()    # benchmarking
    sum_times = dict()      # benchmarking
    learn = True            # enable learning

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
    # model load and save
    SETTING_save_interval = -1
    SETTING_save_path = None
    SETTING_load_path = None

    # ### Initialization ###

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
                    # model load and save
                    save_interval: int = -1,
                    save_path: str = None,
                    load_path: str = None,
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
        self.SETTING_save_interval = save_interval
        self.SETTING_save_path = save_path
        self.SETTING_load_path = save_path

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
        if not self.SETTING_load_path == None:
            self.dqn_q(tf.constant([env.get_state()]))
            try:
                self.dqn_q.load_weights(self.SETTING_load_path)
                print("Loaded model weights from file", self.SETTING_load_path)
            except:
                print("Could not load model from file", self.SETTING_load_path, "; continue with random weights")

        # target network is a copy of the q network
        self.dqn_t = tf.keras.models.clone_model(self.dqn_q)


    # ### Public Interface ###

    def get_action(self, state):
        self.__start_time("nn_query")
        ac = np.argmax(self.dqn_q(tf.constant([state]))[0].numpy())
        self.__end_time("nn_query")
        return ac

    def get_actions(self, states):
        self.__start_time("nn_query")
        ac = [ np.argmax(q_values) for q_values in self.dqn_q(tf.constant(states)).numpy()]
        self.__end_time("nn_query")
        return ac

    def train(self, episodes: int = -1):
        # training initialization
        self.exploration_rate = self.SETTING_exploration_rate_start
        self.random_state_change_probability = self.SETTING_random_state_change_probability_start
        replay_buffer = deque(maxlen=self.SETTING_replay_buffer_size)
        state = self.env.get_state()

        # loop training episodes
        step = 0; self.__start_time("q_learn_loop")
        while (not step == episodes) and (not self.env.abort()):
            self.__log_bm("Cur. expl. rate", self.__format_float(self.exploration_rate))
            self.__log_bm("DQL loop", self.__format_list(self.__end_time("q_learn_loop")))
            self.__log_bm("Steps simulated", str(step))
            self.__start_time("q_learn_loop")
            # more state space exploration
            self.__state_randomization()
            # choose and apply action to current state
            action = self.__choose_action(state)
            (succ_state, reward) = self.env.next(action)
            # store current observation in replay buffer and do training
            replay_buffer.append([state, action, succ_state, reward])
            # training
            if step % self.SETTING_training_interval == 0:
                self.__start_time("ov_ql")
                if self.learn:
                    self.__train_network(replay_buffer)
                self.__log_bm("Loss", self.__format_float(self.nn_stats.history['loss'][0], precision=5))
                self.__log_bm("Overall QL", self.__format_list(self.__end_time("ov_ql")))
                self.__log_bm("Overall QL(%Loop)", self.__format_float(self.__get_time_percentage("ov_ql", "q_learn_loop")))               
            if step % self.SETTING_accept_q_network_interval == 0: self.dqn_t.set_weights(self.dqn_q.get_weights())
            # stats update and visualization
            self.__start_time("nn_query")
            q_values = self.dqn_q(tf.constant([state]))[0].numpy()
            self.__end_time("nn_query")
            self.__log_bm("Q values", str(self.__format_list(q_values, precision=3, precomma=5)))
            self.__log_bm("Best action", str(np.argmax(q_values)))
            self.__start_time("save")
            if (not self.SETTING_save_path == None) and (step % self.SETTING_save_interval == 0): self.dqn_t.save_weights(self.SETTING_save_path, overwrite=True, save_format=None, options=None)
            self.__log_bm("Model save", self.__format_list(self.__end_time("save")))
            self.__start_time("viz")
            self.__log_bm("Visualization", self.__format_list(self.__get_time_sum("viz")))
            if step % self.SETTING_visualize_interval == 0: self.env.visualize(self, step)
            self.__end_time("viz")
            # prepare next iteration
            state = succ_state; step += 1; self.additional_stats = ""

    def get_stats(self):
        return self.additional_stats
        
    def get_network_stats(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        s = ""
        s +=  "Layer shapes: " + " ".join([str(l.output_shape) for l in self.dqn_q.layers]) + "\n"
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys(): s +=    "CUDA: " + os.environ['CUDA_VISIBLE_DEVICES']
        else:                                           s +=    "CUDA: CPU"
        return s


    # ### Training and NN Helper ###

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
        self.__start_time("nn_query")
        q_values = self.dqn_q(tf.constant([state]))[0].numpy()
        self.__end_time("nn_query")
        # choose action (possibly by random with some probability that decreases over time)
        if np.random.rand() < self.exploration_rate:    action = np.random.choice(self.env.get_action_dim())
        else:                                           action = np.argmax(q_values)
        if self.exploration_rate > self.SETTING_exploration_rate_min:
            self.exploration_rate -= self.SETTING_exploration_rate_decrease
            if self.exploration_rate < self.SETTING_exploration_rate_min:
                self.exploration_rate = self.SETTING_exploration_rate_min
        return action

    def __train_network(self, replay_buffer):
        # draw random training set from replay buffer
        (all_states, all_action_masks, all_succ_states, all_rewards) = self.__draw_random_sample(replay_buffer, self.SETTING_sample_size)
        # compute updated q values for the selected training samples (Bellman Equation)
        self.__start_time("nn_query")
        all_states_q_values = self.dqn_q(all_states)                                                                                                            # get q values (current network output) for whole training set
        max_q_values_succ_states = tf.reduce_max(self.dqn_t(all_succ_states), axis=1)                                                                           # get maximum q values for all successor states
        self.__end_time("nn_query")
        all_rewards += self.SETTING_gamma_discout_factor * max_q_values_succ_states                                                                             # add discounted future rewards
        all_rewards_matrix = all_rewards.numpy().repeat(self.env.get_action_dim()).reshape(-1, self.env.get_action_dim())                                       # transform rewards vector into matrix (add a separate copy of the vector for each action)
        all_states_updated_q_values = all_states_q_values + self.SETTING_alpha_q_learning_rate * (all_rewards_matrix - all_states_q_values) * all_action_masks  # update q values **only** for chosen actions (using the action mask)
        # train network
        self.__start_time("nn_train")
        self.nn_stats = self.dqn_q.fit(all_states, tf.constant(all_states_updated_q_values), epochs=self.SETTING_nn_epochs, verbose=0)                          # actual network training
        self.__log_bm("NN-Training", self.__format_list(self.__end_time("nn_train")))
        self.__log_bm("NN-Training (%Loop)", self.__format_float(self.__get_time_percentage("nn_train", "q_learn_loop")))
        self.__log_bm("NN-Query", self.__format_list(self.__get_time_sum("nn_query")))
        self.__log_bm("NN-Query (%Loop)", self.__format_float(self.__get_time_percentage("nn_query", "q_learn_loop")))

    def __draw_random_sample(self, replay_buffer, sample_size):
        trainingset_indexes = random.sample(range(len(replay_buffer)), min(sample_size, len(replay_buffer)))
        # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
        all_states          = tf.constant([ replay_buffer[sample_idx][0] for sample_idx in trainingset_indexes ])
        all_action_masks    = tf.constant([ tf.one_hot(replay_buffer[sample_idx][1], self.env.get_action_dim()).numpy() for sample_idx in trainingset_indexes ])
        all_succ_states     = tf.constant([ replay_buffer[sample_idx][2] for sample_idx in trainingset_indexes ])
        all_rewards         = tf.constant([ replay_buffer[sample_idx][3] for sample_idx in trainingset_indexes ], dtype=tf.float32)
        return (all_states, all_action_masks, all_succ_states, all_rewards)


    # ### Logging and Output ###

    def __format_float(self, number: float, precision: int = 3, precomma: int = -1):
        if precomma == -1:  return ("{:." + str(precision) + "f}").format(number)
        else:               return ("{:" + str(precomma + precision + 1) + "." + str(precision) + "f}").format(number)

    def __format_list(self, li: list, precision: int = 3, precomma: int = -1):
        return "[ " + " ".join(self.__format_float(f, precision, precomma) for f in li) + (" " if len(li) > 0 else "") + "]"

    def __log(self, txt: str):
        self.additional_stats += ("" if self.additional_stats == "" else "\n") + txt
        
    def __log_bm(self, key: str, val: str, padlen: int = 20):
        self.__log("- " + key.rjust(padlen, " ") + ": " + val)
        
    def __start_time(self, name: str = ""):
        self.start_times[name] = time.time()

    def __end_time(self, name: str = ""):
        end = time.time()
        elapsed = end - self.start_times[name]
        if not name in self.sum_times.keys(): self.sum_times[name] = 0
        self.sum_times[name] += elapsed
        return [elapsed, self.sum_times[name]]

    def __get_time_sum(self, name: str = ""):
        if not name in self.sum_times.keys(): self.sum_times[name] = 0
        return [0.0, self.sum_times[name]]
        
    def __get_time_percentage(self, num: str, den: str):
        num_v = self.__get_time_sum(num)[1]
        den_v = self.__get_time_sum(den)[1]
        if den_v > 0:       return num_v * 100 / den_v
        else:               return 0.0