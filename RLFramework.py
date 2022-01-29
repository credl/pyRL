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
    stats = None            # model evaluation
    loss_fn = None          # nn loss function
    env = None              # environment
    additional_stats = ""   # statistics output other than provided by tf

    def __init__(self, env, nn = None, nn_learning_rate: float = 0.01):
        self.env = env

        # construct q network
        if nn == None:
            self.dqn_q = keras.models.Sequential([
                keras.layers.Dense(32, activation="elu", input_shape=(env.get_state_dim(),), kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(32, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
            ])
        else:
            self.dqn_q = nn
        self.loss_fn = keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)
        self.dqn_q.compile(self.opt, loss=self.loss_fn)

        # target network is a copy of the q network
        self.dqn_t = tf.keras.models.clone_model(self.dqn_q)

    def get_action(self, state):
        q_values = self.dqn_q(tf.constant([state]))[0].numpy()
        return np.argmax(q_values)

    def train(self,
                # nn learning
                nn_epochs: int = 1,
                # q learning
                sample_size: int = 32,
                training_interval: int = 1,
                periods: int = -1,
                replay_buffer_size: int = 2000,
                alpha_q_learning_rate: float = 0.1,
                gamma_discout_factor: float = 0.7,
                accept_q_network_interval: int = 1,
                # state space exploration
                exploration_rate_start: float = 0.7,
                exploration_rate_decrease: float = 0.0001,
                exploration_rate_min: float = 0.1,
                random_state_change_probability: float = 0.0,
                random_state_change_probability_decrease: float = 0.0,
                # other
                visualize_interval: int = 1):

        # initialization
        exploration_rate = exploration_rate_start
        replay_buffer = deque(maxlen=replay_buffer_size)
        state = self.env.get_state()

        # loop training periods
        step = 0
        while not step == periods:
            # state randomization
            if np.random.rand() < random_state_change_probability:
                state = self.env.get_random_state()
            if random_state_change_probability > 0:
                random_state_change_probability -= random_state_change_probability_decrease
                if random_state_change_probability < 0:
                    random_state_change_probability = 0

            # estimate q values based on current state
            q_values = self.dqn_q(tf.constant([state]))[0].numpy()

            # choose action (by random with some probability that decreases over time)
            if np.random.rand() < exploration_rate: action = np.random.choice(self.env.get_action_dim())
            else: action = np.argmax(q_values)
            exploration_rate -= exploration_rate_decrease
            if exploration_rate < exploration_rate_min:
                exploration_rate = exploration_rate_min

            # apply action to current state
            (succ_state, reward) = self.env.next(action)
            
            # store current observation in replay buffer
            observation = [state, action, succ_state, reward]
            replay_buffer.append(observation)

            # training
            if step % training_interval == 0:
                # draw random sample from replay buffer
                trainingset = random.sample(replay_buffer, min(len(replay_buffer), sample_size))
                
                # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
                inp = []; out = []
                for (ts_state, ts_action, ts_succ_state, ts_reward) in trainingset:
                    # predict q-values of current and successor state by NNs
                    ts_current_q_values = self.dqn_q(tf.constant([ts_state]))[0].numpy()
                    ts_succ_q_values = self.dqn_t(tf.constant([ts_succ_state]))[0].numpy()

                    # update q-value of chosen action (Bellman equation)
                    ts_current_q_values[ts_action] = ts_current_q_values[ts_action] + alpha_q_learning_rate * (ts_reward + gamma_discout_factor * max(ts_succ_q_values) - ts_current_q_values[ts_action])

                    # build training batch
                    inp.append(ts_state); out.append(ts_current_q_values)

                # train on all instances
                self.stats = self.dqn_q.fit(tf.constant(inp), tf.constant(out), epochs=nn_epochs, verbose=0)

            # stats update
            self.additional_stats += "- Steps simulated: " + str(step) + "\n" + "- Q values: " + str(self.__format_list(q_values, precision=3, precomma=5)) + "\n" + "- Best action: " + str(np.argmax(q_values))

            # accept q network as new target
            if step % accept_q_network_interval == 0: self.dqn_t.set_weights(self.dqn_q.get_weights())

            # visualize
            if step % visualize_interval == 0: self.env.visualize(self)

            # prepare next iteration
            state = succ_state
            step += 1
            self.additional_stats = ""
            
            # possibility for aborting
            if not self.env.cont():
                step = periods

    def get_stats(self):
        return "Statistics:\n" + "- Loss: " + str(self.__format_float(self.stats.history['loss'][0], precision=5)) + "\n" + "Other:\n" + self.additional_stats
        
    def get_network_stats(self):
        s = "Layer shapes:"
        for l in self.dqn_q.layers:
            s += " " + str(l.output_shape)
        return s

    def __format_float(self, number: float, precision: int = 3, precomma: int = -1):
        if precomma == -1: return ("{:." + str(precision) + "f}").format(number)
        else: return ("{:" + str(precomma + precision + 1) + "." + str(precision) + "f}").format(number)

    def __format_list(self, li: list, precision: int = 3, precomma: int = -1):
        nl = "["
        for f in li:
            nl += " " + self.__format_float(f, precision, precomma)
        nl += " ]"
        return nl
