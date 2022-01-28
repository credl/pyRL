import os
import sys
import keyboard
import random
import threading
import math
from collections import deque
import PySimpleGUI as sg
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

class RLEnvironment:
    def get_state_dim(self):    # number of state dimensions
        return 0
    def get_action_dim(self):   # number of actions
        return 0
    def get_succ_state(self, state, action):    # get state after performing an action
        return self.next(state, action)[0]
    def get_reward(self, state, action):        # get reward if action is performed in state
        return self.next(state, action)[1]
    def next(self, state, action):              # get successor state and reward after performing action in state
        return ([], 0)
    def get_start_state(self):                  # get initial state
        return []
    def get_random_state(self):                 # randomize state
        return []
    def visualize(self, state, rlf):            # show environment (GUI or text output)
        return
    def environment_change(self, state, action):# after agent performed action, perform any changes to state other than agent action when going from one frame to the next (e.g. user input)
        return state

class RLTrainer:
    dqn_q = 0   # deep q network
    dqn_t = 0   # deep q target network
    stats = 0   # model evaluation
    loss_fn = 0 # nn loss function
    env = 0     # environment
    additional_stats = "" # statistics output other than provided by tf

    def __init__(self, env, nn_learning_rate: float = 0.01):
        self.env = env

        # construct_q_network
        self.dqn_q = keras.models.Sequential([
            keras.layers.Dense(128, activation="elu", input_shape=(env.get_state_dim(),), kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(128, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(128, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
        ])
        self.loss_fn = keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)
        self.dqn_q.compile(self.opt, loss=self.loss_fn)

        self.dqn_t = tf.keras.models.clone_model(self.dqn_q)
        self.dqn_t.compile(self.opt, loss=self.loss_fn)
        self.dqn_t.set_weights(self.dqn_q.get_weights())

    def get_action(self, state):
        q_values = self.dqn_q(tf.constant([state]))[0].numpy()
        return np.argmax(q_values)

    def train(self,
                periods: int = -1,
                nn_epochs: int = 1,
                sample_size: int = 32,
                alpha_q_learning_rate: float = 0.1,
                gamma_discout_factor: float = 0.7,
                exploration_rate_start: float = 0.7,
                exploration_rate_decrease: float = 0.0001,
                exploration_rate_min: float = 0.1,
                replay_buffer_size: int = 2000,
                accept_q_network_interval: int = 1,
                random_state_change_probability: float = 0.0,
                random_state_change_probability_decrease: float = 0.0,
                training_interval: int = 1,
                visualize_interval: int = 1):

        exploration_rate = exploration_rate_start
        replaybuffer = deque(maxlen=replay_buffer_size)
        state = self.env.get_start_state()

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

            # choose action
            epsilon = np.random.rand()
            if self.train and epsilon < exploration_rate:
                action = np.random.choice(self.env.get_action_dim())
            else:
                action = np.argmax(q_values)
            # decrease random choices over time
            if exploration_rate > exploration_rate_min:
                exploration_rate -= exploration_rate_decrease
                if exploration_rate < exploration_rate_min:
                    exploration_rate = exploration_rate_min

            # apply action
            self.additional_stats += "- Steps simulated: " + str(step) + "\n" + "- Q values: " + str(q_values) + "\n" + "- Best action: " + str(np.argmax(q_values))
            (succ_state, reward) = self.env.next(state, action)
            
            # store current observation in training set
            observation = [state, action, succ_state, reward]
            replaybuffer.append(observation)

            # training
            if step % training_interval == 0:
                # draw random sample from replay buffer
                trainingset = random.sample(replaybuffer, min(len(replaybuffer), sample_size))
                
                # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
                inp = []
                out = []
                old_q_values = []
                updated_q_values = []
                for ts_sample in trainingset:
                    ts_state = ts_sample[0]
                    ts_action = ts_sample[1]
                    ts_succ_state = ts_sample[2]
                    ts_reward = ts_sample[3]

                    # predict q-values by NN
                    ts_current_q_values = self.dqn_q(tf.constant([ts_state]))[0].numpy()
                    ts_succ_q_values = self.dqn_t(tf.constant([ts_succ_state]))[0].numpy()

                    # update q-value of chosen action (Bellman equation)
                    ts_updated_current_q_values = list(ts_current_q_values)
                    ts_updated_current_q_values[ts_action] = ts_updated_current_q_values[ts_action] + alpha_q_learning_rate * (ts_reward + gamma_discout_factor * max(ts_succ_q_values) - ts_updated_current_q_values[ts_action])

                    # build training batch
                    inp.append(ts_state)
                    out.append(ts_updated_current_q_values)

                # train on all instances
                self.stats = self.dqn_q.fit(tf.constant(inp), tf.constant(out), epochs=nn_epochs, verbose=0)

            # accept q network as new target
            if step % accept_q_network_interval == 0:
                self.dqn_t.set_weights(self.dqn_q.get_weights())

            # visualize
            if step % visualize_interval == 0:
                self.env.visualize(state, self)

            # prepare next iteration
            state = succ_state
            state = self.env.environment_change(state, action)
            step += 1
            self.additional_stats = ""

    def get_stats(self):
        return "Statistics:\n" + "- Loss: " + str(self.stats.history['loss'][0]) + "\n" + "Other:\n" + self.additional_stats
