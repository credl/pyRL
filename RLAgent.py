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

class RLFramework:
    dqn_q = 0   # deep q network
    dqn_t = 0   # deep q target network
    loss_fn = 0 # nn loss function
    env = 0     # environment

    class Environment:
        def get_state_dim(self):
            return 0
        def get_action_dim(self):
            return 0
        def get_succ_state(self, state, action):
            return self.next(state, action)[0]
        def get_reward(self, state, action):
            return self.next(state, action)[1]
        def next(self, state, action):
            return ([], 0)
        def get_start_state(self):
            return []
        def get_random_state(self):
            return []
        def visualize(self, state, rlf):
            return

    def __init__(self, env, nn_learning_rate: float = 0.01):
        self.env = env

        # construct_q_network
        self.dqn_q = keras.models.Sequential([
            keras.layers.Dense(32, activation="elu", input_shape=(env.get_state_dim(),), kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(32, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
        ])
        self.dqn_t = keras.models.Sequential([
            keras.layers.Dense(32, activation="elu", input_shape=(env.get_state_dim(),), kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(32, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
        ])
        self.loss_fn = keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)
        self.dqn_t.compile(self.opt, loss=self.loss_fn)
        self.dqn_q.compile(self.opt, loss=self.loss_fn)
        self.dqn_t.set_weights(self.dqn_q.get_weights())

    def get_action(self, state):
        q_values = self.dqn_q(tf.constant([state]))[0].numpy()
        return np.argmax(q_values)

    def train(self,
                periods: int = -1,
                nn_epochs: int = 10,
                sample_size: int = 32,
                alpha_q_learning_rate: float = 0.1,
                gamma_discout_factor: float = 0.7,
                exploration_rate_start: float = 0.5,
                exploration_rate_decrease: float = 0.0,
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
            step += 1

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
            if exploration_rate > 0:
                exploration_rate -= exploration_rate_decrease
                if exploration_rate < 0.0:
                    exploration_rate = 0

            # apply action
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
                self.dqn_q.fit(tf.constant(inp), tf.constant(out), epochs=nn_epochs, verbose=0)

            state = succ_state

            # accept q network as new target
            if step % accept_q_network_interval == 0:
                self.dqn_t.set_weights(self.dqn_q.get_weights())

            # visualize
            if step % visualize_interval == 0:
                self.env.visualize(state, self)

        return 0

class Centering(RLFramework.Environment):
    AC_LEFT = 0
    AC_RIGHT = 1
    AC_UP = 2
    AC_DOWN = 3
    
    STATE_IDX_X = 0
    STATE_IDX_Y = 1
    
    WIDTH = 50
    HEIGHT = 50

    def get_state_dim(self):
        return 2
    def get_action_dim(self):
        return 4
    def next(self, state, action):
        # compute next state
        ss = list(state)
        if action == self.AC_LEFT:
            ss[self.STATE_IDX_X] -= 1
            if ss[self.STATE_IDX_X] < 0:
                ss[self.STATE_IDX_X] = 0
        elif action == self.AC_RIGHT:
            ss[self.STATE_IDX_X] += 1
            if ss[self.STATE_IDX_X] >= self.WIDTH:
                ss[self.STATE_IDX_X] = self.WIDTH - 1
        elif action == self.AC_UP:
            ss[self.STATE_IDX_Y] -= 1
            if ss[self.STATE_IDX_Y] < 0:
                ss[self.STATE_IDX_Y] = 0
        elif action == self.AC_DOWN:
            ss[self.STATE_IDX_Y] += 1
            if ss[self.STATE_IDX_Y] >= self.HEIGHT:
                ss[self.STATE_IDX_Y] = self.HEIGHT - 1

        # compute reward
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(self.WIDTH - ss[self.STATE_IDX_X]), abs(self.HEIGHT - ss[self.STATE_IDX_Y])))

        return (ss, reward)
        
    def get_start_state(self):
        return [self.WIDTH / 2, self.HEIGHT / 2]
        
    def visualize(self, state, rlf):
        print_density = 5
        out = "Current state:\n"
        for y in range(50):
            for x in range(50):
                if x == state[self.STATE_IDX_X] and y == state[self.STATE_IDX_Y]:
                    out += "X"
                else:
                    if x % print_density == 0 and y % print_density == 0:
                        action = rlf.get_action([x, y])
                        out += str(action)
                    else:
                        out += " "
            out += "\n"
        print(out)
        return

if __name__ == "__main__":
    RLFramework(Centering()).train()