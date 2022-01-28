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
    stats = 0   # model evaluation
    loss_fn = 0 # nn loss function
    env = 0     # environment
    additional_stats = "" # statistics output other than provided by tf

    class Environment:
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

class Centering(RLFramework.Environment):
    AC_LEFT = 0
    AC_RIGHT = 1
    AC_UP = 2
    AC_DOWN = 3
    AC_SHOOT = 4
    
    STATE_IDX_X = 0
    STATE_IDX_Y = 1
    STATE_IDX_PX = 2
    STATE_IDX_PY = 3
    
    WIDTH = 50
    HEIGHT = 50
    
    walls = []
    
    # shooting mechanics
    shots = []
    shotdirs = []
    last_agent_non_shoot_action = AC_LEFT

    def __init__(self):
        #self.add_walls()
        pass

    def add_walls(self):
        for x in range(15,30):
            self.walls.append((x, 20))
        for y in range(0,20):
            self.walls.append((20, y))

    def get_state_dim(self):
        return 4
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
        # do not cross walls
        if (ss[self.STATE_IDX_X], ss[self.STATE_IDX_Y]) in self.walls:
            ss = list(state)

        # compute reward
        #reward = (max(self.WIDTH, self.HEIGHT) - max(abs(self.WIDTH / 2 - ss[self.STATE_IDX_X]), abs(self.HEIGHT / 2 - ss[self.STATE_IDX_Y])))  # stay centered
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(ss[self.STATE_IDX_PX] - ss[self.STATE_IDX_X]), abs(ss[self.STATE_IDX_PY] - ss[self.STATE_IDX_Y])))  # stay with other player

        # additional costs for shoots and reward for hits
        if action == self.AC_SHOOT:
            reward -= 20
        for idx in range(len(self.shots) - 1, -1, -1):
            if self.does_hit(state, idx):
                reward += 10000

        return (ss, reward)

    def does_hit(self, state, shot_idx):
        hit_radius = 2
        (x, y) = self.shots[shot_idx]
        return abs(x - state[self.STATE_IDX_PX]) < hit_radius and abs(y- state[self.STATE_IDX_PY]) < hit_radius
        
    def get_start_state(self):
        return [self.WIDTH / 2, self.HEIGHT / 2, 0, 0]
    
    def action_to_char(self, action):
        if action == self.AC_LEFT:
            return "<"
        elif action == self.AC_RIGHT:
            return ">"
        elif action == self.AC_UP:
            return "^"
        elif action == self.AC_DOWN:
            return "v"
        elif action == self.AC_SHOOT:
            return " "
        return " "
    
    def visualize(self, state, rlf):
        # print field
        print_density = 5
        out = "Current state:\n"
        for y in range(50):
            for x in range(50):
                if x == state[self.STATE_IDX_X] and y == state[self.STATE_IDX_Y]:
                    out += "X"
                elif x == state[self.STATE_IDX_PX] and y == state[self.STATE_IDX_PY]:
                    out += "O"
                else:
                    if (x,y) in self.walls:
                        out += "#"
                    elif (x,y) in self.shots:
                        out += "*"
                    elif x % print_density == 0 and y % print_density == 0:
                        #action = -1
                        #action = rlf.get_action([x, y])
                        action = rlf.get_action([x, y, state[self.STATE_IDX_PX], state[self.STATE_IDX_PY]])
                        out += str(self.action_to_char(action))
                    else:
                        out += " "
            out += "\n"
        print(out)
        print(rlf.get_stats())

    def agent_shooting(self, state, action):
        if action == self.AC_SHOOT:
            self.shots.append((state[self.STATE_IDX_X], state[self.STATE_IDX_Y]))
            self.shotdirs.append(self.last_agent_non_shoot_action)
        else:
            self.last_agent_non_shoot_action = action
        for idx in range(len(self.shots) - 1, -1, -1):
            if self.does_hit(state, idx):
                delete = True
            elif self.shots[idx] in self.walls:
                delete = True
            else:
                (x, y) = self.shots[idx]
                d = self.shotdirs[idx]
                delete = False
                if d == self.AC_LEFT:
                    x -= 1
                    if x < 0:
                        delete = True
                elif d == self.AC_RIGHT:
                    x += 1
                    if x >= self.WIDTH:
                        delete = True
                elif d == self.AC_UP:
                    y -= 1
                    if y < 0:
                        delete = True
                elif d == self.AC_DOWN:
                    y += 1
                    if y >= self.HEIGHT:
                        delete = True
                self.shots[idx]= (x, y)
            if delete:
                del self.shots[idx]
                del self.shotdirs[idx]
        return state

    def move_second_player(self, state, action):
        # move second player around
        if state[self.STATE_IDX_PX] == 0:
            # go up at left edge
            if state[self.STATE_IDX_PY] > 0:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] - 1
            else:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] + 1
        elif state[self.STATE_IDX_PY] == 0:
            # go right at top edge
            if state[self.STATE_IDX_PX] < self.WIDTH - 1:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] + 1
            else:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] + 1
        elif state[self.STATE_IDX_PX] == self.WIDTH - 1:
            # go down at right edge
            if state[self.STATE_IDX_PY] < self.HEIGHT - 1:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] + 1
            else:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] - 1
        elif state[self.STATE_IDX_PY] == self.HEIGHT - 1:
            # go left at bottom edge
            if state[self.STATE_IDX_PX] > 0:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] - 1
            else:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] - 1
        return state
        
    def environment_change(self, state, action):
        state = self.agent_shooting(state, action)
        state = self.move_second_player(state, action)
        return state

if __name__ == "__main__":
    RLFramework(Centering()).train(visualize_interval=1)