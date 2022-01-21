import os
import sys
import keyboard
import random
import threading
import PySimpleGUI as sg
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

class RL:
    abort=False
    ax_idx=0
    ay_idx=1
    px_idx=2
    py_idx=3
    width=50
    height=50
    box_size=10
    have_frozen_state=False
    frozen_state=0

    def construct_q_network(self, state_dim: int, action_dim: int) -> keras.Model:
        """Construct the critic network with q-values per action as output"""
        inputs = layers.Input(shape=(state_dim,))  # input dimension
        hidden1 = layers.Dense(
            10, activation="relu", kernel_initializer=initializers.he_normal() # he_normal
        )(inputs)
        hidden2 = layers.Dense(
            10, activation="relu", kernel_initializer=initializers.he_normal() # he_normal
        )(hidden1)
        hidden3 = layers.Dense(
            10, activation="relu", kernel_initializer=initializers.he_normal() # he_normal
        )(hidden2)
        q_values = layers.Dense(
            action_dim, kernel_initializer=initializers.Zeros(), activation="linear"
        )(hidden3)

        deep_q_network = keras.Model(inputs=inputs, outputs=[q_values])

        return deep_q_network


    def mean_squared_error_loss(self, q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
        """Compute mean squared error loss"""
        loss = 0.5 * (q_value - reward) ** 2

        return loss

    def get_next_state(self, state, action):
        succ_state = np.array(state)
        if action == 0:
            succ_state[self.ax_idx] = max(0, state[self.ax_idx] - 1)
        elif action == 1:
            succ_state[self.ax_idx] = min(self.width - 1, state[self.ax_idx] + 1)
        elif action == 2:
            succ_state[self.ay_idx] = max(0, state[self.ay_idx] - 1)
        elif action == 3:
            succ_state[self.ay_idx] = min(self.height - 1, state[self.ay_idx] + 1)
        return succ_state

    def action_name(self, action):
        if action == 0:
            return "left"
        elif action == 1:
            return "right"
        elif action == 2:
            return "up"
        elif action == 3:
            return "down"
        return state

    def get_reward(self, state, action):
        #if self.is_correct_decision(state, action):
        #    return 10
        #else:
        #    return 0
    
        succ_state = self.get_next_state(state, action)
        #reward = succ_state[self.ax_idx] + succ_state[self.ay_idx]
        reward =  50 -abs(25 - succ_state[self.ax_idx]) - abs(25 - succ_state[self.ay_idx])
        return reward

    def format_state(self, state):
        return "{}".format(state[self.ax_idx]) + "/" + "{}".format(state[self.ay_idx]) + ", " + "{}".format(state[self.px_idx]) + "/" + "{}".format(state[self.py_idx])

    def format_step(self, state, action):
        succ_state = self.get_next_state(state, action)
        reward = get_reward(state, action)
        return "(" + format_state(state) + " -> " + format_state(succ_state) + ", A: " + "{}".format(action) + ", R:" + "{}".format(reward) + ")"

    def visualizefield(self, main_window, state):
        main_window["-CANV-"].DrawRectangle((0, 0), (self.width * self.box_size, self.height * self.box_size), fill_color="white")
        main_window["-CANV-"].DrawRectangle((0 * self.box_size, 0 * self.box_size), (self.width * self.box_size - 1, 1 * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle((0 * self.box_size, (self.height - 1) * self.box_size), (self.width * self.box_size - 1, self.height * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle((0 * self.box_size, 0 * self.box_size), (1 * self.box_size - 1, self.height * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle(((self.width - 1) * self.box_size, 0 * self.box_size), (self.width * self.box_size - 1, self.height * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle((state[self.ax_idx] * self.box_size, state[self.ay_idx] * self.box_size), ((state[self.ax_idx] + 1) * self.box_size - 1, (state[self.ay_idx] + 1) * self.box_size - 1), fill_color="red")
        main_window["-CANV-"].DrawRectangle((state[self.px_idx] * self.box_size, state[self.py_idx] * self.box_size), ((state[self.px_idx] + 1) * self.box_size - 1, (state[self.py_idx] + 1) * self.box_size - 1), fill_color="yellow")
        
    def simulate_user_input(self, state):
        if state[self.px_idx] == 0:
            # go up at left edge
            if state[self.py_idx] > 0:
                state[self.py_idx] = state[self.py_idx] - 1
            else:
                state[self.px_idx] = state[self.px_idx] + 1
        elif state[self.py_idx] == 0:
            # go right at top edge
            if state[self.px_idx] < self.width - 1:
                state[self.px_idx] = state[self.px_idx] + 1
            else:
                state[self.py_idx] = state[self.py_idx] + 1
        elif state[self.px_idx] == self.width - 1:
            # go down at right edge
            if state[self.py_idx] < self.height - 1:
                state[self.py_idx] = state[self.py_idx] + 1
            else:
                state[self.px_idx] = state[self.px_idx] - 1
        elif state[self.py_idx] == self.height - 1:
            # go left at bottom edge
            if state[self.px_idx] > 0:
                state[self.px_idx] = state[self.px_idx] - 1
            else:
                state[self.py_idx] = state[self.py_idx] - 1
        return state

    def update_state_by_user_input(self, state):
        if keyboard.is_pressed('a'):
            state[self.px_idx] = state[self.px_idx] = max(state[self.px_idx] - 1, 0)
        if keyboard.is_pressed('d'):
            state[self.px_idx] = min(state[self.px_idx] + 1, self.width - 1)
        if keyboard.is_pressed('w'):
            state[self.py_idx] = max(state[self.py_idx] - 1, 0)
        if keyboard.is_pressed('s'):
            state[self.py_idx] = min(state[self.py_idx] + 1, self.height - 1)
        return state

    def is_correct_decision(self, state, action):
        if action == 0:
            return state[self.ax_idx] >= 25
        if action == 1:
            return state[self.ax_idx] <= 25
        if action == 2:
            return state[self.ay_idx] >= 25
        if action == 3:
            return state[self.ay_idx] <= 25
        return False
    
    def is_trainingsample_correct(self, t_trainingsample):
        (state, q_values) = t_trainingsample
        return self.is_correct_decision(state, np.argmax(q_values))
    
    def show_correct_samples(self, t_trainingset):
        correct = 0
        for t_trainingsample in t_trainingset:
            if self.is_trainingsample_correct(t_trainingsample):
                correct = correct + 1
        print((correct * 100 / len(t_trainingset)), "% decisions in training set correct")

    def game_loop(self, main_window):
        # hyperparameters
        action_dim = 4
        exploration_rate_start = 1.0
        exploration_rate = exploration_rate_start
        exploration_rate_decrease = 0.001
        learning_rate = 0.01
        num_epochs = 10
        alpha = 0.9
        gamma = 0.0
        succ_state = [0, 0, 0, 0]
        state_dim = 4
        print_interval = 1
        max_sample_storage = 100
        training_interval = 10
        sample_size = 100
        explore_all_actions = True

        # construct q-network
        q_network = self.construct_q_network(state_dim, action_dim)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        q_network.compile(opt, loss="mse")
        
        trainingset = list()
        t_trainingset = list()
        step = 0
        while not self.abort:
            step = step + 1
            if self.abort:
                break
            with tf.GradientTape() as tape:
                # define current state and obtain q values (estimated by NN)
                state = succ_state
                self.frozen_state = state
                self.have_frozen_state = True
                state = self.update_state_by_user_input(state)
                #state = self.simulate_user_input(state)
                q_values = q_network(tf.constant([state]))[0].numpy()

                # choose action
                epsilon = np.random.rand()
                if epsilon <= exploration_rate:
                    action = np.random.choice(action_dim)
                else:
                    action = np.argmax(q_values)
                exploration_rate = exploration_rate - exploration_rate_decrease
                if exploration_rate < 0:
                    exploration_rate = 0

                if explore_all_actions:
                    action_range = list(range(0, action_dim))
                    action_range.remove(action)
                    action_range.append(action)
                else:
                    action_range = range(action, action + 1)
                for hypothetical_action in action_range:
                    # go to successor state and obtain reward
                    succ_state = self.get_next_state(state, hypothetical_action)
                    succ_state_q_values = q_network(tf.constant([succ_state]))[0].numpy()
                    reward = self.get_reward(state, hypothetical_action)

                    # update q-value
                    q_value = q_values[hypothetical_action]
                    new_q_value = q_value + alpha * (reward + gamma * max(succ_state_q_values) - q_value)
                    q_values[hypothetical_action] = new_q_value

                # store observation in training set
                #trainingsample = (tuple(state), action, tuple(succ_state), tuple(succ_state_q_values))
                #print("INP:",state,", AC:",action,", OUT:",q_values,", CORR:",self.is_correct_decision(state, np.argmax(q_values)))
                #print(exploration_rate)
                t_trainingsample = [state, q_values]
                while len(t_trainingset) >= max_sample_storage:
                    #del trainingset[0]
                    del t_trainingset[0]
                #trainingset.append(trainingsample)
                t_trainingset.append(t_trainingsample)
                #print("State:", state, ", Action:", action, ", Reward:", reward, ", qvalue[", action, "]:", new_q_value, ", qvalues:", q_values, ", correct:", self.is_trainingsample_correct(t_trainingsample))

                # compute loss value and do NN learn
                if step % training_interval == 0:
                    #sample = random.sample(trainingset, min(len(trainingset), sample_size))
                    #inp = tf.constant([ list(s) for (s, a, ss, q) in sample ])
                    #out = tf.constant([ list(q) for (s, a, ss, q) in sample ])

                    if sample_size == -1:
                        # take only most recent training sample
                        t_sample = tf.constant([t_trainingsample])
                    else:
                        t_sample = tf.constant(random.sample(t_trainingset, min(len(t_trainingset), sample_size)))
                    inp = tf.reshape(tf.gather(t_sample, [0], axis=1), [t_sample.shape[0], 4])
                    out = tf.reshape(tf.gather(t_sample, [1], axis=1), [t_sample.shape[0], 4])
                    q_network.fit(inp, out, epochs=num_epochs)
                    #print("Exploration rate:", exploration_rate)
                    
                    self.show_correct_samples(t_trainingset)
                
    def run(self):
        self.abort = False
        main_window = sg.Window(title="RL",layout = [
                                                    [
                                                        sg.Graph(
                                                            canvas_size=(self.width * self.box_size, self.height * self.box_size),
                                                            graph_bottom_left=(0, self.height * self.box_size),
                                                            graph_top_right=(self.width * self.box_size, 0),
                                                            key="-CANV-",
                                                            pad=(0,0)
                                                        )
                                                    ]
                                                ], size=(self.width * self.box_size, self.height * self.box_size), margins=(0, 0), finalize=True)
        gl = threading.Thread(target=self.game_loop, args=(main_window, ))
        gl.start()
        while not self.abort:
            event, values = main_window.read(timeout=50)
            if event == sg.WIN_CLOSED:
                self.abort = True
                break
            else:
                if self.have_frozen_state:
                    self.visualizefield(main_window, self.frozen_state)
                if keyboard.is_pressed('Esc'):
                    main_window.close()

if __name__ == "__main__":
    RL().run()