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
    abort = False
    
    # meaning of state dimensions
    ax_idx = 0
    ay_idx = 1
    px_idx = 2
    py_idx = 3
    
    # action space
    action_dim = 4

    # environment space
    width = 50
    height = 50
    box_size = 10
    
    # visualization
    have_frozen_state = False
    frozen_state = 0
    
    # training control
    train = True
    
    # quality ensurance
    sliding_corr_pred = []

    def construct_q_network(self, state_dim: int, action_dim: int) -> keras.Model:
        """Construct the critic network with q-values per action as output"""
        inputs = layers.Input(shape=(state_dim,))  # input dimension
        hidden1 = layers.Dense(
            10, activation="relu", kernel_initializer=initializers.he_normal() # he_normal
        )(inputs)
        hidden2 = layers.Dense(
            10, activation="relu", kernel_initializer=initializers.he_normal() # he_normal
        )(hidden1)
#        hidden3 = layers.Dense(
#            10, activation="relu", kernel_initializer=initializers.he_normal() # he_normal
#        )(hidden2)
        q_values = layers.Dense(
            action_dim, kernel_initializer=initializers.Zeros(), activation="linear"
        )(hidden2)

        deep_q_network = keras.Model(inputs=inputs, outputs=[q_values])

        return deep_q_network

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
        succ_state = self.get_next_state(state, action)
        
        if abs(25 - succ_state[self.ax_idx]) < 5 and abs(25 - succ_state[self.ay_idx]) < 5: #abs( #self.is_correct_decision(state, action):
            return 50
        elif abs(25 - succ_state[self.ax_idx]) < 15 and abs(25 - succ_state[self.ay_idx]) < 15: #abs( #self.is_correct_decision(state, action):
            return 10
        else:
            return 0
    
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
        (state, action, succ_state, reward) = t_trainingsample
        return self.is_correct_decision(state, action)
    
    def show_correct_samples(self, t_replaybuffer):
        correct = 0
        for t_trainingsample in t_replaybuffer:
            if self.is_trainingsample_correct(t_trainingsample):
                correct = correct + 1
        print((correct * 100 / len(t_replaybuffer)), "% decisions in training set correct")

    def game_loop(self, main_window):
        # hyperparameters
        exploration_rate_start = 1.0
        exploration_rate = exploration_rate_start
        exploration_rate_decrease = 0.0001
        learning_rate = 0.01
        succ_state = [25, 25, 0, 0]
        state_dim = 4
        max_sample_storage = 10
        training_interval = 1
        accept_q_network_interval = 20

        # construct q-network
        self.t_network = self.construct_q_network(state_dim, self.action_dim)
        self.q_network = self.construct_q_network(state_dim, self.action_dim)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.t_network.compile(opt, loss="mse")
        self.q_network.compile(opt, loss="mse")
        self.copy_weights(self.q_network, self.t_network)
        
        trainingset = list()
        t_replaybuffer = list()
        step = 0
        while not self.abort:
            step = step + 1
            if self.abort:
                break

            # define current state and obtain q values (estimated by NN)
            state = succ_state
            self.frozen_state = state
            self.have_frozen_state = True
            state = self.update_state_by_user_input(state)
            #state = self.simulate_user_input(state)
            q_values = self.q_network(tf.constant([state]))[0].numpy()

            # choose action
            epsilon = np.random.rand()
            if self.train and epsilon < exploration_rate:
                action = np.random.choice(self.action_dim)
            else:
                action = np.argmax(q_values)
            
            # decrease random choices over time
            exploration_rate -= exploration_rate_decrease
            if exploration_rate < 0:
                exploration_rate = 0

            # go to successor state and obtain reward
            succ_state = self.get_next_state(state, action)
            reward = self.get_reward(state, action)

            # store current observation in training set
            t_trainingsample = [state, action, succ_state, reward]
            while len(t_replaybuffer) >= max_sample_storage:
                del t_replaybuffer[0]
            t_replaybuffer.append(t_trainingsample)

            # training
            if self.train and step % training_interval == 0:
                self.train(t_replaybuffer)
            if self.train and step % accept_q_network_interval == 0:
                print("Accepting q network as target. Exploration rate:", exploration_rate)
                self.copy_weights(self.q_network, self.t_network)

    def copy_weights(self, nn_source, nn_target):
        nn_target.set_weights(nn_source.get_weights())
                
    def train(self, t_replaybuffer):
        # hyperparameters
        sample_size = -1
        #num_epochs = 100
        alpha = 1.0
        gamma = 0.9

        if sample_size < 0:
            # take only most recent training sample
            t_samples = t_replaybuffer[sample_size:]
        else:
            # draw random training samples from replay buffer
            t_samples = random.sample(t_replaybuffer, min(len(t_replaybuffer), sample_size))

        # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
        inp = []
        out = []
        pred_corr = 0
        for t_sample in t_samples:
            t_state = t_sample[0]
            t_action = t_sample[1]
            t_succ_state = t_sample[2]
            t_reward = t_sample[3]
            
            # predict q-values by NN
            t_state_q_values = self.q_network(tf.constant([t_state]))[0].numpy()
            t_succ_state_q_values = self.t_network(tf.constant([t_succ_state]))[0].numpy()

            # update q-value of chosen action (Bellman equation)
            old_q = t_state_q_values[t_action]
            t_state_q_values[t_action] = t_state_q_values[t_action] + alpha * (t_reward + gamma * max(t_succ_state_q_values) - t_state_q_values[t_action])

            # quality check
            if self.is_correct_decision(t_state, np.argmax(t_state_q_values)):
                pred_corr += 1

            # build training batch
            #inp.append(t_state)
            #out.append(t_state_q_values)
            
            # train on single instance
            print("Fitting", "State", t_state, "Action", t_action, "Reward", t_reward, "Updated q value for action", t_state_q_values[t_action], "q values", t_state_q_values, "correctness", self.is_correct_decision(t_state, np.argmax(t_state_q_values)))
            self.q_network.fit(tf.constant([t_state]), tf.constant([t_state_q_values]), epochs=1, verbose=0, shuffle=True)

        #self.q_network.fit(np.asarray(inp), np.asarray(out), epochs=num_epochs, verbose=0, shuffle=True)
        
        self.sliding_corr_pred.append(pred_corr / len(t_samples))
        print("Sliding training sample correctness:", sum(self.sliding_corr_pred[-50:]) / 50)
        #self.print_progress_bar(pred_corr * 100 / len(t_samples))
        #if (sum(self.sliding_corr_pred[-5:]) / 5 > 0.98):
        #    print("Stopping training due to good accuracy")
        #    self.train = False
                    

    def print_progress_bar(self, percentage):
        str = "|"
        ch = 0
        per = percentage
        while (per >= 5):
            str = str + "#"
            per -= 5
            ch += 1
        while ch < 20:
            str = str + "."
            ch += 1
        str += " |"
        print(str, percentage)

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