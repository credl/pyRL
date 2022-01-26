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
    corr_dec = 0
    inc_dec = 0
    impr_cnt = 0
    wors_cnt = 0
    pred_corr = 0
    pred_wrong = 0

    def construct_q_network(self, state_dim: int, action_dim: int) -> keras.Model:
        deep_q_network = keras.models.Sequential([
            keras.layers.Dense(32, activation="elu", input_shape=(state_dim,), kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(32, activation="elu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            keras.layers.Dense(action_dim, activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
        ])
        print(deep_q_network.weights)
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
        #print("S:", state, "A:", action)
        succ_state = self.get_next_state(state, action)
        #print("SS:", succ_state)
        
#        if self.is_correct_decision(state, action):
#            return 50
#        else:
#            return 0

#        if abs(25 - succ_state[self.ax_idx]) < 5 and abs(25 - succ_state[self.ay_idx]) < 5: #abs( #self.is_correct_decision(state, action):
#            return 50
#        elif abs(25 - succ_state[self.ax_idx]) < 15 and abs(25 - succ_state[self.ay_idx]) < 15: #abs( #self.is_correct_decision(state, action):
#            return 10
#        else:
#            return 0
    
        #reward = succ_state[self.ax_idx] + succ_state[self.ay_idx]
        reward = (25 - max(abs(25 - succ_state[self.ax_idx]), abs(25 - succ_state[self.ay_idx])))
        return reward

    def format_state(self, state):
        return "{}".format(state[self.ax_idx]) + "/" + "{}".format(state[self.ay_idx]) + ", " + "{}".format(state[self.px_idx]) + "/" + "{}".format(state[self.py_idx])

    def format_step(self, state, action):
        succ_state = self.get_next_state(state, action)
        reward = get_reward(state, action)
        return "(" + format_state(state) + " -> " + format_state(succ_state) + ", A: " + "{}".format(action) + ", R:" + "{}".format(reward) + ")"

    def visualizefield(self, main_window, state):
        main_window["-CANV-"].DrawRectangle((0, 0), (self.width * self.box_size, self.height * self.box_size), fill_color="white")

        nn_out = []
        for y in range(0, self.height, 5):
            for x in range(0, self.height, 5):
                nn_out.append(self.q_network(tf.constant([[x,y]]))[0])
                action = np.argmax(nn_out[-1])
                if action == 0:
                    col = "yellow"
                    offsx = -self.box_size / 2
                    offsy = 0
                elif action == 1:
                    col = "green"
                    offsx = +self.box_size / 2
                    offsy = 0
                elif action == 2:
                    col = "blue"
                    offsx = 0
                    offsy = -self.box_size / 2
                elif action == 3:
                    col = "grey"
                    offsx = 0
                    offsy = +self.box_size / 2
                main_window["-CANV-"].DrawRectangle((x * self.box_size, y * self.box_size), ((x + 1) * self.box_size - 1, (y + 1) * self.box_size - 1), fill_color=col)
                main_window["-CANV-"].DrawRectangle(((x + 0.5) * self.box_size + offsx - 3, (y + 0.5) * self.box_size + offsy - 3), ((x + 0.5) * self.box_size - 1 + offsx + 3, (y + 0.5) * self.box_size - 1 + offsy + 3), fill_color="black")
                #main_window["-CANV-"].DrawText(x * self.box_size, y * self.box_size, fill="red", text="H")
#        print("NNOUT", nn_out)

        main_window["-CANV-"].DrawRectangle((0 * self.box_size, 0 * self.box_size), (self.width * self.box_size - 1, 1 * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle((0 * self.box_size, (self.height - 1) * self.box_size), (self.width * self.box_size - 1, self.height * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle((0 * self.box_size, 0 * self.box_size), (1 * self.box_size - 1, self.height * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle(((self.width - 1) * self.box_size, 0 * self.box_size), (self.width * self.box_size - 1, self.height * self.box_size - 1), fill_color="black")
        main_window["-CANV-"].DrawRectangle((state[self.ax_idx] * self.box_size, state[self.ay_idx] * self.box_size), ((state[self.ax_idx] + 1) * self.box_size - 1, (state[self.ay_idx] + 1) * self.box_size - 1), fill_color="red")
        #main_window["-CANV-"].DrawRectangle((state[self.px_idx] * self.box_size, state[self.py_idx] * self.box_size), ((state[self.px_idx] + 1) * self.box_size - 1, (state[self.py_idx] + 1) * self.box_size - 1), fill_color="yellow")

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
        dest_x = 25
        dest_y = 25
        if abs(state[self.ax_idx] - dest_x) == 0 and abs(state[self.ay_idx] - dest_y) == 0:
            return True
        elif abs(state[self.ax_idx] - dest_x) == abs(state[self.ay_idx] - dest_y):
            if state[self.ax_idx] > dest_x and state[self.ay_idx] > dest_y:
                return action == 0 or action == 2
            elif state[self.ax_idx] > dest_x and state[self.ay_idx] < dest_y:
                return action == 0 or action == 3
            elif state[self.ax_idx] < dest_x and state[self.ay_idx] > dest_y:
                return action == 1 or action == 2
            else:
                return action == 1 or action == 3
        elif abs(state[self.ax_idx] - dest_x) > abs(state[self.ay_idx] - dest_y):
            if state[self.ax_idx] > dest_x:
                dest_action = 0
            else:
                dest_action = 1
        else:
            if state[self.ay_idx] > dest_y:
                dest_action = 2
            else:
                dest_action = 3

        return action == dest_action
    
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
        exploration_rate_decrease = 0.0 #001
        nn_learning_rate = 0.2
        succ_state = [25, 25] #, 0, 0]
        state_dim = 2
        max_sample_storage = 2000
        training_interval = 1
        accept_q_network_interval = 1
        random_state_change_probability = 0.5
        random_state_change_probability_decrease = 0.0 #01

        # construct q-network
        self.t_network = self.construct_q_network(state_dim, self.action_dim)
        self.q_network = self.construct_q_network(state_dim, self.action_dim)
        self.opt = tf.keras.optimizers.Adam(learning_rate=nn_learning_rate)
        self.t_network.compile(self.opt, loss="mse")
        self.q_network.compile(self.opt, loss="mse")
        self.copy_weights(self.q_network, self.t_network)
        
        trainingset = list()
        t_replaybuffer = deque(maxlen=max_sample_storage)
        step = 0
        while not self.abort:
            step = step + 1
            if self.abort:
                break

            state = succ_state
            
            # state randomization
            if np.random.rand() < random_state_change_probability:
                state[self.ax_idx] = np.random.choice(self.width)
                state[self.ay_idx] = np.random.choice(self.height)

            if random_state_change_probability > 0:
                random_state_change_probability -= random_state_change_probability_decrease
                if random_state_change_probability < 0:
                    random_state_change_probability = 0

            # define current state and obtain q values (estimated by NN)
            self.frozen_state = state
            self.have_frozen_state = True
            #state = self.update_state_by_user_input(state)
            #state = self.simulate_user_input(state)
            q_values = self.q_network(tf.constant([state]))[0].numpy()
            #print("Decision:", "S", state, "Q", np.array_str(q_values, precision=2), "Best action by Q", self.action_name(np.argmax(q_values)), "ER:", exploration_rate, "CORRECT:", self.corr_dec)
            if self.is_correct_decision(state, np.argmax(q_values)):
                self.corr_dec += 1
            else:
                self.inc_dec += 1
            print("Correctness of NN decisions:", self.corr_dec, "/", self.inc_dec, "(", self.corr_dec * 100 / (self.corr_dec + self.inc_dec), "% correct)")

            # choose action
            epsilon = np.random.rand()
            if self.train and epsilon < exploration_rate:
                action = np.random.choice(self.action_dim)
            else:
                action = np.argmax(q_values)
            
            # decrease random choices over time
            if exploration_rate > 0:
                exploration_rate -= exploration_rate_decrease
                if exploration_rate < 0.0:
                    exploration_rate = 0
                    self.train = False
                    print("Stopping random choices")

            # go to successor state and obtain reward
            succ_state = self.get_next_state(state, action)
            reward = self.get_reward(state, action)

            # store current observation in training set
            t_trainingsample = [state, action, succ_state, reward]
            t_replaybuffer.append(t_trainingsample)

            # training
            if self.train and step % training_interval == 0:
                self.train_fit(t_replaybuffer)
            if self.train and step % accept_q_network_interval == 0:
                #print("Accepting q network as target. Exploration rate:", exploration_rate)
                self.copy_weights(self.q_network, self.t_network)

    def copy_weights(self, nn_source, nn_target):
        nn_target.set_weights(nn_source.get_weights())

    def train_grad(self, t_replaybuffer):
        # hyperparameters
        sample_size = 32
        alpha_q_learning_rate = 1.0
        gamma_discout_factor = 0.0
        loss_fn = keras.losses.MeanSquaredError()

        if sample_size < 0:
            # take only most recent training sample
            t_samples = t_replaybuffer[sample_size:]
        else:
            # draw random training samples from replay buffer
            t_samples = random.sample(t_replaybuffer, min(len(t_replaybuffer), sample_size))

        # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
        inp = []
        mask = []
        target = []
        for t_sample in t_samples:
            t_state = t_sample[0]
            t_action = t_sample[1]
            t_succ_state = t_sample[2]
            t_reward = t_sample[3]

            # predict q-values by NN
            t_state_q_values = self.q_network(tf.constant([t_state]))[0]
            t_succ_state_q_values = self.t_network(tf.constant([t_succ_state]))[0].numpy()

            # update q-value of chosen action (Bellman equation)
            new_t_state_target_q_value = t_state_q_values[t_action] + alpha_q_learning_rate * (t_reward + gamma_discout_factor * max(t_succ_state_q_values) - t_state_q_values[t_action])

            # build training batch
            inp.append(list(t_state))
            mask.append(tf.one_hot(t_action, self.action_dim).numpy())
            target.append(new_t_state_target_q_value.numpy())

        target = tf.constant(target)

        with tf.GradientTape() as tape:
            nn_out = self.q_network(tf.constant(inp))
            #print("NNO", nn_out)
            all_q_values = tf.reduce_sum(nn_out * tf.constant(mask), axis=1)
            loss = tf.reduce_mean(loss_fn(target, all_q_values))
            #print("Inpt:", inp)
            #print("Curr:", all_q_values)
            #print("Targ:", target)
            dif1 = sum(abs(target - all_q_values)).numpy()
            #print("Dif1:", target - all_q_values, "(", dif1, ")")
            #print("Loss:", loss)
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))
        all_q_values = tf.reduce_sum(self.q_network(tf.constant(inp)) * tf.constant(mask), axis=1)
        #print("Updt:", all_q_values)
        dif2 = sum(abs(target - all_q_values)).numpy()
        #print("Dif2:", target - all_q_values, "(", dif2, ")")
        if dif2 < dif1:
            self.impr_cnt += 1
            #print("IMPROVED by", dif1 - dif2, "(imp%:", self.impr_cnt * 100 / (self.impr_cnt + self.wors_cnt), ")")
        else:
            self.wors_cnt += 1
            #print("WORSENED by", dif2 - dif1, "(imp%:", self.impr_cnt * 100 / (self.impr_cnt + self.wors_cnt), ")")
        print("L:", loss.numpy(), "C:", dif2, "IR:", self.impr_cnt, "/", self.wors_cnt, "(imp%:", self.impr_cnt * 100 / (self.impr_cnt + self.wors_cnt), ")")

        #self.sliding_corr_pred.append(self.pred_corr / len(t_samples))
        #print("Sliding training sample correctness:", sum(self.sliding_corr_pred[-50:]) / 50)
        #self.print_progress_bar(pred_corr * 100 / len(t_samples))
        #if (sum(self.sliding_corr_pred[-5:]) / 5 > 0.98):
        #    print("Stopping training due to good accuracy")
        #    self.train = False

    corr_t_q = deque(maxlen=100)
    stat_avg = deque(maxlen=100)
    ts = 0
    def statavg(self):
        xs = ys = zs = ws = 0
        for (x,y,z,w) in self.stat_avg:
            xs += x
            ys += y
            zs += z
            ws += w
        return (xs / len(self.stat_avg), ys / len(self.stat_avg), zs / len(self.stat_avg), ws / len(self.stat_avg))
    
    def train_fit(self, t_replaybuffer):
        # hyperparameters
        sample_size = 32
        num_epochs = 10
        alpha_q_learning_rate = 1.0
        gamma_discout_factor = 0.0
        loss_fn = keras.losses.MeanSquaredError()

        if sample_size < 0:
            # take only most recent training sample
            t_samples = [t_replaybuffer[sample_size]]
        else:
            # draw random training samples from replay buffer
            t_samples = random.sample(t_replaybuffer, min(len(t_replaybuffer), sample_size))

        # get current q-values (current NN prediction) of selected training samples and update them according to observed reward
        inp = []
        out = []
        for t_sample in t_samples:
            t_state = t_sample[0]
            t_action = t_sample[1]
            t_succ_state = t_sample[2]
            t_reward = t_sample[3]

            # predict q-values by NN
            t_state_q_values = self.q_network(tf.constant([t_state]))[0]
            t_succ_state_q_values = self.t_network(tf.constant([t_succ_state]))[0].numpy()

            # update q-value of chosen action (Bellman equation)
            new_t_state_q_values = t_state_q_values.numpy()
#            for t_action in range(4):
            t_reward = self.get_reward(t_state, t_action)
            new_t_state_q_values[t_action] = new_t_state_q_values[t_action] + alpha_q_learning_rate * (t_reward + gamma_discout_factor * max(t_succ_state_q_values) - new_t_state_q_values[t_action])

            # quality check
            if self.is_correct_decision(t_state, np.argmax(t_state_q_values)):
                self.pred_corr += 1
            else:
                self.pred_wrong += 1

            # build training batch
            inp.append(t_state)
            out.append(new_t_state_q_values)
            
            # train on single instance
            self.q_network.fit(tf.constant([t_state]), tf.constant([new_t_state_q_values]), epochs=num_epochs, verbose=0)

            new_q_values = self.q_network(tf.constant([t_state]))[0].numpy()
            diff = new_q_values - t_state_q_values.numpy()
            if new_t_state_q_values[t_action] > t_state_q_values[t_action]:
                change_in_correct_direction = new_q_values[t_action] > t_state_q_values[t_action]
            else:
                change_in_correct_direction = new_q_values[t_action] < t_state_q_values[t_action]
            largest_change = abs(diff[t_action]) >= max(abs(diff))
            correct_dec = self.is_correct_decision(t_state, np.argmax(new_q_values))
            self.stat_avg.append((change_in_correct_direction.numpy(), largest_change, correct_dec, sum(abs(diff))))
            self.ts += 1
            print("TS", self.ts, "CICD", 1 if change_in_correct_direction.numpy() else 0, "LC", 1 if largest_change else 0, "CORR", 1 if self.is_correct_decision(t_state, np.argmax(new_q_values)) else 0, "AVG", self.statavg(), "C", sum(abs(diff)),
                "S", t_state, "A", t_action, "R", t_reward, "O", np.array_str(t_state_q_values.numpy(), precision=2), "T", np.array_str(new_t_state_q_values, precision=2), "U", np.array_str(new_q_values), "D", np.array_str(diff, precision=2), "correctness", correct_dec)

        # train on single instance
        corr_t = 0
        for (i, o) in zip(inp, out):
            if self.is_correct_decision(i, np.argmax(o)):
                corr_t += 1
            #print("Training:", i, o, self.is_correct_decision(i, np.argmax(o)))
        self.corr_t_q.append(corr_t / len(inp))
        #print("Corr T:", corr_t, "/", len(inp), "Corr T avg:", sum(self.corr_t_q) * 100 / len(self.corr_t_q), "%")
        #self.q_network.fit(tf.constant(inp), tf.constant(out), epochs=num_epochs, verbose=0)

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
            event, values = main_window.read(timeout=1)
            if event == sg.WIN_CLOSED:
                self.abort = True
                break
            else:
                if self.have_frozen_state:
                    self.visualizefield(main_window, self.frozen_state)
                #if keyboard.is_pressed('Esc'):
                #    main_window.close()

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .2f}'.format})
    RL().run()