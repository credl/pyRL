# Needed for training the network
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

# Needed for animation
import matplotlib.pyplot as plt

abort=False
ax_idx=0
ay_idx=1
px_idx=2
py_idx=3
width=50
height=50
box_size=10

def construct_q_network(state_dim: int, action_dim: int) -> keras.Model:
    """Construct the critic network with q-values per action as output"""
    inputs = layers.Input(shape=(state_dim,))  # input dimension
    hidden1 = layers.Dense(
        10, activation="relu", kernel_initializer=initializers.he_normal()
    )(inputs)
    hidden2 = layers.Dense(
        10, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden1)
    hidden3 = layers.Dense(
        10, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden2)
    q_values = layers.Dense(
        action_dim, kernel_initializer=initializers.Zeros(), activation="linear"
    )(hidden3)

    deep_q_network = keras.Model(inputs=inputs, outputs=[q_values])

    return deep_q_network


def mean_squared_error_loss(q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
    """Compute mean squared error loss"""
    loss = 0.5 * (q_value - reward) ** 2

    return loss

def get_next_state(state, action):
    succ_state = np.array(state)
    if action == 0:
        succ_state[ax_idx] = max(0, state[ax_idx] - 1)
    elif action == 1:
        succ_state[ax_idx] = min(width - 1, state[ax_idx] + 1)
    elif action == 2:
        succ_state[ay_idx] = max(0, state[ay_idx] - 1)
    elif action == 3:
        succ_state[ay_idx] = min(height - 1, state[ay_idx] + 1)
    return succ_state

def action_name(action):
    if action == 0:
        return "left"
    elif action == 1:
        return "right"
    elif action == 2:
        return "up"
    elif action == 3:
        return "down"
    return state

def get_reward(state, action):
    succ_state = get_next_state(state, action)
    # try to stay at player's position
    #reward = [-abs(ax - px) - abs(ay - py)]
    #reward = -succ_state[ax_idx] + succ_state[ay_idx]
    reward = -abs(25 - succ_state[ax_idx]) - abs(25 - succ_state[ay_idx])
    return reward

def format_state(state):
    return "{}".format(state[ax_idx]) + "/" + "{}".format(state[ay_idx]) + ", " + "{}".format(state[px_idx]) + "/" + "{}".format(state[py_idx])

def format_step(state, action):
    succ_state = get_next_state(state, action)
    reward = get_reward(state, action)
    return "(" + format_state(state) + " -> " + format_state(succ_state) + ", A: " + "{}".format(action) + ", R:" + "{}".format(reward) + ")"

def visualizefield(main_window, state):
    #main_window["-CANV-"].DrawRectangle((0, 0), (width * box_size, height * box_size), fill_color="white")
    main_window["-CANV-"].DrawRectangle((0 * box_size, 0 * box_size), (width * box_size - 1, 1 * box_size - 1), fill_color="black")
    main_window["-CANV-"].DrawRectangle((0 * box_size, (height - 1) * box_size), (width * box_size - 1, height * box_size - 1), fill_color="black")
    main_window["-CANV-"].DrawRectangle((0 * box_size, 0 * box_size), (1 * box_size - 1, height * box_size - 1), fill_color="black")
    main_window["-CANV-"].DrawRectangle(((width - 1) * box_size, 0 * box_size), (width * box_size - 1, height * box_size - 1), fill_color="black")
    main_window["-CANV-"].DrawRectangle((state[ax_idx] * box_size, state[ay_idx] * box_size), ((state[ax_idx] + 1) * box_size - 1, (state[ay_idx] + 1) * box_size - 1), fill_color="red")
    main_window["-CANV-"].DrawRectangle((state[px_idx] * box_size, state[py_idx] * box_size), ((state[px_idx] + 1) * box_size - 1, (state[py_idx] + 1) * box_size - 1), fill_color="yellow")
    
def update_state_by_user_input(state):
    if keyboard.is_pressed('a'):
        state[px_idx] = state[px_idx] = max(state[px_idx] - 1, 0)
    if keyboard.is_pressed('d'):
        state[px_idx] = min(state[px_idx] + 1, width - 1)
    if keyboard.is_pressed('w'):
        state[py_idx] = max(state[py_idx] - 1, 0)
    if keyboard.is_pressed('s'):
        state[py_idx] = min(state[py_idx] + 1, height - 1)
    return state

def game_loop(main_window):
    # hyperparameters
    action_dim = 4
    exploration_rate = 0.1
    learning_rate = 0.01
    num_episodes = 1000000
    alpha = 0.10
    gamma = 0.99
    succ_state = [0, 0, 10, 10]
    state_dim = 4
    print_interval = 1
    max_sample_storage = 10000
    training_interval = 1000

    # construct q-network
    q_network = construct_q_network(state_dim, action_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    q_network.compile(opt, loss="mse")
    
    trainingset = list()
    step = 0
    while True:
        step = step + 1
        if abort:
            break
        with tf.GradientTape() as tape:
            # define current state and obtain q values (estimated by NN)
            state = succ_state
            state = update_state_by_user_input(state)
            q_values = q_network(tf.constant([state]))

            # choose action
            epsilon = np.random.rand()
            if epsilon <= exploration_rate:
                action = np.random.choice(action_dim)
            else:
                action = np.argmax(q_values)

            # go to successor state and obtain reward
            succ_state = get_next_state(state, action)
            succ_state_q_values = q_network(tf.constant([succ_state]))[0].numpy()
            reward = get_reward(state, action)

            # update q-value
            q_value = q_values[0, action].numpy()
            new_q_value = q_value + alpha * (reward + gamma * max(succ_state_q_values) - q_value)
            succ_state_q_values[action] = new_q_value

            # store observation in training set
            trainingsample = (tuple(state), action, tuple(succ_state), tuple(succ_state_q_values))
            #trainingsample = tf.constant(state, succ_state_q_values)
            #trainingsample = tf.constant([state, succ_state_q_values])
            #print("T", tensor)
            if len(trainingset) >= max_sample_storage:
                trainingset[random.randint(0, max_sample_storage - 1)] = trainingsample
            else:
                trainingset.append(trainingsample)

            # compute loss value and do NN learn
            if step % training_interval == 0:
                #loss_value = mean_squared_error_loss(q_value, new_q_value)
                sample = random.sample(trainingset, min(len(trainingset), 10))
                inp = tf.constant([ list(s) for (s, a, ss, q) in sample ])
                out = tf.constant([ list(q) for (s, a, ss, q) in sample ])
                #q_network.fit(inp, out)
            
            # visualize
            if step % print_interval == 0:
                main_window.write_event_value("redraw", succ_state)
                #sys.stdout.write(format_step(state, action))
                #sys.stdout.write("\n")
                #print(trainingsample, "chosen action:", action_name(action), ", direct reward:", reward)
                #sys.stdout.flush()
                #main_window["-CANV-"].update()

            #q_network.fit()
            #grads = tape.gradient(loss_value[0], q_network.trainable_variables)
            #opt.apply_gradients(zip(grads, q_network.trainable_variables))

if __name__ == "__main__":
    abort = False
    main_window = sg.Window(title="RL",layout = [
                                                [
                                                    sg.Graph(
                                                        canvas_size=(width * box_size, height * box_size),
                                                        graph_bottom_left=(0, height * box_size),
                                                        graph_top_right=(width * box_size, 0),
                                                        key="-CANV-",
                                                        pad=(0,0)
                                                    )
                                                ]
                                            ], size=(width * box_size,height * box_size), margins=(0, 0), finalize=True)
    gl = threading.Thread(target=game_loop, args=(main_window,))
    gl.start()
    while True:
        event, values = main_window.read()
        if event == sg.WIN_CLOSED:
            abort = True
            break
        elif event == "redraw":
            if not abort:
                visualizefield(main_window, values["redraw"])
    gl.join()
