# Needed for training the network
import os
import sys
import keyboard
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

# Needed for animation
import matplotlib.pyplot as plt

width=50
height=50

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
    ax = state[0,0].numpy()
    ay = state[0,1].numpy()
    px = state[0,2].numpy()
    py = state[0,3].numpy()
    if action == 0:
        ax = max(0, ax - 1)
    elif action == 1:
        ax = min(width - 1, ax + 1)
    elif action == 2:
        ay = max(0, ay - 1)
    elif action == 3:
        ay = min(height - 1, ay + 1)
    ret = tf.constant([[ax,ay,px,py]])
    return ret

def get_reward(state, action):
    succ_state = get_next_state(state, action)
    ax = state[0,0].numpy()
    ay = state[0,1].numpy()
    px = state[0,2].numpy()
    py = state[0,3].numpy()
    # try to stay at player's position
    #reward = [-abs(ax - px) - abs(ay - py)]
    reward = [-abs(25 - ax) - abs(25 - ay)]
    return reward

def format_state(state):
    ax = state[0,0].numpy()
    ay = state[0,1].numpy()
    px = state[0,2].numpy()
    py = state[0,3].numpy()
    return "{}".format(ax) + "/" + "{}".format(ay) + ", " + "{}".format(px) + "/" + "{}".format(py)

def format_step(state, action):
    succ_state = get_next_state(state, action)
    reward = get_reward(state, action)
    return "(" + format_state(state) + " -> " + format_state(succ_state) + ", A: " + "{}".format(action) + ", R:" + "{}".format(reward) + ")"

def visualizefield(state):
    ax = state[0,0].numpy()
    ay = state[0,1].numpy()
    px = state[0,2].numpy()
    py = state[0,3].numpy()
    over = 0
    str = ""
    for yc in range(0,height):
        over = 0
        for xc in range(0, width):
            if xc == ax and yc == ay:
                #statestr = format_state(state)
                str = str + "x" #+ statestr
                #over = len(statestr)
            elif xc == px and yc == py:
                #statestr = format_state(state)
                str = str + "@" #+ statestr
                #over = len(statestr)
            else:
                if over == 0:
                    if yc == 0 or yc == height - 1 or xc == 0 or xc == width - 1:
                        str = str + "#"
                    elif xc == 0 and yc == 0:
                        str = str + "."
                    else:
                        str = str + " "
                else:
                    over = over - 1
        str = str + "\n"
    #os.system('cls')
    sys.stdout.write(str)
    
def update_state_by_user_input(state):
    ax = state[0,0].numpy()
    ay = state[0,1].numpy()
    px = state[0,2].numpy()
    py = state[0,3].numpy()
    if keyboard.is_pressed('a'):
        px = max(px - 1, 0)
    if keyboard.is_pressed('d'):
        px = min(px + 1, width - 1)
    if keyboard.is_pressed('w'):
        py = max(py - 1, 0)
    if keyboard.is_pressed('s'):
        py = min(py + 1, height - 1)
    return tf.constant([[ax,ay,px,py]])
 
if __name__ == "__main__":

    # hyperparameters
    action_dim = 4
    exploration_rate = 0.1
    learning_rate = 0.01
    num_episodes = 1000000
    alpha = 0.1
    gamma = 0.0
    succ_state = tf.constant([[0,0,10,10]])
    state_dim = 4

    # construct q-network
    q_network = construct_q_network(state_dim, action_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    i = 0
    while True:
        i = i + 1
        with tf.GradientTape() as tape:
            # define current state and obtain q values (estimated by NN)
            state = succ_state
            state = update_state_by_user_input(state)
            q_values = q_network(state)

            # choose action
            epsilon = np.random.rand()
            if epsilon <= exploration_rate:
                action = np.random.choice(action_dim)
            else:
                action = np.argmax(q_values)

            # go to successor state and obtain reward
            succ_state = get_next_state(state, action)
            succ_state_q_values = q_network(succ_state)
            reward = get_reward(state, action)

            # visualize
            if i % 100 == 0:
                visualizefield(succ_state)
                sys.stdout.write(format_step(state, action))
                sys.stdout.write("\n")
                sys.stdout.flush()

            # update q-value
            q_value = q_values[0, action]
            new_q_value = q_value + alpha * (reward + gamma * max(succ_state_q_values) - q_value)

            # compute loss value and do NN learn
            loss_value = mean_squared_error_loss(q_value, new_q_value)
            grads = tape.gradient(loss_value[0], q_network.trainable_variables)
            opt.apply_gradients(zip(grads, q_network.trainable_variables))
