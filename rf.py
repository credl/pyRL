# Needed for training the network
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

# Needed for animation
import matplotlib.pyplot as plt

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
    x = state[0,0].numpy()
    y = state[0,1].numpy()
    if action == 0:
        x = x - 1
    elif action == 1:
        x = x + 1
    elif action == 2:
        y = y - 1
    elif action == 3:
        y = y + 1
    ret = tf.constant([[x,y]])
    return ret

def get_reward(state, action):
    succ_state = get_next_state(state, action)
    x = succ_state[0,0].numpy()
    y = succ_state[0,1].numpy()
    reward = [-abs(0 - x) - abs(0 - y) + - 3 * abs(150 - x) - 3 * abs(150 - y)]
    return reward

def format_state(state):
    x = state[0,0].numpy()
    y = state[0,1].numpy()
    return "{}".format(x) + "/" + "{}".format(y)

def format_step(state, action):
    succ_state = get_next_state(state, action)
    reward = get_reward(state, action)
    return "(" + format_state(state) + " -> " + format_state(succ_state) + ", A: " + "{}".format(action) + ", R:" + "{}".format(reward) + ")"

def visualizefield(state):
    x = state[0,0].numpy()
    y = state[0,1].numpy()
    over = 0
    str = ""
    for yc in range(-20, 20):
        over = 0
        for xc in range(-20, 20):
            if xc == round(x / 10) and yc == round(y / 10):
                statestr = format_state(state)
                str = str + "x" + statestr
                over = len(statestr)
            else:
                if over == 0:
                    if yc == -20 or yc == 19 or xc == -20 or xc == 19:
                        str = str + "#"
                    elif xc == 0 and yc == 0:
                        str = str + "0"
                    else:
                        str = str + " "
                else:
                    over = over - 1
        str = str + "\n"
    #os.system('cls')
    sys.stdout.write(str)

if __name__ == "__main__":
    # Initialize parameters
    state = tf.constant([[1,1]]) # tf.constant([[1]])
    bandits = np.array([0.9, 1.2, 0.7, 1.0])
    state_dim = 2
    action_dim = len(bandits)
    exploration_rate = 0.1
    learning_rate = 0.01
    num_episodes = 1000000
    alpha = 0.9
    gamma = 0.0

    # Construct Q-network
    q_network = construct_q_network(state_dim, action_dim)

    # Define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    succ_state = tf.constant([[0,0]])
    for i in range(num_episodes + 1):
        with tf.GradientTape() as tape:
            # define current state
            state = succ_state
            
            # Obtain Q-values from network
            q_values = q_network(state)

            epsilon = np.random.rand()
            if epsilon <= exploration_rate:
                # Select random action
                action = np.random.choice(len(bandits))
            else:
                # Select action with highest q-value
                action = np.argmax(q_values)

            # go to successor state and obtain reward from bandit
            old_state = state
            succ_state = get_next_state(state, action)

            succ_state_q_values = q_network(succ_state)
            reward = get_reward(state, action)
            
            if i % 1 == 0:
                visualizefield(succ_state)
                sys.stdout.write(format_step(state, action))
                sys.stdout.write("\n")
                sys.stdout.flush()

            # Obtain Q-value
            q_value = q_values[0, action]

            # compute new q value
            new_q_value = q_value - alpha * (reward + gamma * max(succ_state_q_values) - q_value)

            # Compute loss value
            loss_value = mean_squared_error_loss(q_value, new_q_value)

            # Compute gradients
            grads = tape.gradient(loss_value[0], q_network.trainable_variables)

            # Apply gradients to update network weights
            opt.apply_gradients(zip(grads, q_network.trainable_variables))
