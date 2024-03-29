import keras
import RLFramework
import MyConsole
import itertools
import numpy as np

cons = MyConsole.MyConsole()

class SnakeEnvironment(RLFramework.RLEnvironment):
    inpkey = 0
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_NOTHING = 4
    direction = AC_LEFT
    WIDTH: int = 10; HEIGHT: int = 10
    COLL_RADIUS = 2
    agent_x: int = 0; agent_y: int = 0; coin_x = -1; coin_y = -1
    snakelen: int = 2
    resetcoin_steps = 1000
    coin_steps = 0
    snakeelem = []
    nn_dec = None
    viz_print_density = 5
    viz_nn_update_interval = 1
    overall_rewards = []
    survived_len = []
    survived_steps = []
    step = 0
    walls = []

    def __init__(self):
        # set initial state
        self.agent_x = int(self.WIDTH / 2)
        self.agent_y = int(self.HEIGHT / 2)
        self.walls = [ [False for y in range(self.HEIGHT)] for y in range(self.WIDTH) ]
        for x in range(self.WIDTH):
            self.walls[x][0] = True
            self.walls[x][self.HEIGHT - 1] = True
        for y in range(self.HEIGHT):
            self.walls[0][y] = True
            self.walls[self.WIDTH - 1][y] = True
        #for x in range(3, 6):
        #    self.walls[x][5] = True
            
    def get_state_dim(self):
        return len(self.__encode_state())

    def get_action_dim(self):
        return 5

    def next(self, action):
        self.step += 1
        finished = False
        reward = 0
        # move snake
        if len(self.snakeelem) >= self.snakelen:
            self.snakeelem = self.snakeelem[1:]
        self.snakeelem += [ (self.agent_x, self.agent_y) ]
        # compute next state
        if action != self.AC_NOTHING:
            if self.direction == action: reward -= 1 # penalize action that should be AC_NOTHING
            if action == self.AC_LEFT and action != self.AC_RIGHT: self.direction = action
            if action == self.AC_RIGHT and action != self.AC_LEFT: self.direction = action
            if action == self.AC_UP and action != self.AC_DOWN: self.direction = action
            if action == self.AC_DOWN and action != self.AC_UP: self.direction = action
            if self.direction != action: reward -= 5 # penalize disallowed action
        if self.direction == self.AC_LEFT:      self.agent_x -= 1
        elif self.direction == self.AC_RIGHT:   self.agent_x += 1
        elif self.direction == self.AC_UP:      self.agent_y -= 1
        elif self.direction == self.AC_DOWN:    self.agent_y += 1
        self.coin_steps += 1
        if self.coin_steps >= self.resetcoin_steps:
            self.coin_steps = 0
            self.coin_x = -1
            self.coin_y = -1
            reward -= 50
        # compute reward for collecting coins
        if self.coin_x != -1 and self.coin_y != -1 and abs(self.agent_x - self.coin_x) < self.COLL_RADIUS and abs(self.agent_y - self.coin_y) < self.COLL_RADIUS:
            reward += 100
            finished = False
            self.coin_x = -1
            self.coin_y = -1
            self.snakelen += 1
            self.coin_steps = 0
        self.overall_rewards += [reward]
        # do not bump into walls or snake
        coll = False
        if self.walls[self.agent_x][self.agent_y]:
            coll = True
        if (self.agent_x, self.agent_y) in self.snakeelem:
            coll = True
        if coll:
            reward += -200
            self.agent_x = -1
            self.agent_y = -1
            finished = True
        if self.agent_x == -1 and self.agent_y == -1:
            self.agent_x = 1 + np.random.choice(self.WIDTH - 2)
            self.agent_y = 1 + np.random.choice(self.HEIGHT - 2)
            self.snakeelem = []
            self.survived_len += [self.snakelen]
            self.survived_steps += [self.step]
            self.snakelen = 2
            self.step = 0
        if self.coin_x == -1 and self.coin_y == -1:
            self.coin_x = 1 + np.random.choice(self.WIDTH - 2)
            self.coin_y = 1 + np.random.choice(self.HEIGHT - 2)
        return (self.get_state(), reward, finished)

    def get_state(self):
        return self.__encode_state()

    def visualize(self, rlframework, step):
        out = [ [" "] * self.WIDTH for i in range(self.HEIGHT)]
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                if self.walls[x][y]:
                    out[y][x] = "+"
        # print agent and player
        # print agent and player
        out[self.agent_y][self.agent_x] = "X"
        if self.coin_x != -1 and self.coin_y != -1:
            out[self.coin_y][self.coin_x] = "C"
        for (x,y) in self.snakeelem:
            out[y][x] = "#"
        # output
        cons.erase()
        if len(self.overall_rewards) > 100:
            self.overall_rewards = self.overall_rewards[len(self.overall_rewards) - 100:]
        if len(self.survived_len) > 30:
            self.survived_len = self.survived_len[len(self.survived_len) - 30:]
        if len(self.survived_steps) > 30:
            self.survived_steps = self.survived_steps[len(self.survived_steps) - 30:]
        cons.myprint("Current state:\n" + cons.matrix_to_string(out) + rlframework.get_stats() + "\nRewards in last 100 steps:" + str(sum(self.overall_rewards)) + "\nRewards/steps:" + str(sum(self.overall_rewards) / len(self.overall_rewards)) + "\nSurvived len:" + str(self.survived_len) + "\nSurvived steps:" + str(self.survived_steps))
        cons.refresh()
        
        self.inpkey = cons.getch()
        if self.inpkey == ord('e'):
            if rlframework.exploration_rate > rlframework.SETTING_exploration_rate_min:
                rlframework.exploration_rate = 0
            else:
                rlframework.exploration_rate = rlframework.SETTING_exploration_rate_start
        if self.inpkey == ord('l'):
            rlframework.learn = not rlframework.learn

    def abort(self):
        return (self.inpkey == 27) # 'escape' key

    def __action_to_char(self, action):
        if action == self.AC_LEFT:      return "<"
        elif action == self.AC_RIGHT:   return ">"
        elif action == self.AC_UP:      return "^"
        elif action == self.AC_DOWN:    return "v"
        else:                           return " "

    def __encode_state(self):
        # simple encoding of just agent and player positions
        return self.__encode_state_complex_ndim()

    def __encode_state_simple(self):
        # simple encoding of just agent and player positions
        return [self.agent_x, self.agent_y, self.direction, self.coin_x, self.coin_y]
    
    def __encode_state_complex_1dim(self):
        # complex encoding of the whole field
        state = [ [0.0, 0.0, 0.0] for j in range(self.HEIGHT * self.WIDTH)]
        state[self.__coord_to_idx(self.agent_x, self.agent_y)] = [255, 255, 255]
        state[self.__coord_to_idx(self.coin_x, self.coin_y)] = [255, 255, 0]
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                if self.walls[x][y]:
                    state[self.__coord_to_idx(x,y)] = [255, 0, 0]
        for (x,y) in self.snakeelem:
            state[self.__coord_to_idx(x,y)] = [0, 255, 255]
        return state
        
    def __encode_state_complex_ndim(self):
        # complex encoding of the whole field (multidimensional)
        state = [ [ [0.0, 0.0, 0.0] for j in range(self.HEIGHT) ] for i in range(self.WIDTH)]
        state[self.agent_y][self.agent_x] = [255, 255, 255]
        state[self.coin_y][self.coin_x] = [255, 255, 0]
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                if self.walls[x][y]:
                    state[x][y] = [255, 0, 0]
        for (x,y) in self.snakeelem:
            state[y][x] = [0, 255, 255]
        return state


    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    env = SnakeEnvironment()
    net = keras.models.Sequential([
                keras.layers.Reshape((env.WIDTH, env.HEIGHT, 3), input_shape=(env.WIDTH, env.HEIGHT, 3)),
                keras.layers.Conv2D(10, 2, 1, padding='same', activation="leaky_relu"),
#                keras.layers.Conv2D(32, 2, 1, padding='same', activation="relu"),
#                keras.layers.MaxPooling2D((1, 1), strides=1),
                keras.layers.Flatten(),
                keras.layers.Dense(10, activation="leaky_relu"),
                keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')

#                keras.layers.Flatten(),
#                keras.layers.Dense(300, activation="leaky_relu"),
#                keras.layers.Dense(300, activation="leaky_relu"),
#                keras.layers.Dense(env.get_action_dim())
            ])
    tr = RLFramework.RLTrainer(env, nn=net, visualize_interval=1, load_path="./RLSnakeAgent_trained.h5", save_path="./RLSnakeAgent_trained.h5", exploration_rate_start=0.99, exploration_rate_decrease=0.001, exploration_rate_min=0.05, save_interval=100, gamma_discout_factor=0.99, nn_learning_rate=0.001, replay_buffer_size=200, sample_size=64, accept_q_network_interval=1)
    tr.get_action(env.get_state())
    print("Network stats:\n"  + tr.get_network_stats())
    cons.myprint("Network stats:\n" + tr.get_network_stats())
    cons.refresh()
    tr.train()
    cons.end()