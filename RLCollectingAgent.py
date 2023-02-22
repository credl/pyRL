import keras
import RLFramework
import MyConsole
import itertools
import numpy as np

cons = MyConsole.MyConsole()

class CollectingEnvironment(RLFramework.RLEnvironment):
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_SHOOT: int = 4
    WIDTH: int = 25; HEIGHT: int = 25
    agent_x: int = 0; agent_y: int = 0; key_x: int = -1; key_y: int = -1; lock_x = -1; lock_y = -1; coin_x = -1; coin_y = -1
    spawn_complex_objects = False
    resetcoin_steps = 1000
    coincounter = 0
    points = []
    pointcount = 20
    nn_dec = None
    viz_print_density = 5
    viz_nn_update_interval = 1

    def __init__(self):
        # set initial state
        self.agent_x = int(self.WIDTH / 2)
        self.agent_y = int(self.HEIGHT / 2)

    def get_state_dim(self):
        return len(self.__encode_state())

    def get_action_dim(self):
        return 4

    def spawn_objects(self):
        if self.spawn_complex_objects:
            self.coincounter += 1
            if self.coincounter >= self.resetcoin_steps or (self.coin_x == -1 and self.coin_y == -1):
                self.coin_x = np.random.choice(self.WIDTH)
                self.coin_y = np.random.choice(self.HEIGHT)
                self.coincounter = 0
            if self.lock_x == -1 and self.lock_y == -1:
                self.key_x = np.random.choice(self.WIDTH)
                self.key_y = np.random.choice(self.HEIGHT)
                self.lock_x = np.random.choice(self.WIDTH)
                self.lock_y = np.random.choice(self.HEIGHT)
        if len(self.points) < self.pointcount:
            self.points += [ (np.random.choice(self.WIDTH), np.random.choice(self.HEIGHT)), ]

    def next(self, action):
        # compute next state
        if action == self.AC_LEFT:      self.agent_x = max(self.agent_x - 1, 0)
        elif action == self.AC_RIGHT:   self.agent_x = min(self.agent_x + 1, self.WIDTH - 1)
        elif action == self.AC_UP:      self.agent_y = max(self.agent_y - 1, 0)
        elif action == self.AC_DOWN:    self.agent_y = min(self.agent_y + 1, self.HEIGHT - 1)
        # compute reward
        reward = 0
        if abs(self.agent_x - self.coin_x) < 10 and abs(self.agent_y - self.coin_y) < 10:
            reward += 20
            self.coin_x = -1
            self.coin_y = -1
        if abs(self.agent_x - self.key_x) < 10 and abs(self.agent_y - self.key_y) < 10:
            reward += 30
            self.key_x = -1
            self.key_y = -1
        if self.key_x == -1 and self.key_y == -1 and abs(self.agent_x - self.lock_x) < 10 and abs(self.agent_y - self.lock_y) < 10:
            reward += 100
            self.lock_x = -1
            self.lock_y = -1
        if (self.agent_x, self.agent_y) in self.points:
            reward += 10
            self.points.remove((self.agent_x, self.agent_y))
        # changes to the environment other than agent action
        self.spawn_objects()
        return (self.get_state(), reward)

    def get_state(self):
        return self.__encode_state()

    def visualize(self, rlframework, step):
        out = [ [" "] * self.WIDTH for i in range(self.HEIGHT)]
        # print agent and player
        out[0][0] = out[0][self.WIDTH - 1] = out[self.HEIGHT - 1][0] = out[self.HEIGHT - 1][self.WIDTH - 1] = "+"
        # print agent and player
        out[self.agent_y][self.agent_x] = "X"
        if self.coin_x != -1 and self.coin_y != -1:
            out[self.coin_y][self.coin_x] = "C"
        if self.key_x != -1 and self.key_x != -1:
            out[self.key_y][self.key_x] = "K"
        if self.lock_y != -1 and self.lock_x != -1:
            out[self.lock_y][self.lock_x] = "L"
        for (x,y) in self.points:
            out[y][x] = "."
        # output
        cons.erase()
        cons.myprint("Current state:\n" + cons.matrix_to_string(out) + rlframework.get_stats())
        cons.refresh()

    def abort(self):
        return (cons.getch() == 27) # 'escape' key

    def __action_to_char(self, action):
        if action == self.AC_LEFT:      return "<"
        elif action == self.AC_RIGHT:   return ">"
        elif action == self.AC_UP:      return "^"
        elif action == self.AC_DOWN:    return "v"
        else:                           return " "

    def __encode_state(self):
        # simple encoding of just agent and player positions
        return self.__encode_state_complex_1dim()


    def __encode_state_simple(self):
        # simple encoding of just agent and player positions
        return [self.agent_x, self.agent_y, self.coin_x, self.coin_y, self.key_x, self.key_y, self.lock_x, self.lock_y]
    
    def __encode_state_complex_1dim(self):
        # complex encoding of the whole field
        state = [ [0.0, 0.0, 0.0, 0.0, 0.0] for j in range(self.HEIGHT * self.WIDTH)]
        state[self.__coord_to_idx(self.agent_x, self.agent_y)][0] = 1.0
        if self.spawn_complex_objects:
            state[self.__coord_to_idx(self.coin_x, self.coin_y)][1] = 1.0
            state[self.__coord_to_idx(self.key_x, self.key_y)][2] = 1.0
            state[self.__coord_to_idx(self.lock_x, self.lock_y)][3] = 1.0
        for (x,y) in self.points:
            state[self.__coord_to_idx(x,y)][4] = 1.0
        return state
        
    def __encode_state_complex_ndim(self):
        # complex encoding of the whole field (multidimensional)
        state = [ [ [0.0, 0.0, 0.0, 0.0, 0.0] for j in range(self.HEIGHT) ] for i in range(self.WIDTH)]
        state[self.agent_y][self.agent_x][0] = 1.0
        if self.spawn_complex_objects:
            state[self.coin_y][self.coin_x][1] = 1.0
            state[self.key_y][self.key_x][2] = 1.0
            state[self.lock_y][self.lock_x][3] = 1.0
        for (x,y) in self.points:
            state[y][x][4] = 1.0
        return state


    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    env = CollectingEnvironment()
    net = keras.models.Sequential([
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu", kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(32, activation="relu", kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
            ])
    tr = RLFramework.RLTrainer(env, nn=net, visualize_interval=1, load_path="./RLCollectingAgent_trained.h5", save_path="./RLCollectingAgent_trained.h5", save_interval=10)
    tr.get_action(env.get_state())
    print("Network stats:\n"  + tr.get_network_stats())
    cons.myprint("Network stats:\n" + tr.get_network_stats())
    cons.refresh()
    tr.train()
    cons.end()