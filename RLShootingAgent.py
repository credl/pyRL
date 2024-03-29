import tensorflow as tf
import keras
import RLFramework
import MyConsole
import itertools
import numpy as np
from collections import deque
import argparse

cons = MyConsole.MyConsole()

class ShootingEnvironment(RLFramework.RLEnvironment):
    inpkey = 0
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_SHOOT: int = 4
    WIDTH: int = 10; HEIGHT: int = 10
    agent_x: int = 0; agent_y: int = 0; player_x: int = 0; player_y: int = 0
    player_circular_movement = False; player_dir: int = 0; player_change_interval: int = 10; player_move_step: int = 0
    walls = []
    shots = []; shotdirs = []; last_agent_non_shoot_action = AC_LEFT
    CAN_SHOOT = True
    nn_dec = None
    viz_print_density = 5
    viz_nn_update_interval = 100
    state_stacking = 1
    prev_states = deque(maxlen=state_stacking)
    overall_hits = []
    overall_misses = []

    def __init__(self):
        # set initial state
        self.agent_x = int(self.WIDTH / 2)
        self.agent_y = int(self.HEIGHT / 2)
        self.player_x = int(self.WIDTH / 2)
        self.player_y = int(self.HEIGHT / 2)
        #self.__init_walls()
        self.walls = [ (0, i) for i in range(self.HEIGHT) ] + [ (self.WIDTH - 1, i) for i in range(self.HEIGHT) ] + [ (i, 0) for i in range(self.WIDTH) ] + [ (i, self.HEIGHT - 1) for i in range(self.WIDTH) ]

    def get_state_dim(self):
        return len(self.__encode_state(self.agent_x, self.agent_y))

    def get_action_dim(self):
        return 5 if self.CAN_SHOOT else 4

    def next(self, action):
        # compute next state
        if action == self.AC_LEFT:
            if not (self.agent_x - 1, self.agent_y) in self.walls: self.agent_x = max(self.agent_x - 1, 0)
        elif action == self.AC_RIGHT:
            if not (self.agent_x + 1, self.agent_y) in self.walls: self.agent_x = min(self.agent_x + 1, self.WIDTH - 1)
        elif action == self.AC_UP:
            if not (self.agent_x, self.agent_y - 1) in self.walls: self.agent_y = max(self.agent_y - 1, 0)
        elif action == self.AC_DOWN:
            if not (self.agent_x, self.agent_y + 1) in self.walls: self.agent_y = min(self.agent_y + 1, self.HEIGHT - 1)
        # compute reward
        reward = 0 #(max(self.WIDTH, self.HEIGHT) - max(abs(self.player_x - self.agent_x), abs(self.player_y - self.agent_y)))  # stay with other player
        # changes to the environment other than agent action
        self.__move_player()
        if self. CAN_SHOOT: reward += self.__shot_movement(action)
        return (self.get_state(), reward)

    def get_state(self):
        return self.__encode_state(self.agent_x, self.agent_y)

    def visualize(self, rlframework, step):
        out = [ [" "] * self.WIDTH for i in range(self.HEIGHT)]
        out[0][0] = out[0][self.WIDTH - 1] = out[self.HEIGHT - 1][0] = out[self.HEIGHT - 1][self.WIDTH - 1] = "+"
        # print shots
        for (x, y) in self.shots: out[y][x] = "*"
        # print network decisions
        self.__update_nn_dec(rlframework, step)
#        for (x, y) in itertools.product(range(0, self.WIDTH, self.viz_print_density), range(0, self.HEIGHT, self.viz_print_density)):
#            if not self.nn_dec is None:
#                out[y][x] = str(self.__action_to_char(self.nn_dec[y][x]))
        for (x, y) in self.walls: out[y][x] = "#"
        # print agent and player
        out[self.agent_y][self.agent_x] = "X"
        if self.last_agent_non_shoot_action == self.AC_LEFT and self.agent_x < self.WIDTH - 1:
            out[self.agent_y][self.agent_x + 1] = "<"
        if self.last_agent_non_shoot_action == self.AC_RIGHT and self.agent_x > 0:
            out[self.agent_y][self.agent_x - 1] = ">"
        if self.last_agent_non_shoot_action == self.AC_UP and self.agent_y < self.HEIGHT - 1:
            out[self.agent_y + 1][self.agent_x] = "^"
        if self.last_agent_non_shoot_action == self.AC_DOWN and self.agent_y > 0:
            out[self.agent_y - 1][self.agent_x] = "v"
        out[self.player_y][self.player_x] = "O"
        # output
        cons.erase()
        if len(self.overall_hits) > 100:
            self.overall_hits = self.overall_hits[len(self.overall_hits) - 100:]
        if len(self.overall_misses) > 100:
            self.overall_misses = self.overall_misses[len(self.overall_misses) - 100:]
        cons.myprint("Current state:\n" + cons.matrix_to_string(out) + rlframework.get_stats() + "\nHits in last 100 steps: " + str(sum(self.overall_hits)) + "\nMisses in last 100 steps: " + str(sum(self.overall_misses)))
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

    def __update_nn_dec(self, rlframework, step):
        if step % self.viz_nn_update_interval > 0: return
        if self.nn_dec == None: self.nn_dec = [ [-1] * self.WIDTH for i in range(self.HEIGHT)]
        if self.state_stacking > 1: return
        coords = list(itertools.product(range(0, self.WIDTH, self.viz_print_density), range(0, self.HEIGHT, self.viz_print_density)))
        states = [ self.__encode_state(x, y) for (x, y) in coords ]
        actions = rlframework.get_actions(states)       
        for ((x, y), a) in zip(coords, actions): self.nn_dec[y][x] = a

    def __init_walls(self):
        for x in range(15,30): self.walls.append((x, 20))
        for y in range(0,20): self.walls.append((20, y))

    def __action_to_char(self, action):
        if action == self.AC_LEFT:      return "<"
        elif action == self.AC_RIGHT:   return ">"
        elif action == self.AC_UP:      return "^"
        elif action == self.AC_DOWN:    return "v"
        elif action == self.AC_SHOOT:   return "."
        else:                           return " "

    def __move_player(self):
        if self.player_circular_movement:   self.__move_player_circular()
        else:                               self.__move_player_random()

    def __move_player_circular(self):
        if self.player_x == 0:
            # go up at left edge
            if self.player_y > 0:               self.player_y -= 1
            else:                               self.player_x += 1
        elif self.player_y == 0:
            # go right at top edge
            if self.player_x < self.WIDTH - 1:  self.player_x += 1
            else:                               self.player_y += 1
        elif self.player_x == self.WIDTH - 1:
            # go down at right edge
            if self.player_y < self.HEIGHT - 1: self.player_y += 1
            else:                               self.player_x -= 1
        elif self.player_y == self.HEIGHT - 1:
            # go left at bottom edge
            if self.player_x > 0:               self.player_x -= 1
            else:                               self.player_y -= 1

    def __move_player_random(self):
        x = self.player_x
        y = self.player_y
        if self.player_move_step % self.player_change_interval == 0: self.player_dir = np.random.choice(4)
        if self.player_dir == self.AC_LEFT:         self.player_x = max(self.player_x - 1, 0)
        elif self.player_dir == self.AC_RIGHT:      self.player_x = min(self.player_x + 1, self.WIDTH - 1)
        elif self.player_dir == self.AC_UP:         self.player_y = max(self.player_y - 1, 0)
        elif self.player_dir == self.AC_DOWN:       self.player_y = min(self.player_y + 1, self.HEIGHT - 1)
        if (self.player_x, self.player_y) in self.walls:
            self.player_x = x
            self.player_y = y
        self.player_move_step += 1

    def __shot_movement(self, action):
        add_reward = 0
        if action == self.AC_SHOOT:
            self.shots.append((self.agent_x, self.agent_y))
            self.shotdirs.append(self.last_agent_non_shoot_action)
            # no unnecessay shots
            add_reward += -1
        else:
            self.last_agent_non_shoot_action = action
        hits = 0
        misses = 0
        for idx in range(len(self.shots) - 1, -1, -1):
            if self.__does_hit(idx):
                delete = True
                add_reward += 10
                hits += 1
            elif self.shots[idx] in self.walls:
                add_reward += -2
                delete = True
                misses += 1
            else:
                (x, y) = self.shots[idx]
                d = self.shotdirs[idx]
                delete = False
                if d == self.AC_LEFT:
                    x -= 1
                    if x < 0: delete = True
                elif d == self.AC_RIGHT:
                    x += 1
                    if x >= self.WIDTH: delete = True
                elif d == self.AC_UP:
                    y -= 1
                    if y < 0: delete = True
                elif d == self.AC_DOWN:
                    y += 1
                    if y >= self.HEIGHT: delete = True
                self.shots[idx]= (x, y)
            if delete: del self.shots[idx]; del self.shotdirs[idx]
        self.overall_hits += [ hits ]
        self.overall_misses += [ misses ]
        return add_reward

    def __does_hit(self, shot_idx):
        hit_radius = 1
        (x, y) = self.shots[shot_idx]
        return abs(x - self.player_x) <= hit_radius and abs(y - self.player_y) <= hit_radius

    def __encode_state(self, agent_x, agent_y):
        state = self.__encode_state_complex_ndim()
        if self.state_stacking > 1:
            self.prev_states.append(state)
            if len(self.prev_states) == self.state_stacking:
                return list(self.prev_states)
            else:
                return [ state for i in range(self.state_stacking) ]
        else:
            return state

    def __encode_state_simple(self):
        # simple encoding of just agent and player positions
        return [self.agent_x, self.agent_y, self.player_x, self.player_y]
    
    def __encode_state_complex_1dim(self):
        # complex encoding of the whole field
        state = [0] * self.WIDTH * self.HEIGHT
        state[self.__coord_to_idx(self.agent_x, self.agent_y)] = 1
        state[self.__coord_to_idx(self.player_x, self.player_y)] = 2
        for (x, y) in self.shots: state[self.__coord_to_idx(x, y)] = 3
        return state
        
    def __encode_state_complex_ndim(self):
        # complex encoding of the whole field (multidimensional)
        state = [ [ [0.0, 0.0, 0.0] for j in range(self.HEIGHT) ] for i in range(self.WIDTH)]
        state[self.agent_y][self.agent_x] = [255, 255, 255]
        if self.last_agent_non_shoot_action == self.AC_LEFT and self.agent_x < self.WIDTH - 1:
            state[self.agent_y][self.agent_x + 1] = [255, 255, 0]
        if self.last_agent_non_shoot_action == self.AC_RIGHT and self.agent_x > 0:
            state[self.agent_y][self.agent_x - 1] = [255, 255, 0]
        if self.last_agent_non_shoot_action == self.AC_UP and self.agent_y < self.HEIGHT - 1:
            state[self.agent_y + 1][self.agent_x] = [255, 255, 0]
        if self.last_agent_non_shoot_action == self.AC_DOWN and self.agent_y > 0:
            state[self.agent_y - 1][self.agent_x] = [255, 255, 0]
        state[self.player_y][self.player_x] = [0, 0, 255]
        for (x, y) in self.walls:
            state[y][x] = [255, 0, 0]
        for (x, y) in self.shots:
            state[y][x] = [0, 255, 0]
        return state

    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Agent.')
    parser.add_argument('--nnfile', help='file to load and save the network')
    nnfile = parser.parse_args().nnfile
    if nnfile == None:  nnfile = "./RLShootingAgent_trained.h5"
    else:               nnfile = "./" + nnfile.split("=")[0]
    env = ShootingEnvironment()
    net = keras.models.Sequential([
                keras.layers.Conv2D(10, 2, 1, padding='same', activation="leaky_relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(10, activation="leaky_relu"),
                keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
            ])
    tr = RLFramework.RLTrainer(env, nn=net, visualize_interval=1, load_path=nnfile, save_path=nnfile, exploration_rate_start=0.99, exploration_rate_decrease=0.001, exploration_rate_min=0.05, save_interval=100, gamma_discout_factor=0.99, nn_learning_rate=0.001, replay_buffer_size=200, sample_size=64, accept_q_network_interval=1) #, exploration_rate_start=1.0, exploration_rate_decrease=0.000001)
    tr.get_action(env.get_state())
    print(env.get_state_dim())
    print("Network stats:\n"  + tr.get_network_stats())
    cons.myprint("Network stats:\n" + tr.get_network_stats())
    cons.refresh()
    tr.train()
    cons.end()