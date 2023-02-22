import RLFramework
import MyConsole
import itertools
import numpy as np

cons = MyConsole.MyConsole()

class CollectingEnvironment(RLFramework.RLEnvironment):
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_SHOOT: int = 4
    WIDTH: int = 25; HEIGHT: int = 25
    agent_x: int = 0; agent_y: int = 0; key_x: int = -1; key_y: int = -1; lock_x = -1; lock_y = -1; coin_x = -1; coin_y = -1
    shots = []; shotdirs = []; last_agent_non_shoot_action = AC_LEFT
    nn_dec = None
    viz_print_density = 5
    viz_nn_update_interval = 1

    def __init__(self):
        # set initial state
        self.agent_x = int(self.WIDTH / 2)
        self.agent_y = int(self.HEIGHT / 2)

    def get_state_dim(self):
        return len(self.__encode_state(self.agent_x, self.agent_y))

    def get_action_dim(self):
        return 4

    def spawn_objects(self):
        if self.coin_x == -1 and self.coin_y == -1:
            self.coin_x = np.random.choice(self.WIDTH)
            self.coin_y = np.random.choice(self.HEIGHT)
        if self.lock_x == -1 and self.lock_y == -1:
            self.key_x = np.random.choice(self.WIDTH)
            self.key_y = np.random.choice(self.HEIGHT)
            self.lock_x = np.random.choice(self.WIDTH)
            self.lock_y = np.random.choice(self.HEIGHT)

    def next(self, action):
        # compute next state
        if action == self.AC_LEFT:      self.agent_x = max(self.agent_x - 1, 0)
        elif action == self.AC_RIGHT:   self.agent_x = min(self.agent_x + 1, self.WIDTH - 1)
        elif action == self.AC_UP:      self.agent_y = max(self.agent_y - 1, 0)
        elif action == self.AC_DOWN:    self.agent_y = min(self.agent_y + 1, self.HEIGHT - 1)
        # compute reward
        reward = 0
        if abs(self.agent_x - self.coin_x) < 5 and abs(self.agent_y - self.coin_y) < 5:
            reward += 10
            self.coin_x = -1
            self.coin_y = -1
        if abs(self.agent_x - self.key_x) < 5 and abs(self.agent_y - self.key_y) < 5:
            reward += 10
            self.key_x = -1
            self.key_y = -1
        if self.key_x == -1 and self.key_y == -1 and abs(self.agent_x - self.lock_x) < 5 and abs(self.agent_y - self.lock_y) < 5:
            reward += 100
            self.lock_x = -1
            self.lock_y = -1
        # changes to the environment other than agent action
        self.spawn_objects()
        return (self.get_state(), reward)

    def get_state(self):
        return self.__encode_state(self.agent_x, self.agent_y)

    def visualize(self, rlframework, step):
        out = [ [" "] * self.WIDTH for i in range(self.HEIGHT)]
        # print agent and player
        out[0][0] = out[0][self.WIDTH - 1] = out[self.HEIGHT - 1][0] = out[self.HEIGHT - 1][self.WIDTH - 1] = "+"
        # print network decisions
        self.__update_nn_dec(rlframework, step)
        for (x, y) in itertools.product(range(0, self.WIDTH, self.viz_print_density), range(0, self.HEIGHT, self.viz_print_density)):
            out[y][x] = str(self.__action_to_char(self.nn_dec[y][x]))
        # print agent and player
        out[self.agent_y][self.agent_x] = "X"
        if self.coin_x != -1 and self.coin_y != -1:
            out[self.coin_y][self.coin_x] = "C"
        if self.key_x != -1 and self.key_x != -1:
            out[self.key_y][self.key_x] = "K"
        if self.lock_y != -1 and self.lock_x != -1:
            out[self.lock_y][self.lock_x] = "L"
        # output
        cons.erase()
        cons.myprint("Current state:\n" + cons.matrix_to_string(out) + rlframework.get_stats())
        cons.refresh()

    def abort(self):
        return (cons.getch() == 27) # 'escape' key

    def __update_nn_dec(self, rlframework, step):
        if step % self.viz_nn_update_interval > 0: return
        if self.nn_dec == None: self.nn_dec = [ [-1] * self.WIDTH for i in range(self.HEIGHT)]
        coords = list(itertools.product(range(0, self.WIDTH, self.viz_print_density), range(0, self.HEIGHT, self.viz_print_density)))
        states = [ self.__encode_state(x, y) for (x, y) in coords ]
        actions = rlframework.get_actions(states)       
        for ((x, y), a) in zip(coords, actions): self.nn_dec[y][x] = a

    def __action_to_char(self, action):
        if action == self.AC_LEFT:      return "<"
        elif action == self.AC_RIGHT:   return ">"
        elif action == self.AC_UP:      return "^"
        elif action == self.AC_DOWN:    return "v"
        else:                           return " "

    def __encode_state(self, agent_pos_x, agent_pos_y):
        # simple encoding of just agent and player positions
        return [agent_pos_x, agent_pos_y, self.coin_x, self.coin_y, self.key_x, self.key_y, self.lock_x, self.lock_y]

    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    env = CollectingEnvironment()
    tr = RLFramework.RLTrainer(env, visualize_interval=1, load_path="./RLChasingAgent_trained.h5", save_path="./RLChasingAgent_trained.h5", save_interval=10)
    tr.get_action(env.get_state())
    print("Network stats:\n"  + tr.get_network_stats())
    cons.myprint("Network stats:\n" + tr.get_network_stats())
    cons.refresh()
    tr.train()
    cons.end()