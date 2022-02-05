import RLFramework
import MyConsole
import itertools
import numpy as np

cons = MyConsole.MyConsole()

class FollowingEnvironment(RLFramework.RLEnvironment):
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_SHOOT: int = 4
    WIDTH: int = 50; HEIGHT: int = 50
    agent_x: int = 0; agent_y: int = 0; player_x: int = 0; player_y: int = 0
    player_circular_movement = False; player_dir: int = 0; player_change_interval: int = 10; player_move_step: int = 0
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

    def next(self, action):
        # compute next state
        if action == self.AC_LEFT:      self.agent_x = max(self.agent_x - 1, 0)
        elif action == self.AC_RIGHT:   self.agent_x = min(self.agent_x + 1, self.WIDTH - 1)
        elif action == self.AC_UP:      self.agent_y = max(self.agent_y - 1, 0)
        elif action == self.AC_DOWN:    self.agent_y = min(self.agent_y + 1, self.HEIGHT - 1)
        # compute reward
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(self.player_x - self.agent_x), abs(self.player_y - self.agent_y)))  # stay with other player
        # changes to the environment other than agent action
        self.__move_player()
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
        out[self.player_y][self.player_x] = "O"
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
        if self.player_move_step % self.player_change_interval == 0: self.player_dir = np.random.choice(4)
        if self.player_dir == self.AC_LEFT:         self.player_x = max(self.player_x - 1, 0)
        elif self.player_dir == self.AC_RIGHT:      self.player_x = min(self.player_x + 1, self.WIDTH - 1)
        elif self.player_dir == self.AC_UP:         self.player_y = max(self.player_y - 1, 0)
        elif self.player_dir == self.AC_DOWN:       self.player_y = min(self.player_y + 1, self.HEIGHT - 1)
        self.player_move_step += 1

    def __encode_state(self, agent_pos_x, agent_pos_y):
        # simple encoding of just agent and player positions
        return [agent_pos_x, agent_pos_y, self.player_x, self.player_y]

    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    env = FollowingEnvironment()
    tr = RLFramework.RLTrainer(env, visualize_interval=1, load_path="./RLFollowingAgent_trained.h5", save_path="./RLFollowingAgent_trained.h5", save_interval=10)
    tr.get_action(env.get_state())
    print("Network stats:\n"  + tr.get_network_stats())
    cons.myprint("Network stats:\n" + tr.get_network_stats())
    cons.refresh()
    tr.train()
    cons.end()