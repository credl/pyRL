import tensorflow as tf
import keras
import RLFramework
import MyConsole
import itertools

cons = MyConsole.MyConsole()

class ShootingEnvironment(RLFramework.RLEnvironment):
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_SHOOT: int = 4
    WIDTH: int = 50; HEIGHT: int = 50
    agent_x: int = 0; agent_y: int = 0; player_x: int = 0; player_y: int = 0
    player_circular_movement = True; player_dir: int = 0; player_change_interval: int = 10; player_move_step: int = 0
    walls = []
    shots = []; shotdirs = []; last_agent_non_shoot_action = AC_LEFT
    CAN_SHOOT = False
    nn_dec = None
    viz_print_density = 5
    viz_nn_update_interval = 1

    def __init__(self):
        # set initial state
        self.agent_x = int(self.WIDTH / 2)
        self.agent_y = int(self.HEIGHT / 2)
        #self.__init_walls()

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
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(self.player_x - self.agent_x), abs(self.player_y - self.agent_y)))  # stay with other player
        # changes to the environment other than agent action
        self.__move_player()
        if self. CAN_SHOOT: reward += self.__shot_movement(action)
        return (self.get_state(), reward)

    def get_state(self):
        return self.__encode_state(self.agent_x, self.agent_y)

    def visualize(self, rlframework, step):
        out = [ [" "] * self.WIDTH for i in range(self.HEIGHT)]
        # print shots
        for (x, y) in self.shots: out[y][x] = "*"
        # print network decisions
        self.__update_nn_dec(rlframework, step)
        for (x, y) in itertools.product(range(0, self.WIDTH, self.viz_print_density), range(0, self.HEIGHT, self.viz_print_density)):
            out[y][x] = str(self.__action_to_char(self.nn_dec[y][x]))
        for (x, y) in self.walls: out[y][x] = "#"
        # print agent and player
        out[self.agent_y][self.agent_x] = "X"
        out[self.player_y][self.player_x] = "O"
        out[0][0] = out[0][self.WIDTH - 1] = out[self.HEIGHT - 1][0] = out[self.HEIGHT - 1][self.WIDTH - 1] = "+"
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

    def __init_walls(self):
        for x in range(15,30): self.walls.append((x, 20))
        for y in range(0,20): self.walls.append((20, y))

    def __action_to_char(self, action):
        if action == self.AC_LEFT:      return "<"
        elif action == self.AC_RIGHT:   return ">"
        elif action == self.AC_UP:      return "^"
        elif action == self.AC_DOWN:    return "v"
        elif action == self.AC_SHOOT:   return " "
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

    def __shot_movement(self, action):
        add_reward = 0
        if action == self.AC_SHOOT:
            self.shots.append((self.agent_x, self.agent_y))
            self.shotdirs.append(self.last_agent_non_shoot_action)
            add_reward -= 20
        else:
            self.last_agent_non_shoot_action = action
        for idx in range(len(self.shots) - 1, -1, -1):
            if self.__does_hit(idx):
                delete = True
                add_reward += 1000000
            elif self.shots[idx] in self.walls:
                delete = True
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
        return add_reward

    def __does_hit(self, shot_idx):
        hit_radius = 5
        (x, y) = self.shots[shot_idx]
        return abs(x - self.player_x) <= hit_radius and abs(y- self.player_y) <= hit_radius
    
    def __encode_state(self, agent_pos_x, agent_pos_y):
        return self.__encode_state_complex_ndim(agent_pos_x, agent_pos_y)

    def __encode_state_simple(self, agent_pos_x, agent_pos_y):
        # simple encoding of just agent and player positions
        return [agent_pos_x, agent_pos_y, self.player_x, self.player_y]
    
    def __encode_state_complex_1dim(self, agent_pos_x, agent_pos_y):
        # complex encoding of the whole field
        state = [0] * self.WIDTH * self.HEIGHT
        state[self.__coord_to_idx(agent_pos_x, agent_pos_y)] = 1
        state[self.__coord_to_idx(self.player_x, self.player_y)] = 2
        for (x, y) in self.shots: state[self.__coord_to_idx(x, y)] = 3
        return state
        
    def __encode_state_complex_ndim(self, agent_pos_x, agent_pos_y):
        # complex encoding of the whole field (multidimensional)
        state = [ [0.0] * self.HEIGHT for i in range(self.WIDTH)]
        state[agent_pos_y][agent_pos_x] = 50
        state[self.player_y][self.player_x] = 100
        for (x, y) in self.shots: state[y][x] = 150
        for (x, y) in self.walls: state[y][x] = 200
        return state

    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    env = ShootingEnvironment()
    net = keras.models.Sequential([
                tf.keras.layers.Reshape((env.WIDTH, env.HEIGHT, 1)),
                tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="elu", input_shape=(env.WIDTH, env.HEIGHT)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="elu", input_shape=(env.WIDTH, env.HEIGHT)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="elu", input_shape=(env.get_state_dim(),), kernel_initializer='random_normal', bias_initializer='random_normal'),
                keras.layers.Dense(env.get_action_dim(), activation="linear", kernel_initializer='random_normal', bias_initializer='random_normal')
            ])
    tr = RLFramework.RLTrainer(env, nn=net, visualize_interval=1)
    tr.get_action(env.get_state())
    print("Network stats:\n"  + tr.get_network_stats())
    cons.myprint("Network stats:\n" + tr.get_network_stats())
    cons.refresh()
    tr.train()
    cons.end()