import RLFramework

class ShootingEnvironment(RLFramework.RLEnvironment):
    AC_LEFT = 0
    AC_RIGHT = 1
    AC_UP = 2
    AC_DOWN = 3
    AC_SHOOT = 4
    
    STATE_IDX_X = 0
    STATE_IDX_Y = 1
    STATE_IDX_PX = 2
    STATE_IDX_PY = 3
    
    WIDTH = 50
    HEIGHT = 50
    
    walls = []
    
    # shooting mechanics
    shots = []
    shotdirs = []
    last_agent_non_shoot_action = AC_LEFT

    def __init__(self):
        #self.add_walls()
        pass

    def add_walls(self):
        for x in range(15,30):
            self.walls.append((x, 20))
        for y in range(0,20):
            self.walls.append((20, y))

    def get_state_dim(self):
        return 4
    def get_action_dim(self):
        return 5
    def next(self, state, action):
        # compute next state
        ss = list(state)
        if action == self.AC_LEFT:
            ss[self.STATE_IDX_X] -= 1
            if ss[self.STATE_IDX_X] < 0:
                ss[self.STATE_IDX_X] = 0
        elif action == self.AC_RIGHT:
            ss[self.STATE_IDX_X] += 1
            if ss[self.STATE_IDX_X] >= self.WIDTH:
                ss[self.STATE_IDX_X] = self.WIDTH - 1
        elif action == self.AC_UP:
            ss[self.STATE_IDX_Y] -= 1
            if ss[self.STATE_IDX_Y] < 0:
                ss[self.STATE_IDX_Y] = 0
        elif action == self.AC_DOWN:
            ss[self.STATE_IDX_Y] += 1
            if ss[self.STATE_IDX_Y] >= self.HEIGHT:
                ss[self.STATE_IDX_Y] = self.HEIGHT - 1
        # do not cross walls
        if (ss[self.STATE_IDX_X], ss[self.STATE_IDX_Y]) in self.walls:
            ss = list(state)

        # compute reward
        #reward = (max(self.WIDTH, self.HEIGHT) - max(abs(self.WIDTH / 2 - ss[self.STATE_IDX_X]), abs(self.HEIGHT / 2 - ss[self.STATE_IDX_Y])))  # stay centered
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(ss[self.STATE_IDX_PX] - ss[self.STATE_IDX_X]), abs(ss[self.STATE_IDX_PY] - ss[self.STATE_IDX_Y])))  # stay with other player

        # additional costs for shoots and reward for hits
        if action == self.AC_SHOOT:
            reward -= 20
        for idx in range(len(self.shots) - 1, -1, -1):
            if self.does_hit(state, idx):
                reward += 1000000

        return (ss, reward)

    def does_hit(self, state, shot_idx):
        hit_radius = 5
        (x, y) = self.shots[shot_idx]
        return abs(x - state[self.STATE_IDX_PX]) < hit_radius and abs(y- state[self.STATE_IDX_PY]) < hit_radius
        
    def get_start_state(self):
        return [self.WIDTH / 2, self.HEIGHT / 2, 0, 0]
    
    def action_to_char(self, action):
        if action == self.AC_LEFT:
            return "<"
        elif action == self.AC_RIGHT:
            return ">"
        elif action == self.AC_UP:
            return "^"
        elif action == self.AC_DOWN:
            return "v"
        elif action == self.AC_SHOOT:
            return " "
        return " "
    
    def visualize(self, state, rlframework):
        # print field
        print_density = 5
        out = "Current state:\n"
        for y in range(50):
            for x in range(50):
                if x == state[self.STATE_IDX_X] and y == state[self.STATE_IDX_Y]:
                    out += "X"
                elif x == state[self.STATE_IDX_PX] and y == state[self.STATE_IDX_PY]:
                    out += "O"
                else:
                    if (x,y) in self.walls:
                        out += "#"
                    elif (x,y) in self.shots:
                        out += "*"
                    elif x % print_density == 0 and y % print_density == 0:
                        #action = -1
                        #action = rlframework.get_action([x, y])
                        action = rlframework.get_action([x, y, state[self.STATE_IDX_PX], state[self.STATE_IDX_PY]])
                        out += str(self.action_to_char(action))
                    else:
                        out += " "
            out += "\n"
        print(out)
        print(rlframework.get_stats())

    def agent_shooting(self, state, action):
        if action == self.AC_SHOOT:
            self.shots.append((state[self.STATE_IDX_X], state[self.STATE_IDX_Y]))
            self.shotdirs.append(self.last_agent_non_shoot_action)
        else:
            self.last_agent_non_shoot_action = action
        for idx in range(len(self.shots) - 1, -1, -1):
            if self.does_hit(state, idx):
                delete = True
            elif self.shots[idx] in self.walls:
                delete = True
            else:
                (x, y) = self.shots[idx]
                d = self.shotdirs[idx]
                delete = False
                if d == self.AC_LEFT:
                    x -= 1
                    if x < 0:
                        delete = True
                elif d == self.AC_RIGHT:
                    x += 1
                    if x >= self.WIDTH:
                        delete = True
                elif d == self.AC_UP:
                    y -= 1
                    if y < 0:
                        delete = True
                elif d == self.AC_DOWN:
                    y += 1
                    if y >= self.HEIGHT:
                        delete = True
                self.shots[idx]= (x, y)
            if delete:
                del self.shots[idx]
                del self.shotdirs[idx]
        return state

    def move_second_player(self, state, action):
        # move second player around
        if state[self.STATE_IDX_PX] == 0:
            # go up at left edge
            if state[self.STATE_IDX_PY] > 0:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] - 1
            else:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] + 1
        elif state[self.STATE_IDX_PY] == 0:
            # go right at top edge
            if state[self.STATE_IDX_PX] < self.WIDTH - 1:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] + 1
            else:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] + 1
        elif state[self.STATE_IDX_PX] == self.WIDTH - 1:
            # go down at right edge
            if state[self.STATE_IDX_PY] < self.HEIGHT - 1:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] + 1
            else:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] - 1
        elif state[self.STATE_IDX_PY] == self.HEIGHT - 1:
            # go left at bottom edge
            if state[self.STATE_IDX_PX] > 0:
                state[self.STATE_IDX_PX] = state[self.STATE_IDX_PX] - 1
            else:
                state[self.STATE_IDX_PY] = state[self.STATE_IDX_PY] - 1
        return state

    def environment_change(self, state, action):
        state = self.agent_shooting(state, action)
        state = self.move_second_player(state, action)
        return state

if __name__ == "__main__":
    RLFramework.RLTrainer(ShootingEnvironment()).train()