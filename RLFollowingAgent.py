import RLFramework

class FollowingEnvironment(RLFramework.RLEnvironment):
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

    def __init__(self):
        pass

    def get_state_dim(self):
        return 4
    def get_action_dim(self):
        return 4
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

        # compute reward
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(ss[self.STATE_IDX_PX] - ss[self.STATE_IDX_X]), abs(ss[self.STATE_IDX_PY] - ss[self.STATE_IDX_Y])))  # stay with other player

        return (ss, reward)

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
        return " "
    
    def visualize(self, state, rlf):
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
                    if x % print_density == 0 and y % print_density == 0:
                        action = rlf.get_action([x, y, state[self.STATE_IDX_PX], state[self.STATE_IDX_PY]])
                        out += str(self.action_to_char(action))
                    else:
                        out += " "
            out += "\n"
        print(out)
        print(rlf.get_stats())

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
        state = self.move_second_player(state, action)
        return state

if __name__ == "__main__":
    RLFramework.RLTrainer(FollowingEnvironment()).train()