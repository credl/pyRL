import RLFramework

class FollowingEnvironment(RLFramework.RLEnvironment):
    AC_LEFT: int = 0; AC_RIGHT: int = 1; AC_UP: int = 2; AC_DOWN: int = 3; AC_SHOOT: int = 4
    WIDTH: int = 50; HEIGHT: int = 50
    agent_x: int = 0; agent_y: int = 0; player_x: int = 0; player_y: int = 0
    shots = []; shotdirs = []; last_agent_non_shoot_action = AC_LEFT

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
        if action == self.AC_LEFT: self.agent_x = max(self.agent_x - 1, 0)
        elif action == self.AC_RIGHT: self.agent_x = min(self.agent_x + 1, self.WIDTH - 1)
        elif action == self.AC_UP: self.agent_y = max(self.agent_y - 1, 0)
        elif action == self.AC_DOWN: self.agent_y = min(self.agent_y + 1, self.HEIGHT - 1)
        # compute reward
        reward = (max(self.WIDTH, self.HEIGHT) - max(abs(self.player_x - self.agent_x), abs(self.player_y - self.agent_y)))  # stay with other player
        # changes to the environment other than agent action
        self.__move_player()
        return (self.get_state(), reward)

    def get_state(self):
        return self.__encode_state(self.agent_x, self.agent_y)
#        return [self.agent_x, self.agent_y, self.player_x, self.player_y]

    def visualize(self, rlframework):
        # print field
        print_density = 5
        out = "Current state:\n"
        for y in range(50):
            for x in range(50):
                if (x, y) == (self.agent_x, self.agent_y): out += "X"
                elif (x, y) == (self.player_x, self.player_y): out += "O"
                else:
                    if x % print_density == 0 and y % print_density == 0:
                        out += str(self.__action_to_char(rlframework.get_action(self.__encode_state(x, y))))
                    else: out += " "
            out += "\n"
        print(out, "\n", rlframework.get_stats())

    def __action_to_char(self, action):
        if action == self.AC_LEFT: return "<"
        elif action == self.AC_RIGHT: return ">"
        elif action == self.AC_UP: return "^"
        elif action == self.AC_DOWN: return "v"
        else: return " "

    def __move_player(self):
        # move second player around
        if self.player_x == 0:
            # go up at left edge
            if self.player_y > 0: self.player_y -= 1
            else: self.player_x += 1
        elif self.player_y == 0:
            # go right at top edge
            if self.player_x < self.WIDTH - 1: self.player_x += 1
            else: self.player_y += 1
        elif self.player_x == self.WIDTH - 1:
            # go down at right edge
            if self.player_y < self.HEIGHT - 1: self.player_y += 1
            else: self.player_x -= 1
        elif self.player_y == self.HEIGHT - 1:
            # go left at bottom edge
            if self.player_x > 0: self.player_x -= 1
            else: self.player_y -= 1

    def __encode_state(self, agent_pos_x, agent_pos_y):
        # simple encoding of just agent and player positions
        return [agent_pos_x, agent_pos_y, self.player_x, self.player_y]

    def __coord_to_idx(self, x: int, y: int):
        return y * self.WIDTH + x

if __name__ == "__main__":
    RLFramework.RLTrainer(FollowingEnvironment()).train()