from gym.envs.toy_text import discrete
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
PICK = 4
USE = 5


# TODO DEBUG
class KerDoorEnv(discrete.DiscreteEnv):
    def __init__(self):
        """
        1 key and 2 doors
        :param map_size: (width x height)
        """
        key_num = 1
        door_num = 2
        self.map_size = (6, 6)  # (x, y)
        nS = self.map_size[0] * self.map_size[1] * 4   # agent has key or not x which door is open or not
        nA = 4 + 1 + 1  # left down right up + pick a key + use the key
        # no key & doors closed, key & door closed, no key & door1, no key & door2

        self.hasHey = False
        self.door1Open = False
        self.door2Open = False

        self.key_pos = [0, 4]
        self.door1_pos = [2, 1]
        self.door2_pos = [2, 4]
        self.jewel1_pos = [5, 2]
        self.jewel2_pos = [5, 5]

        self.flags = {"none": 0, "hasKey": 1, "door1Open": 2, "door2Open": 3}
        nx, ny = self.map_size

        # TODO define forbidden transition
        # x == 2.5, y をドア開以外で通過不可にする
        self.left_forbidden_s = [self.pos_to_state(f, 2, y) for f in self.flags.values() for y in range(ny)]
        self.right_forbidden_s = [self.pos_to_state(f, 3, y) for f in self.flags.values() for y in range(ny)]
        self.left_forbidden_s.remove(self.pos_to_state(self.flags["door1Open"], *self.door1_pos))
        self.left_forbidden_s.remove(self.pos_to_state(self.flags["door2Open"], *self.door2_pos))
        self.right_forbidden_s.remove(self.pos_to_state(self.flags["door1Open"], *self.door1_pos))
        self.right_forbidden_s.remove(self.pos_to_state(self.flags["door2Open"], *self.door2_pos))

        # x == [3, 5], y == 2.5 を通過禁止にする
        self.down_forbidden_s = [self.pos_to_state(f, x, 2) for f in self.flags.values() for x in np.arange(3, 5)]
        self.up_forbidden_s = [self.pos_to_state(f, x, 3) for f in self.flags.values() for x in np.arange(3, 5)]

        # calc transition table
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for f in sorted(self.flags.values()):
            for x in range(nx):
                for y in range(ny):
                    s = f * nx * ny + y * nx + x
                    for a in range(nA):
                        s_r_done = self.inc(f, x, y, a)
                        p_s_r_done = [1.0] + s_r_done  # (probability, next_state, reward, done), list addition
                        P[s][a].append(p_s_r_done)

        initial_state_distribution = np.zeros(nS)
        initial_state_distribution[0] = 1.0
        super(KerDoorEnv, self).__init__(nS, nA, P, initial_state_distribution)

    def decode_state(self):
        _dict = {"none": 0, "hasKey": 1, "door1Open": 2, "door2Open": 3}
        if not self.hasHey and not self.door1Open and not self.door2Open:
            mode = "none"
        elif self.hasHey and not self.door1Open and not self.door2Open:
            mode = "hasKey"
        elif not self.hasHey and self.door1Open and not self.door2Open:
            mode = "door1Open"
        elif not self.hasHey and not self.door1Open and self.door2Open:
            mode = "door2Open"
        else:
            raise AssertionError
        return _dict[mode]

    def pos_to_state(self, flag, x, y):
        nx, ny = self.map_size
        coord = y * nx + x
        s = flag * nx * ny + coord
        return s

    def state_to_pos(self, s):
        nx, ny = self.map_size
        flag = int(s / nx / ny)
        coord_s = s % (nx * ny)
        y = int(coord_s / nx)
        x = coord_s % nx
        return flag, x, y

    def inc(self, flag, y, x, a):
        done = False
        nx, ny = self.map_size
        coord = y * nx + x
        s = flag * nx * ny + coord
        rew = 0.
        if a == LEFT:
            if s not in self.left_forbidden_s:
                x = max(x - 1, 0)
        elif a == DOWN:
            if s not in self.down_forbidden_s:
                y = min(y + 1, nx - 1)
        elif a == RIGHT:
            if s not in self.right_forbidden_s:
                x = min(x + 1, ny - 1)
        elif a == UP:
            if s not in self.up_forbidden_s:
                y = max(y - 1, 0)
        elif a == PICK:
            if s == self.pos_to_state(flag, *self.key_pos):
                flag = self.flags["hasKey"]
                rew = 1.0
        elif a == USE:
            if s == self.pos_to_state(self.flags["hasKey"], *self.door1_pos):
                flag = self.flags["door1Open"]
                rew = 2.0
            elif s == self.pos_to_state(self.flags["hasKey"], *self.door2_pos):
                flag = self.flags["door2Open"]
                rew = 2.0

        # jewel reward
        if s in [self.pos_to_state(f, *self.jewel1_pos) for f in self.flags]:
            rew = 10.0
            done = True
        elif s in [self.pos_to_state(f, *self.jewel2_pos) for f in self.flags]:
            rew = 5.
            done = True
        s = self.pos_to_state(flag, x, y)
        return [s, rew, done]

    def reset(self):
        self.s = 0
        self.lastaction = None
        return self.s