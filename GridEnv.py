from typing import Any, Optional
import gym
import numpy as np
from gym import Env, spaces


# Grid-world: A rover travels in a fixed square region. It
# starts from the top left corner and its destination is the
# top right corner. The rover gets a negative reward for each
# step it moves. There are fixed holes in the region. If the
# rover falls into a hole, the trip terminates.
#
# The constraint is on the possibility of the rover falling into a hole.

# 这个如何建模 ???


# Mars Rover with mean valued constraints.

def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Env:
    # create GridWorld environment from id
    if id == "GridWorldEnv":
        env = GridWorldEnv(env_name=id, nrow=10, ncol=10)
    elif id == "GridWorldEnvRandomGoal":
        env = GridWorldEnv(env_name=id, nrow=10, ncol=10)
    return env


class GridWorldEnv(gym.Env):
    def __init__(self, env_name, nrow=5, ncol=5 , holes = [], mines = []):
        self.env_name = env_name
        self.nrow = nrow
        self.ncol = ncol
        self.goal = np.array([ nrow-1 , 0])
        self.curr_pos = np.array([0, 0])
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.nrow - 1, self.ncol - 1, self.nrow - 1, self.ncol - 1]),
            dtype=int,
        )  # current position and target position
        self.action_space = spaces.Discrete(
            5
        )  # action [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]

        self.holes = []
        for (posi_x , posi_y) in holes:
            if posi_x == self.goal[0] and posi_x == self.goal[1] :
                continue

            self.holes.append( (posi_x , posi_y) )
        self.mines = []
        for (posi_x , posi_y) in mines:
            self.mines.append( (posi_x , posi_y) )
        self.steps = 0


    def step(self, action):
        origin_state = (self.curr_pos[0] , self.curr_pos[1])
        if action == 0:  # stay
            pass
        elif action == 1:  # left
            self.curr_pos[0] -= 1
        elif action == 2:  # right
            self.curr_pos[0] += 1
        elif action == 3:  # up
            self.curr_pos[1] -= 1
        elif action == 4:  # down
            self.curr_pos[1] += 1
        else:
            raise ValueError("Invalid action!")

        self.curr_pos = np.clip(
            self.curr_pos,
            a_min=np.array([0, 0]),
            a_max=np.array([self.nrow - 1, self.ncol - 1]),
        )

        obs = np.concatenate((self.curr_pos, self.goal)) # why + self.goal, 增加决策空间 ?
        reward = 0
        done = False
        constrain_dict = {'hole':0 ,'find_optimal':0 } # {'hole':0 , 'mine':0}

        if (self.curr_pos == self.goal).all():
            reward += 3 # 10
            done = True
            constrain_dict['find_optimal'] = 1 # win

        elif ( tuple(self.curr_pos) in self.holes ):
            reward += 0 # TODO
            constrain_dict['hole'] = 1 # game over
            # print('hit hole')
            done = True
        elif ( tuple(self.curr_pos) in self.mines ):
            reward -= 2 # TODO , 可与特定动作结合
            print('barrier Or Track')
        else:
            reward -= 1 # 每步 -1

        # # 给原地不动额外惩罚
        # if (origin_state == self.curr_pos ).all():
        #     reward -= 1

        if self.steps == int(self.ncol * 500): # 绕了2圈
            done = True
            reward -= 10
        else:
            self.steps += 1

        return obs, reward, done, False, constrain_dict

    def reset(self, seed=None, options=None ):
        self.steps = 0
        if seed:
            np.random.seed(seed)
            while True:
                self.curr_pos = np.random.randint(low=[0, 0], high=[self.nrow, self.ncol])
                if not (self.curr_pos == self.goal).all() and not (len(self.holes) > 0 and tuple(self.curr_pos) in self.holes):
                    obs = np.concatenate((self.curr_pos, self.goal))
                    return obs, {}
        else:
            self.curr_pos = np.array([0, 0])
            obs = np.concatenate((self.curr_pos, self.goal))
            return obs, {}

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    N_GRID = 10
    # generate holes
    np.random.seed(42)
    holes = []
    for _ in range( N_GRID**2 // 10 ):
        holes.append( np.random.randint(low=[0, 0], high=[N_GRID , N_GRID] ) )
    # generate barrier

    env = GridWorldEnv(env_name="GridWorldEnv",ncol=N_GRID ,nrow=N_GRID , holes=holes)
    obs, _ = env.reset(seed=0)
    print(env.curr_pos)
    while True:
        action = np.random.randint(0, 5)
        obs, reward, done, _, info = env.step(action)
        print("action: ", action)
        print("obs: ", obs, "reward: ", reward, "done: ", done)
        if done:
            break