import numpy as np
import gymnasium as gym
from gymnasium import spaces
from maze import Maze
 
 
class MazeEnv(gym.Env):
    def __init__(self, rows=15, cols=15, wall_prob=0.3, fixed_maze=False):
        super(MazeEnv, self).__init__()
        self.rows = rows
        self.cols = cols
        self.wall_prob = wall_prob
        self.fixed_maze = fixed_maze
 
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(rows * cols + 4,),
            dtype=np.float32
        )
 
        self.maze = Maze(self.rows, self.cols, self.wall_prob)
        self.agent_pos = None
        self.steps = 0
        self.max_steps = rows * cols * 2
 
    def _get_obs(self):
        flat_grid = self.maze.grid.flatten().astype(np.float32)
        pos  = np.array(self.agent_pos,   dtype=np.float32) / self.rows
        goal = np.array(self.maze.goal,   dtype=np.float32) / self.rows
        return np.concatenate([flat_grid, pos, goal])
 
    def reset(self, seed=None, options=None):
        if not self.fixed_maze:
            self.maze = Maze(self.rows, self.cols, self.wall_prob)
        self.agent_pos = list(self.maze.start)
        self.steps = 0
        return self._get_obs(), {}
 
    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc
 
        hit_wall = False
        if (0 <= new_r < self.rows and
                0 <= new_c < self.cols and
                self.maze.grid[new_r][new_c] == 0):
            self.agent_pos = [new_r, new_c]
        else:
            hit_wall = True  
 
        self.steps += 1
        reached_goal = tuple(self.agent_pos) == self.maze.goal
        timeout = self.steps >= self.max_steps
 
        if reached_goal:
            reward = 200.0                          
        elif hit_wall:
            reward = -1.0                          
        else:
            
            dist = abs(self.agent_pos[0] - self.maze.goal[0]) + \
                   abs(self.agent_pos[1] - self.maze.goal[1])
            max_dist = self.rows + self.cols
            reward = -0.05 + (1.0 - dist / max_dist) * 0.5
 
        return self._get_obs(), reward, reached_goal, timeout, {}
 
    def render(self):
        self.maze.render(path=[tuple(self.agent_pos)])
 
