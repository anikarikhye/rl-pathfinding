import pickle
import numpy as np
from maze_env import MazeEnv
from stable_baselines3 import PPO
 
 

with open("models/train_maze.pkl", "rb") as f:
    saved_maze = pickle.load(f)
 

env = MazeEnv(rows=15, cols=15, wall_prob=0.3, fixed_maze=True)
env.maze = saved_maze
env.agent_pos = list(saved_maze.start)
env.steps = 0
 

model = PPO.load("models/ppo_maze")
 

obs = env._get_obs()
 
print(f"Observation shape : {obs.shape}")   
print(f"Start             : {saved_maze.start}")
print(f"Goal              : {saved_maze.goal}")
print(f"\nFirst 30 steps:")
 
positions = []
for i in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    pos = tuple(env.agent_pos)
    positions.append(pos)
    print(f"  Step {i+1:>2}: pos={pos}  action={action}  reward={reward:.3f}")
    if terminated:
        print("  >> REACHED GOAL!")
        break
 
last10 = positions[-10:]
unique = set(last10)
print(f"\nUnique positions in last 10 steps : {len(unique)}")
if len(unique) <= 3:
    print("CONFIRMED: Agent is stuck in a loop!")
else:
    print("Agent is moving — no loop detected.")
 
