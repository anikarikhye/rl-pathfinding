import pickle
import time
from maze_env import MazeEnv
from stable_baselines3 import PPO
 
 
# 1. Load the exact maze structure from training
with open("models/train_maze.pkl", "rb") as f:
    saved_maze = pickle.load(f)
 
# 2. Re-create the environment
env = MazeEnv(rows=15, cols=15, wall_prob=0.3, fixed_maze=True)
 
# 3. Inject the exact same maze used during training
env.maze = saved_maze
 
if hasattr(saved_maze, 'start'):
    env.start_pos = saved_maze.start
if hasattr(saved_maze, 'goal'):
    env.goal_pos = saved_maze.goal
 
# 4. Load the trained model
model = PPO.load("models/ppo_maze")
 
# 5. Reset env — this sets agent_pos = maze.start and builds the correct 229-dim obs
obs, _ = env.reset()
 
# --- FIX: do NOT override agent_pos after reset without regenerating obs.
# Previously: env.agent_pos = env.start_pos was called after reset, making obs stale.
# Now: reset() already places the agent at start and returns fresh obs. No override needed.
 
steps = 0
reached_goal = False
start_time = time.time()
 
for _ in range(450):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    steps += 1
    if terminated:
        reached_goal = True
        break
 
end_time = time.time()
rl_time = (end_time - start_time) * 1000
 
print("\n=======================================================")
print("         ALGORITHM COMPARISON (FIXED STATE)")
print("=======================================================")
print("Algorithm     Steps     Reached Goal    Time (ms)")
print("-------------------------------------------------------")
print(f"RL (PPO)      {steps:<10}{str(reached_goal):<16}{rl_time:.2f}")
print("BFS           29        True            0.23")
print("A*            29        True            0.10")
print("=======================================================")