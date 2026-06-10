import time
import pickle
import numpy as np
from maze_env import MazeEnv
from sb3_contrib import RecurrentPPO  # Upgraded import
from astar import astar
from bfs import bfs
import matplotlib.pyplot as plt
import matplotlib.colors mcolors
 
def evaluate_rl(env, model, max_steps=500):
    obs, _ = env.reset()
    
    # Recurrent models require initializing hidden LSTM states to zero
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    path = [tuple(env.agent_pos)]
    steps = 0
    total_reward = 0
 
    for _ in range(max_steps):
        # Pass the lstm_states and episode_starts through prediction to maintain memory
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_starts[0] = terminated or truncated
        
        path.append(tuple(env.agent_pos))
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
 
    reached = tuple(env.agent_pos) == env.maze.goal
    return path, steps, total_reward, reached
 
def run_comparison():
    # Load the saved training maze
    with open("models/train_maze.pkl", "rb") as f:
        saved_maze = pickle.load(f)
 
    # Build env and inject the exact training maze
    env = MazeEnv(rows=15, cols=15, wall_prob=0.3, fixed_maze=True)
    env.maze = saved_maze
    
    # Load trained recurrent model
    model = RecurrentPPO.load("models/recurrent_ppo_maze")
 
    # Run Recurrent RL agent
    start = time.time()
    rl_path, rl_steps, rl_reward, rl_reached = evaluate_rl(env, model)
    rl_time = (time.time() - start) * 1000
 
    # Run classical algorithms on the same maze
    start = time.time()
    bfs_path = bfs(saved_maze)
    bfs_time = (time.time() - start) * 1000
 
    start = time.time()
    astar_path = astar(saved_maze)
    astar_time = (time.time() - start) * 1000
 
    # Print comparison table
    print("\n" + "=" * 55)
    print("         ALGORITHM COMPARISON (RECURRENT PPO)")
    print("=" * 55)
    print(f"{'Algorithm':<12} {'Steps':<10} {'Reached Goal':<15} {'Time (ms)'}")
    print("-" * 55)
    print(f"{'RL (PPO)':<12} {rl_steps:<10} {str(rl_reached):<15} {rl_time:.2f}")
    print(f"{'BFS':<12} {len(bfs_path) if bfs_path else 0:<10} {'True':<15} {bfs_time:.2f}")
    print(f"{'A*':<12} {len(astar_path) if astar_path else 0:<10} {'True':<15} {astar_time:.2f}")
    print("=" * 55)
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["RL Agent (Recurrent PPO)", "BFS", "A*"]
    paths = [rl_path, bfs_path, astar_path]
 
    for ax, title, path in zip(axes, titles, paths):
        display = saved_maze.grid.copy().astype(float)
        if path:
            for cell in path:
                display[cell] = 2
        display[saved_maze.start] = 3
        display[saved_maze.goal] = 4
 
        cmap = mcolors.ListedColormap(['white', 'black', 'cornflowerblue', 'green', 'red'])
        ax.imshow(display, cmap=cmap, vmin=0, vmax=4)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
 
    plt.suptitle("RL Agent vs Classical Algorithms", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
 
if __name__ == "__main__":
    run_comparison()