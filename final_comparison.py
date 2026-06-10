import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from maze import Maze
from bfs import bfs
from astar import astar
from qlearning import QLearningAgent

with open("models/train_maze.pkl", "rb") as f:
    maze = pickle.load(f)

with open("models/q_agent.pkl", "rb") as f:
    agent = pickle.load(f)

start = time.time()
ql_path = agent.get_path()
ql_time = (time.time() - start) * 1000

start = time.time()
bfs_path = bfs(maze)
bfs_time = (time.time() - start) * 1000

start = time.time()
astar_path = astar(maze)
astar_time = (time.time() - start) * 1000

print("\n" + "=" * 60)
print("         FINAL ALGORITHM COMPARISON")
print("=" * 60)
print(f"{'Algorithm':<15} {'Path Length':<14} {'Reached Goal':<15} {'Time (ms)'}")
print("-" * 60)
print(f"{'Q-Learning':<15} {len(ql_path) if ql_path else 0:<14} {str(ql_path is not None):<15} {ql_time:.3f}")
print(f"{'BFS':<15} {len(bfs_path) if bfs_path else 0:<14} {'True':<15} {bfs_time:.3f}")
print(f"{'A*':<15} {len(astar_path) if astar_path else 0:<14} {'True':<15} {astar_time:.3f}")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Q-Learning", "BFS", "A*"]
paths = [ql_path, bfs_path, astar_path]

for ax, title, path in zip(axes, titles, paths):
    display = maze.grid.copy().astype(float)
    if path:
        for cell in path:
            display[cell] = 2
    display[maze.start] = 3
    display[maze.goal] = 4
    cmap = mcolors.ListedColormap(['white', 'black', 'cornflowerblue', 'green', 'red'])
    ax.imshow(display, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle("Q-Learning vs Classical Algorithms", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
