# RL vs Classical Pathfinding

Comparing reinforcement learning (Q-Learning) against classical pathfinding algorithms (BFS, Dijkstra, A*) on procedurally generated mazes.

## What it does
- Generates random mazes with guaranteed valid paths
- Implements BFS, Dijkstra, and A* from scratch
- Trains a Q-Learning agent to navigate the same mazes
- Benchmarks all algorithms on path quality and inference speed

## Key Result
All algorithms found the optimal path (19 steps). Q-Learning matched BFS and A* on path quality and was faster at inference — but required 5000 training episodes upfront.

| Algorithm | Path Length | Reached Goal | Time (ms) |
|-----------|-------------|--------------|-----------|
| Q-Learning | 19 | True | 0.058 |
| BFS | 19 | True | 0.122 |
| A* | 19 | True | 0.147 |

## Tech Stack
Python, NumPy, Matplotlib, Gymnasium, Stable-Baselines3

## Run it
```bash
pip install numpy matplotlib gymnasium stable-baselines3
python3 qlearning.py        # train Q-Learning agent
python3 final_comparison.py # run benchmark
```
