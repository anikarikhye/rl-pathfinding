import time
import heapq
import pickle
import numpy as np
from maze import Maze
from maze_env import MazeEnv
from stable_baselines3 import PPO
from bfs import bfs
from dijkstra import dijkstra
from astar import astar
 
 

def count_nodes_bfs(maze):
    queue = [maze.start]
    visited = set()
    visited.add(maze.start)
    parent = {maze.start: None}
    nodes_explored = 0
 
    while queue:
        node = queue.pop(0)
        nodes_explored += 1
        if node == maze.goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path)), nodes_explored
        for neighbor in maze._neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)
    return None, nodes_explored
 
 
def count_nodes_dijkstra(maze):
    heap = []
    heapq.heappush(heap, (0, maze.start))
    visited = set()
    parent = {maze.start: None}
    cost = {maze.start: 0}
    nodes_explored = 0
 
    while heap:
        current_cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1
        if node == maze.goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path)), nodes_explored
        for neighbor in maze._neighbors(node):
            new_cost = current_cost + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(heap, (new_cost, neighbor))
    return None, nodes_explored
 
 
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
 
def count_nodes_astar(maze):
    heap = []
    heapq.heappush(heap, (0, maze.start))
    visited = set()
    parent = {maze.start: None}
    g_cost = {maze.start: 0}
    nodes_explored = 0
 
    while heap:
        f_cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1
        if node == maze.goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path)), nodes_explored
        for neighbor in maze._neighbors(node):
            new_g = g_cost[node] + 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                f = new_g + heuristic(neighbor, maze.goal)
                parent[neighbor] = node
                heapq.heappush(heap, (f, neighbor))
    return None, nodes_explored
 
 

def evaluate_rl(maze, model_path="models/ppo_maze"):
    """
    Run the trained PPO agent on the given maze.
    Returns (path, steps, reached_goal, time_ms).
    nodes_explored is reported as steps taken (RL has no traditional node count).
    """
    env = MazeEnv(rows=maze.rows, cols=maze.cols, wall_prob=0.3, fixed_maze=True)
    env.maze = maze
    env.agent_pos = list(maze.start)
    env.steps = 0
 
    model = PPO.load(model_path)
 
   
    obs = env._get_obs()
 
    path = [tuple(env.agent_pos)]
    steps = 0
    reached_goal = False
 
    start_time = time.time()
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        path.append(tuple(env.agent_pos))
        steps += 1
        if terminated:
            reached_goal = True
            break
        if truncated:
            break
    elapsed = (time.time() - start_time) * 1000
 
    return path, steps, reached_goal, elapsed
 
 

 
if __name__ == "__main__":
    
    try:
        with open("models/train_maze.pkl", "rb") as f:
            maze = pickle.load(f)
        print(f"Loaded training maze  Start: {maze.start}  Goal: {maze.goal}")
    except FileNotFoundError:
        print("No saved training maze found — generating a fresh one.")
        maze = Maze(rows=15, cols=15, wall_prob=0.3)
 
    results = []
 
   
    for name, fn in [
        ("BFS",      count_nodes_bfs),
        ("Dijkstra", count_nodes_dijkstra),
        ("A*",       count_nodes_astar),
    ]:
        start_time = time.time()
        path, nodes = fn(maze)
        elapsed = (time.time() - start_time) * 1000
        results.append({
            "name":           name,
            "path_length":    len(path) if path else 0,
            "nodes_explored": nodes,
            "reached":        path is not None,
            "time_ms":        round(elapsed, 3),
        })
 

    rl_path, rl_steps, rl_reached, rl_time = evaluate_rl(maze)
    results.append({
        "name":           "RL (PPO)",
        "path_length":    len(rl_path) if rl_path else 0,
        "nodes_explored": rl_steps,   # steps ≈ nodes visited for RL
        "reached":        rl_reached,
        "time_ms":        round(rl_time, 3),
    })
 
   
    print(f"\n{'Algorithm':<12} {'Path Len':<12} {'Nodes/Steps':<15} {'Reached':<10} {'Time (ms)'}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['name']:<12} {r['path_length']:<12} "
            f"{r['nodes_explored']:<15} {str(r['reached']):<10} {r['time_ms']}"
        )
 
    maze.render()
