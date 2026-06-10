from maze import Maze
from collections import deque

def bfs(maze):
    queue = deque()
    queue.append(maze.start)
    visited = set()
    visited.add(maze.start)
    parent = {maze.start: None}

    while queue:
        node = queue.popleft()

        if node == maze.goal:
            return reconstruct_path(parent, maze.goal)

        for neighbor in maze._neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    return None  

def reconstruct_path(parent, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path

if __name__ == "__main__":
    maze = Maze(rows=15, cols=15, wall_prob=0.3)
    path = bfs(maze)

    if path:
        print(f"Path found! Length: {len(path)} steps")
        maze.render(path=path)
    else:
        print("No path found")
