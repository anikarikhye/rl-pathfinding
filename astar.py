import heapq
from maze import Maze

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze):
    heap = []
    heapq.heappush(heap, (0, maze.start))
    visited = set()
    parent = {maze.start: None}
    g_cost = {maze.start: 0}

    while heap:
        f_cost, node = heapq.heappop(heap)

        if node == maze.goal:
            return reconstruct_path(parent, maze.goal)

        if node in visited:
            continue
        visited.add(node)

        for neighbor in maze._neighbors(node):
            new_g = g_cost[node] + 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                f = new_g + heuristic(neighbor, maze.goal)
                parent[neighbor] = node
                heapq.heappush(heap, (f, neighbor))

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
    path = astar(maze)

    if path:
        print(f"Path found! Length: {len(path)} steps")
        maze.render(path=path)
    else:
        print("No path found")