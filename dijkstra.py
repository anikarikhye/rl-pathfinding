import heapq
from maze import Maze

def dijkstra(maze):
    heap = []
    heapq.heappush(heap, (0, maze.start))
    visited = set()
    parent = {maze.start: None}
    cost = {maze.start: 0}

    while heap:
        current_cost, node = heapq.heappop(heap)

        if node == maze.goal:
            return reconstruct_path(parent, maze.goal)

        if node in visited:
            continue
        visited.add(node)

        for neighbor in maze._neighbors(node):
            new_cost = current_cost + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(heap, (new_cost, neighbor))

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
    path = dijkstra(maze)

    if path:
        print(f"Path found! Length: {len(path)} steps")
        maze.render(path=path)
    else:
        print("No path found")