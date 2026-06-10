import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Maze:
    def __init__(self, rows=15, cols=15, wall_prob=0.3):
        self.rows = rows
        self.cols = cols
        self.wall_prob = wall_prob
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.grid = None
        self.generate()

    def generate(self):
        while True:
            grid = np.random.choice(
                [0, 1],
                size=(self.rows, self.cols),
                p=[1 - self.wall_prob, self.wall_prob]
            )
            grid[self.start] = 0
            grid[self.goal] = 0

            if self._is_reachable(grid):
                self.grid = grid
                break

    def _is_reachable(self, grid):
        visited = set()
        queue = [self.start]
        while queue:
            node = queue.pop(0)
            if node == self.goal:
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._neighbors(node, grid):
                queue.append(neighbor)
        return False

    def _neighbors(self, node, grid=None):
        if grid is None:
            grid = self.grid
        r, c = node
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        result = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if grid[nr][nc] == 0:
                    result.append((nr, nc))
        return result

    def render(self, path=None):
        display = self.grid.copy().astype(float)
        if path:
            for cell in path:
                display[cell] = 2
        display[self.start] = 3
        display[self.goal] = 4

        cmap = mcolors.ListedColormap(['white', 'black', 'cornflowerblue', 'green', 'red'])
        plt.figure(figsize=(8, 8))
        plt.imshow(display, cmap=cmap, vmin=0, vmax=4)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    maze = Maze(rows=15, cols=15, wall_prob=0.3)
    maze.render()
    print("Maze generated successfully!")
    print(f"Start: {maze.start}, Goal: {maze.goal}")