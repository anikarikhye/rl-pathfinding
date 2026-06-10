import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from maze import Maze

class QLearningAgent:
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.maze = maze
        self.rows = maze.rows
        self.cols = maze.cols
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state = (row, col), actions = 0,1,2,3
        self.q_table = np.zeros((maze.rows, maze.cols, 4))
        self.moves = [(-1,0),(1,0),(0,-1),(0,1)]  # up down left right

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state[0], state[1]])

    def step(self, state, action):
        dr, dc = self.moves[action]
        nr, nc = state[0] + dr, state[1] + dc

        # check valid move
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze.grid[nr][nc] == 0:
            next_state = (nr, nc)
        else:
            next_state = state  # stay in place if wall

        # reward
        if next_state == self.maze.goal:
            reward = 100.0
        elif next_state == state:
            reward = -1.0  # hit a wall
        else:
            reward = -0.1  # step penalty

        return next_state, reward

    def train(self, episodes=5000):
        rewards_per_episode = []
        steps_per_episode = []
        successes = 0

        for ep in range(episodes):
            state = self.maze.start
            total_reward = 0
            steps = 0
            max_steps = self.rows * self.cols * 4

            while steps < max_steps:
                action = self.get_action(state)
                next_state, reward = self.step(state, action)

                # Q-Learning update
                best_next = np.max(self.q_table[next_state[0], next_state[1]])
                self.q_table[state[0], state[1], action] += self.alpha * (
                    reward + self.gamma * best_next - self.q_table[state[0], state[1], action]
                )

                state = next_state
                total_reward += reward
                steps += 1

                if state == self.maze.goal:
                    successes += 1
                    break

            # decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_per_episode.append(total_reward)
            steps_per_episode.append(steps)

            if (ep + 1) % 500 == 0:
                recent_success = sum(1 for s in steps_per_episode[-500:] if s < max_steps) / 5
                print(f"Episode {ep+1}: avg_reward={np.mean(rewards_per_episode[-500:]):.1f}, success_rate={recent_success:.1f}%, epsilon={self.epsilon:.3f}")

        return rewards_per_episode, steps_per_episode

    def get_path(self):
        state = self.maze.start
        path = [state]
        visited = set()
        max_steps = self.rows * self.cols * 2

        for _ in range(max_steps):
            if state in visited:
                return None  # loop detected
            visited.add(state)
            action = np.argmax(self.q_table[state[0], state[1]])
            dr, dc = self.moves[action]
            nr, nc = state[0] + dr, state[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze.grid[nr][nc] == 0:
                state = (nr, nc)
            path.append(state)
            if state == self.maze.goal:
                return path

        return None

if __name__ == "__main__":
    # create and save maze
    maze = Maze(rows=10, cols=10, wall_prob=0.2)
    with open("models/train_maze.pkl", "wb") as f:
        pickle.dump(maze, f)

    print(f"Maze created. Start: {maze.start}, Goal: {maze.goal}")

    agent = QLearningAgent(maze)
    print("Training Q-Learning agent...")
    rewards, steps = agent.train(episodes=5000)

    # save agent
    with open("models/q_agent.pkl", "wb") as f:
        pickle.dump(agent, f)

    # get learned path
    path = agent.get_path()
    if path:
        print(f"\nAgent reached goal! Path length: {len(path)} steps")
        maze.render(path=path)
    else:
        print("\nAgent failed to find path — try more episodes")
