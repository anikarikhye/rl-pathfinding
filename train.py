import os
import pickle
from maze_env import MazeEnv
from sb3_contrib import RecurrentPPO  # Upgraded import

# 1. Create ONE single environment and save its maze
env = MazeEnv(rows=15, cols=15, wall_prob=0.3, fixed_maze=True)

# 2. Save the maze immediately for evaluation later
os.makedirs("models", exist_ok=True)
with open("models/train_maze.pkl", "wb") as f:
    pickle.dump(env.maze, f)

print(f"Training maze saved. Goal: {env.maze.goal}")

# 3. Initialize the Recurrent PPO Agent with LSTM layers
model = RecurrentPPO(
    "MlpLstmPolicy",  # Gives the model short-term memory capability
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    policy_kwargs=dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])  # Balanced network for LSTM tracking
    )
)

# 4. Start the learning process
print("Starting training on single fixed maze with RecurrentPPO...")
print("-" * 50)
model.learn(total_timesteps=300_000)

# 5. Save the trained weights
model.save("models/recurrent_ppo_maze")
print("\nTraining complete! Recurrent Model saved.")