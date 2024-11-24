import os
import time
import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import numpy as np

def create_env(render_mode=None):
    """Create and wrap the Breakout environment with enhanced preprocessing"""
    # Create the Breakout environment using the ALE (Arcade Learning Environment)
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    return env

def main():
    # Load the trained model
    model_path = "models/policy.h5"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        return

    print("Loading the trained model...")
    model = DQN.load(model_path)
    print("Model loaded successfully!")

    # Create the environment for playing
    env = create_env(render_mode="human")

    # Run several episodes and visualize the agent's performance
    n_episodes = 5
    max_steps_per_episode = 1000  # Set a limit for max steps per episode to avoid endless episodes
    for episode in range(1, n_episodes + 1):
        print(f"\nStarting episode {episode}...")
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Occasionally take a random action to test exploration
            if np.random.rand() < 0.1:
                action = env.action_space.sample()
                print(f"[Random Action] Step: {steps}, Action: {action}")
            else:
                action, _states = model.predict(obs, deterministic=True)  # Greedy action selection
                print(f"[Model Action] Step: {steps}, Action: {action}")

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Render every few steps to avoid overwhelming the system
            if steps % 10 == 0:
                env.render()

            # Debugging information to understand agent's behavior
            print(f"Step: {steps}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

            # Handle both done and truncated states
            if done or truncated:
                break

        print(f"Episode {episode} completed with reward: {episode_reward}")

    env.close()
    print("\nAll episodes completed!")

if __name__ == "__main__":
    main()
