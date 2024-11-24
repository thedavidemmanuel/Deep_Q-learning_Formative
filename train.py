import os
import sys
import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import ale_py

def create_env(render_mode=None):
    """Create and wrap the Breakout environment with enhanced preprocessing"""
    # Create the Breakout environment using the ALE (Arcade Learning Environment)
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    # Use Monitor to record statistics like episode rewards and lengths
    env = Monitor(env)
    return env

class ProgressBarCallback(CheckpointCallback):
    """
    Custom callback for displaying an enhanced progress bar during training.
    Includes training time, ETA, and episode rewards.
    """
    def __init__(self, check_freq, save_path, name_prefix="rl_model", total_timesteps=0):
        super().__init__(check_freq, save_path, name_prefix)
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.start_time = time.time()
        self.last_episode_reward = 0
        self.highest_reward = float('-inf')

    def _get_elapsed_time(self):
        """Format elapsed time in HH:MM:SS format"""
        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _get_eta(self):
        """Calculate and format ETA in HH:MM:SS format"""
        if self.current_step == 0:
            return "--:--:--"
        elapsed = time.time() - self.start_time
        steps_left = self.total_timesteps - self.current_step
        eta = int((elapsed / self.current_step) * steps_left)
        hours = eta // 3600
        minutes = (eta % 3600) // 60
        seconds = eta % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _on_step(self) -> bool:
        self.current_step += 1

        # Get current episode reward if available
        if hasattr(self.training_env, 'envs'):
            current_reward = self.training_env.envs[0].get_episode_rewards()
            if len(current_reward) > 0:
                self.last_episode_reward = current_reward[-1]
                # Update highest reward only if it's greater than the previous value
                self.highest_reward = max(self.highest_reward, self.last_episode_reward)

        # Calculate progress metrics
        progress = int((self.current_step / self.total_timesteps) * 50)
        percentage = (self.current_step / self.total_timesteps) * 100

        # Create the progress bar with multiple lines of information
        sys.stdout.write('\033[2K\033[1G')  # Clear current line

        # First line: Progress bar and basic stats
        bar = f"[{'=' * (progress-1)}{'>' if progress > 0 else ''}{' ' * (50-progress)}]"
        sys.stdout.write(f"Training Progress: {bar} {percentage:>6.1f}%\n")

        # Second line: Steps and timing information
        sys.stdout.write(f"\033[2K\033[1GSteps: {self.current_step:>7}/{self.total_timesteps} | ")
        sys.stdout.write(f"Time: {self._get_elapsed_time()} | ETA: {self._get_eta()}\n")

        # Third line: Reward information
        sys.stdout.write(f"\033[2K\033[1GLast Episode Reward: {self.last_episode_reward:>6.1f} | ")
        sys.stdout.write(f"Highest Reward: {self.highest_reward:>6.1f}")

        # Move cursor up to prepare for next update
        sys.stdout.write('\033[3A')
        sys.stdout.flush()

        return super()._on_step()

    def on_training_end(self):
        """Clean up the display when training ends"""
        # Move cursor down after the progress bar and add a newline
        sys.stdout.write('\033[3B\n')
        sys.stdout.flush()

def main():
    # Set total timesteps for training
    total_timesteps = 50000

    # Create directories for saving models, logs, and evaluations
    os.makedirs("models", exist_ok=True)  # Directory for saving trained models
    os.makedirs("logs", exist_ok=True)    # Directory for saving TensorBoard logs
    os.makedirs("eval_logs", exist_ok=True)  # Directory for saving evaluation logs

    # Create vectorized environment for training
    env = DummyVecEnv([lambda: create_env()])  # DummyVecEnv allows for easier handling of multiple environments
    eval_env = DummyVecEnv([lambda: create_env()])  # Separate environment for evaluation

    # Initialize DQN with optimized hyperparameters
    model = DQN(
        "CnnPolicy",  # Use a convolutional neural network (CNN) policy, suitable for image input
        env,
        learning_rate=1e-4,  # Learning rate for the optimizer
        buffer_size=10000,    # Replay buffer size
        learning_starts=1000,  # Number of steps to collect before starting training
        batch_size=32,  # Number of experiences sampled from the replay buffer for each training step
        gamma=0.99,  # Discount factor for future rewards
        exploration_fraction=0.2,  # Fraction of training during which exploration decreases
        exploration_initial_eps=1.0,  # Initial value of epsilon for epsilon-greedy exploration
        exploration_final_eps=0.05,  # Final value of epsilon for epsilon-greedy exploration
        train_freq=4,  # Frequency of training updates (in environment steps)
        gradient_steps=1,  # Number of gradient steps after each training step
        target_update_interval=1000,  # Frequency of updating the target network
        verbose=0,  # Set to 0 to avoid conflicting with progress bar
        tensorboard_log="logs/",  # Directory for TensorBoard logs
        device="auto"  # Automatically use GPU if available, otherwise use CPU
    )

    # Create a progress bar callback
    progress_callback = ProgressBarCallback(
        check_freq=1000,  # More frequent checkpoints for shorter training
        save_path="models/",
        name_prefix="dqn_breakout",
        total_timesteps=total_timesteps
    )

    # Create a callback to stop training once a certain reward threshold is reached
    reward_threshold = StopTrainingOnRewardThreshold(
        reward_threshold=200,  # Stop training once the agent reaches an average reward of 200
        verbose=1  # Print a message when the reward threshold is reached
    )

    # Create an evaluation callback to evaluate the model periodically during training
    eval_callback = EvalCallback(
        eval_env,  # Evaluation environment
        best_model_save_path="models/",  # Save the best model in the models directory
        log_path="eval_logs/",  # Directory to save evaluation logs
        eval_freq=5000,  # Evaluate the model every 5,000 steps
        deterministic=True,  # Use deterministic actions during evaluation
        render=False,  # Do not render the evaluation environment
        n_eval_episodes=5,  # Number of episodes to run for each evaluation
        callback_after_eval=reward_threshold  # Stop training if reward threshold is reached
    )

    try:
        print("\nStarting training for 50,000 timesteps...")
        start_time = time.time()

        # Train the agent with progress tracking
        model.learn(
            total_timesteps=total_timesteps,  # Total number of steps to train the model
            callback=[progress_callback, eval_callback],  # Use the callbacks defined above
            tb_log_name="DQN_breakout"  # Name for the TensorBoard log
        )

        print(f"\n\nTraining completed in {time.time() - start_time:.2f} seconds!")
        print(f"Best model saved as 'policy.h5'")

    except KeyboardInterrupt:
        # Handle early termination by the user
        print("\n\nTraining interrupted by user. Saving current model...")
        model.save("models/policy.h5")
        print("Model saved as 'policy.h5'")
    except Exception as e:
        # Print the error message if an exception occurs during training
        print(f"\n\nAn error occurred during training: {e}")
    finally:
        # Ensure that the environments are properly closed
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()

