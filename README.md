# Deep Q-Learning for Atari Breakout

## Overview

This project implements a Deep Q-Learning (DQN) algorithm to train a reinforcement learning agent for playing the classic Atari game Breakout. The project is based on Python and leverages libraries such as Stable-Baselines3, Gymnasium, Keras, and others for both training and playing the game. The goal of this project is to apply Deep Reinforcement Learning to learn an optimal policy that maximizes the agent's performance in the game. The project includes scripts for training the model, playing the game with a trained model, and evaluating the agent's performance.

## Project Structure

The project repository contains the following key files and folders:

- **train.py**: Python script for training the DQN model on the Breakout environment.
- **play.py**: Python script to visualize and evaluate the performance of the trained DQN model.
- **models/**: Directory to save trained models, specifically the DQN agent's policy model.
- **requirements.txt**: List of dependencies and Python packages required for the project.
- **.gitignore**: Git configuration file to exclude unnecessary files from version control.
- **.gitattributes**: Configuration for Git LFS (Large File Storage).
- **README.md**: Detailed documentation for the project.
- **test_imports.py**: Script to verify that all required Python modules can be imported successfully.

## Requirements

### Prerequisites

To run the project, ensure you have the following prerequisites installed:

- Python 3.8+
- pip (Python package installer)
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/thedavidemmanuel/Deep_Q-learning_Formative.git
   cd Deep_Q-learning_Formative
   ```

2. Set up a virtual environment (optional):

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Set up Git LFS to handle large files (if not done already):

   ```sh
   git lfs install
   ```

5. Make sure to track the model file using Git LFS:
   ```sh
   git lfs track "models/policy.h5"
   ```

## How to Run

### Training the DQN Model

To train the agent using Deep Q-Learning, use the `train.py` script. This will create and train the DQN model using the Atari Breakout environment:

```sh
python train.py
```

The script uses the Stable-Baselines3 library for model training. The model will be saved to the `models` directory after training is completed. Training can take several hours depending on the computational resources available.

### Playing with a Trained Model

Once the model is trained, you can evaluate it using the `play.py` script to visualize how well the agent plays Breakout:

```sh
python play.py
```

The `play.py` script loads the trained model from `models/policy.h5` and plays a specified number of episodes in the Breakout environment. The agent's performance is rendered, and the game can be observed step by step.

## Implementation Details

### Environment Setup

The environment used is `ALE/Breakout-v5` provided by the gymnasium and ale_py packages. The `create_env` function wraps the environment and provides a preprocessing setup that includes features such as frame skipping and grayscale conversion to simplify learning.

### Model Training

The DQN (Deep Q-Network) agent is implemented using Stable-Baselines3. During training, the agent learns a policy function π that maps observations to actions to maximize the cumulative reward. The neural network learns by approximating the action-value function using a replay buffer and target network to stabilize training. The training process uses an ε-greedy strategy for exploration, which gradually reduces ε from a high value to encourage early exploration and late exploitation.

### Model Playing

The trained model is evaluated by running it on multiple episodes of Breakout. Exploration is maintained during evaluation by occasionally taking a random action (ε = 0.1). The model's action selection is otherwise greedy, using its current learned policy. The script limits each episode to 1000 steps to avoid endless games.

## Key Concepts

- **Reinforcement Learning (RL)**: RL is the process by which an agent learns to make decisions by taking actions that maximize some notion of cumulative reward.
- **Deep Q-Learning**: An extension of Q-Learning that uses a deep neural network to approximate the Q-function, enabling it to handle high-dimensional input spaces like visual input from video games.
- **DQN Algorithm**: A powerful approach for learning policies directly from high-dimensional sensory inputs by approximating the optimal action-value function.

## Evaluation Metrics

- **Episode Reward**: The cumulative reward the agent receives during an episode of play.
- **Training Loss**: Used to monitor the performance of the neural network during training.
- **Average Reward Over Episodes**: The average reward across multiple episodes is used to measure the agent's consistency in performance.

## Results

After training, the agent shows improved performance over random play. The final agent is capable of playing the game competently, with episodes showing increased survival times and higher scores. A sample reward for the evaluation episode is printed at the end of each run to provide insights into the agent's effectiveness.

## Contributions

Feel free to contribute to this project by opening issues and submitting pull requests. Any improvements, bug fixes, or additional features are welcome!

## References

- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- [Deep Q-Learning with Atari Games](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
