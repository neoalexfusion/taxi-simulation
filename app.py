import gym
import numpy as np
import os
import time

# Setup environment variables for headless display
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Initialize the environment
env = gym.make("Taxi-v3")

# Q-table initialization
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

# Training the agent
for i in range(100000):
    state = env.reset()
    penalties, reward, done = 0, 0, False
    
    while not done:
        # Exploration-exploitation tradeoff
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Q-learning formula
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        if reward == -10:
            penalties += 1

        state = next_state

# Evaluation of the trained agent
total_penalties = 0
episodes = 50
frames = []  # For animation

for _ in range(episodes):
    state = env.reset()
    penalties, reward = 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
            "frame": env.render(mode='ansi'),
            "state": state,
            "action": action,
            "reward": reward
        })

    total_penalties += penalties

print(f"Total penalties after {episodes} episodes: {total_penalties}")

# Visualization of the taxi's journey in VS Code
def clear_terminal():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS/Linux
    else:
        _ = os.system('clear')

for frame in frames:
    clear_terminal()  # Clear terminal before printing the next frame
    print(frame["frame"])  # Display the taxi environment
    print(f"State: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")
    time.sleep(0.5)