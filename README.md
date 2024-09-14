# Taxi-Simulation with Reinforcement Learning (Q-Learning)

Welcome to the documentation for the Taxi Simulation built using Q-Learning, a type of reinforcement learning algorithm. This project simulates a taxi environment where the goal is to pick up and drop off passengers while learning the most efficient route.

<img width="110" alt="Screenshot 2024-09-14 at 10 47 02" src="https://github.com/user-attachments/assets/b9dcca30-f33e-4ed1-bdc0-1cbf3ad987a3">

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup Instructions](#setup-instructions)
5. [Code Walkthrough](#code-walkthrough)
   - [Importing Libraries](#importing-libraries)
   - [Setting Up the Camera](#setting-up-the-camera)
   - [Q-Learning Algorithm](#q-learning-algorithm)
   - [Training the Taxi](#training-the-taxi)
   - [Evaluating the Model](#evaluating-the-model)
   - [Visualizing the Journey](#visualizing-the--journey)
7. [Future Improvements](#future-improvements)
7. [Conclusion](#conclusion)

## Introduction
This project is a taxi simulation where a taxi is trained using Q-Learning, a type of reinforcement learning. The goal of the simulation is to teach the taxi to navigate a grid, pick up passengers, and drop them off at designated locations efficiently.

## Features
- **Q-learning-based Taxi Simulation:** Trains a taxi to navigate a grid using reinforcement learning.
- **Performance Tracking:** The taxi is evaluated based on rewards (successful pickups and drop-offs) and penalties (incorrect actions).
- **Visualized Output:** See the taxi move around the grid as it learns through multiple episodes.

## Technologies Used
- **Python** 
- **OpenAI Gym:** Provides the Taxi-v3 environment.
- **NumPy:** Used for storing and updating Q-values.
- **IPython:** For rendering the taxi’s movement in the terminal.

## Setup Instructions

### Clone the Repository:
```bash
git clone https://github.com/neoalexfusion/taxi-simulation.git
cd taxi-simulation
```
## Install Dependencies
```bash
pip install -r requirements.txt
```
## Run the Game
```bash
python app.py
```
## Code Walkthrough

## Importing Libraries
```bash
import gym
import numpy as np
from IPython.display import clear_output
from time import sleep
```
## Q-Learning Algorithm
```bash
q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1
```
## Training the Taxi
```bash
for episode in range(100000):
    state = env.reset()
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, done, info = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
```
## Evaluating the Model
```bash
total_penalties, total_rewards = 0, 0
for _ in range(50):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        if reward == -10:
            penalties += 1
```
## Visualizing the Taxi’s Journey
```bash
frames = []
for _ in range(50):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })

for frame in frames:
    clear_output(wait=True)
    print(frame['frame'])
    sleep(0.5)
```

## Future Improvements
- Implement more advanced reward strategies to improve learning.
- Expand the grid environment for more challenging simulations.
- Upgrade the model to use Deep Q-Learning (DQN) for better performance in larger environments.

## Conclusion
Thank you for checking out this Taxi Simulation using Q-Learning. The project demonstrates how reinforcement learning can teach a simple agent to optimize its decisions over time. Feel free to explore the code, improve the learning algorithm, or expand the environment for more complex simulations!

