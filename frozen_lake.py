# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
import random
import numpy as np
from dp import value_iteration, policy_iteration

env_name = "FrozenLake-v0"
#env = gym.make(env_name)
#print("Observation space", env.observation_space)
#print("Action space", env.action_space)

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size", self.action_size)
        
    def get_action(self):
            action = random.choice(range(self.action_size))
            return action

#agent = Agent(enviroment)

def play_episode(env, n_episodes, policy):
    wins = 0
    total_reward = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        terminated = False
        
        while not terminated:
            
            # select best action from policy
            action = np.argmax(policy[state])
            
            # perform action to observe how enviroment acted in response
            next_state, reward, terminated, info = env.step(action)
            
            env.render()
            
            # summarize total reward
            total_reward += reward
            
            #update current state
            state = next_state
            
            if terminated and reward == 1.0:
                wins += 1
    average_reward = total_reward / n_episodes
    return wins, total_reward, average_reward

# Function to find best policy
solvers = [("Value iteration", policy_iteration)]

for iteration_name, iteration_func in solvers:
    
    #load frozen lake enviroment
    enviroment = gym.make(env_name)
    
    #search for optimal poicy using iteration
    policy, V = iteration_func(enviroment)
    
    # apply poicy to enviroment
    wins, total_reward, average_reward = play_episode(enviroment, 1, policy)



