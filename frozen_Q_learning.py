# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:09:14 2019

@author: Kabelo
"""

import gym 
import numpy as np
import random
import matplotlib.pyplot as plt

env_name = 'FrozenLake-v0'

class Agent():
    def __init__(self, env):
        self.env = env
        self.action_size = env.nA
        self.actionSpace = ['LEFT', 'DOWN', 'RIGHT', 'UP']

        
    def get_action(self):
        action = random.choice(range(self.action_size))
        return action
    
def maxAction(Q, state, action):
    values = np.array([Q[state, a] for a in range(action)] )
    action = np.argmax(values)
    return action

if __name__ == '__main__':
    env = gym.make(env_name)
    agent = Agent(env)
#    env.render()
    
    alpha = 0.1
    gamma = 1.0
    eps = 1.0
    
    Q = {}
    for state in range(env.nS):
        for action in range(env.nA):
            Q[state, action] = 0
            
    numGames = 1000
    totalRewards = np.zeros(numGames)
    epsilons = np.zeros(numGames)
    
    for i in range(numGames):
        if i % 200 == 0:
            print('starting game', i)
            
        done = False
        epRewards = 0
        state = env.reset()
#        reward = 0
        
        while not done:
            rand = np.random.random()
#                print(rand, eps)
            if rand < (1 - eps):
                action = maxAction(Q, state, env.nA)
            else:
                action = agent.get_action()
                
            state_, reward, done, prob = env.step(action)
#            reward -= 1
            
            action_ = maxAction(Q, state_, env.nA)
            Q[state, action] = Q[state, action] + alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_
            
            if i == numGames-1:
               env.render()
            
        if eps - 2/numGames > 0:
            eps -= 2/numGames
        else:
            eps = 0
                
        totalRewards[i] = reward
        epsilons[i] = eps
    plt.plot(totalRewards)#, plt.plot(epsilons)
    plt.show()
                
    
    