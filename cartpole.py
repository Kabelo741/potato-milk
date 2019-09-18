import gym
from DQN_Cartpole import Agent
import matplotlib.pyplot as plt
import numpy as np

#tf.reset_default_graph() ## reseting wights

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 250
    lr = 0.0005
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=[4], n_actions=2,
                  mem_size=100000, batch_size=32)
    
    filename = 'cartpole.png'
    scores=[]
    eps_history = []
    losses = []
    
    score = 0
    
    for i in range(n_games):
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode', i, 'score', score, 'average score %.3f' % avg_score,
                  'epsilon %.3f' %agent.epsilon)
        else:
            print('episode', i, 'score', score)
            
        observation = env.reset()
        score = 0
            
        while not done:
#            if i == 110 or i >= n_games-10:
#                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, observation_, action, reward, int(done))
            agent.learn()
            observation = observation_
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
    x = [i + 1 for i in range(n_games)]
    plt.subplot(2, 1, 1)
    plt.plot(x, scores)
    plt.title("Score")
    plt.subplot(2, 1, 2)
    plt.plot(x, eps_history)
    plt.title("Epsilon")
#    plt.subplot(3, 1, 3)
#    plt.plot(losses)
#    plt.title("Squared mean difference")
#    plt.savefig(filename)
    plt.show()
    
    

    
