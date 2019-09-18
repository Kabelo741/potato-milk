import tensorflow as tf
import numpy as np
import os

class DeepQNet(object):
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims=256, fc2_dims=256):
        self.name = name
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
#        self.saver = tf.train.Saver()
        
    def build_network(self):

        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims],
                                        name='inputs')
            self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                           name='Q_value')
            
            ## BUILDING THE NN
            
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, activation=tf.nn.relu)
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions)
            
            self.loss = tf.reduce_mean(tf.square(self.Q_values - self.q_target)) #will calculate q target in learning function
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            
class Agent(object):
    def __init__(self, lr, gamma, epsilon, mem_size, batch_size, n_actions, input_dims,
                 epsilon_dec = 0.998, epsilon_min = 0.01):
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.q_eval = DeepQNet(lr, n_actions,'q_eval', input_dims)
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(mem_size)
        self.terminal_memory = np.zeros(mem_size, dtype=np.int8)
        self.mem_cntr = 0
        
    def store_transition(self, state, state_, action, reward, terminal):
        action = int(action)
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - terminal
        
    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                          feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min
        
        return action
    
    def learn(self):
        if self.mem_cntr > self.batch_size:
            if self.mem_cntr < self.mem_size:   
                max_mem = self.mem_cntr  
            else:
                max_mem = self.mem_size
            
            batch = np.random.choice(max_mem, self.batch_size)
            
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            
            q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                           feed_dict={self.q_eval.input: state_batch})
            q_next = self.q_eval.sess.run(self.q_eval.Q_values,
                                          feed_dict={self.q_eval.input: new_state_batch})
            
            q_target = q_eval.copy()
            
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            
            q_target[batch_index, action_indices] = reward_batch + \
            self.gamma*np.max(q_next, axis=1)*terminal_batch
            
            _ = self.q_eval.sess.run(self.q_eval.train_op, feed_dict={self.q_eval.input: state_batch,
                                                                      self.q_eval.q_target: q_target})
            
        self.mem_cntr += 1

    
            
                
            
            

