import numpy as np
import tensorflow as tf
from config_v2 import config
from utils.general import get_logger, export_plot
from utils.network import build_mlp
from utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from utils.merge_dicts import merge_dicts

# Represents one agent
# Has a policy network that take's the agent's observation of the world and outputs 
# the action to take
class Agent:
  
  def __init__(self, scope, env, config, session, logger, agent_idx, eval_states, target_states):
    self.scope = scope
    self.eval_scope = self.scope + '/eval'
    self.target_scope = self.scope + '/target'
    
    self.env = env
    self.config = config
    self.sess = session
    self.logger = logger
    self.agent_idx = agent_idx #not used currently 
    
    self.action_dim = self.env.action_space[0].n
    self.observation_dim = self.env.observation_space[0].shape[0]
    self.lr = self.config.learning_rate
    self.tau = self.config.tau
    
    self.eval_states = eval_states
    self.target_states = target_states
    
    self.noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(self.action_dim),
        sigma=0.3,
        theta=0.15,
        dt=1e-2,
        x0=None)
    
    with tf.variable_scope(self.scope):
      self.mu_eval = build_mlp(self.eval_states, self.action_dim, 
                            scope='eval', n_layers=self.config.n_layers,
                            size=self.config.layer_size,
                            output_activation=self.config.output_activation,
                            trainable=True)
      self.mu_target = build_mlp(self.target_states, self.action_dim, 
                            scope='target', n_layers=self.config.n_layers,
                            size=self.config.layer_size,
                            output_activation=self.config.output_activation,
                            trainable=False)
                                                    
      # create update op for target network
      self.eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.eval_scope)
      self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)
    
      self.update_target_mu_op = [tf.assign(t, (1 - self.tau)*t + self.tau*e)
                                for e,t in zip(self.eval_vars, self.target_vars)]
  
  # Create the op that updates the gradients for mu
  # Requires the gradients of the Q function
  def update_mu_op(self, action_gradients):
    with tf.variable_scope(self.scope):
      self.action_gradients = tf.gradients(ys=self.mu_eval,
                                      xs=self.eval_vars,
                                      grad_ys=action_gradients)
      grads_vars = zip(self.action_gradients, self.eval_vars)
      
      if self.config.grad_clip:
        variables = [v for g,v in grads_vars]
        clipped = [tf.clip_by_norm(g, self.config.clip_val) for g,v in grads_vars]
        grads_vars = zip(clipped, variables)
        
      self.optimizer = tf.train.AdamOptimizer(-self.lr)
      self.update_mu_op = self.optimizer.apply_gradients(grads_vars)
            
    
  # Update the policy network and also update the target policy network
  def update(self, agents, states):
    a = {}
    for i in range(len(states)):
      a[agents[i].eval_states] = states[i]
    
    self.sess.run(self.update_mu_op, feed_dict=a)  
      
    #self.sess.run(self.update_mu_op, feed_dict={self.obs_placeholder: obs,
    #                                      self.action_placeholder: actions,
    #                                    })
    
    self.sess.run(self.update_target_mu_op)
  
  # given a state, return an action
  def get_sampled_action(self, obs):
    return self.sess.run(self.mu_eval, feed_dict = {self.eval_states: obs[np.newaxis,:]})[0]
    

# The centralized Q network (for a particular agent)
#
# Takes as an input the current state across all agents and an action for every agent,
# and outputs the Q-value for the particular agent
class Critic:
  def __init__(self, scope, env, config, session, logger, agent_idx, eval_actions, target_actions, eval_states, target_states, rewards):
    self.scope = scope
    self.eval_scope = self.scope + '/eval'
    self.target_scope = self.scope + '/target'
       
    self.env = env
    self.config = config
    self.sess = session
    self.logger = logger
    self.agent_idx = agent_idx

    self.action_dim = self.env.action_space[0].n
    self.observation_dim = self.env.observation_space[0].shape[0]
    self.lr = self.config.learning_rate
    self.gamma = self.config.gamma
    self.tau = self.config.tau
    
    self.eval_states = eval_states
    self.eval_actions = eval_actions
    self.target_states = target_states
    self.target_actions = target_actions
    self.rewards = rewards

    with tf.variable_scope(self.scope):
      # flat_eval_act = [tf.layers.flatten(i) for i in self.eval_actions_n]
      # flat_target_act = [tf.layers.flatten(i) for i in self.target_actions_n]
      
      # flat_eval_act.append(tf.layers.flatten(self.eval_states))
      # flat_target_act.append(tf.layers.flatten(self.target_states))
      
      # self.eval_input = tf.concat(flat_eval_act, axis=1)
      # self.target_input = tf.concat(flat_target_act, axis=1)                     
      '''
      self.q_eval = build_mlp(self.eval_input, 1, 
                               scope='eval', n_layers=self.config.n_layers,
                               size=self.config.layer_size,
                               output_activation=self.config.output_activation,
                               trainable=True)
      self.q_target = build_mlp(self.target_input, 1, 
                               scope='target', n_layers=self.config.n_layers,
                               size=self.config.layer_size,
                               output_activation=self.config.output_activation,
                               trainable=False)
      '''
      self.q_eval = self.build_network(self.eval_states,
                              self.eval_actions,
                              'eval', trainable=True)
      self.q_target = self.build_network(self.target_states,
                              self.target_actions,
                              'target', trainable=False)
                                                          
                                                                                                        
      # create update op for critic
      self.eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.eval_scope)
      self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.target_scope)
      
      self.update_target_q_op = [tf.assign(t, (1 - self.tau)*t + self.tau*e)
                                   for e,t in zip(self.eval_vars, self.target_vars)]
                                   
      # create optimization op for critic
      # if done, then target Q-value is just reward (as the episode is over)
      y = self.rewards + self.gamma*self.q_target
      self.loss = tf.losses.mean_squared_error(y, self.q_eval)
      
      self.optimizer = tf.train.AdamOptimizer(self.lr)
      grad_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
      grads_vars = self.optimizer.compute_gradients(self.loss, grad_vars)
      
      if self.config.grad_clip:
        variables = [v for g,v in grads_vars]
        clipped = [tf.clip_by_norm(g, self.config.clip_val) for g,v in grads_vars]
        grads_vars = zip(clipped, variables)
      
      self.update_q_op = self.optimizer.apply_gradients(grads_vars)
      
      # save the gradients with respect to each action, as the actor needs them
      # for its gradient update
      self.action_grads = []
      for i in range(self.env.n):
        self.action_grads.append(tf.gradients(ys=self.q_eval, 
                                  xs=self.eval_actions[i])[0])
    
  # Update the critic network and return the loss, for logging purposes
  # Also update the target critic network 
  def update(self, states_n, actions_n, rewards, states_next_n):
    s = {i:d for i,d in zip(self.eval_states, states_n)}
    a = {i:d for i,d in zip(self.eval_actions, actions_n)}
    sn = {i:d for i,d in zip(self.target_states, states_next_n)}
    rewards = {self.rewards:rewards}
    
    feed = merge_dicts(s, a, sn, rewards)
    loss, _ = self.sess.run([self.loss, self.update_q_op], feed_dict=feed)
                                
    self.sess.run(self.update_target_q_op)
    return loss
    
  def build_network(self, x1, x2, scope, trainable):
    with tf.variable_scope(scope):
      W = tf.random_normal_initializer(0.0, 0.1)
      b = tf.constant_initializer(0.1)

      first = True
      for i in range(len(x1)):
        h1 = tf.layers.dense(x1[i], 50, activation=tf.nn.relu,
                        kernel_initializer=W, bias_initializer=b,
                        name='h1-' + str(i), trainable=trainable)
        h21 = tf.get_variable('h21-' + str(i), [50, 50],
                        initializer=W, trainable=trainable)
        h22 = tf.get_variable('h22-' + str(i), [self.action_dim, 50],
                        initializer=W, trainable=trainable)

        if first == True:
          h3 = tf.matmul(h1, h21) + tf.matmul(x2[i], h22)
          first = False
        else:
          h3 = h3 + tf.matmul(h1, h21) + tf.matmul(x2[i], h22)

      b2 = tf.get_variable('b2', [1, 50], initializer=b,
                      trainable=trainable)
      h3 = tf.nn.relu(h3 + b2)
      values = tf.layers.dense(h3, 1, kernel_initializer=W,
                      bias_initializer=b, name='values',
                      trainable=trainable)

    return values