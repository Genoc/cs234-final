import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import time
import inspect
import pickle
from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.collisions import count_agent_collisions
from utils.distance_from_landmarks import get_distance_from_landmarks
from config_v2 import config

from actor_critic_v2 import Agent
from actor_critic_v2 import Critic
from memory import Memory
from make_env import make_env
import random

from utils.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# run for some timesteps to fill up the buffer initially
def prep_buffer():
  num_steps = config.batch_size+1
  obs_n = env.reset()
  for t in range(num_steps):
    # figure out the next action to take
    act_n = []
    for i in range(env.n):
      action = agents[i].get_sampled_action(obs_n[i])
      if config.exploration == 1:
        action = action + agents[i].noise()
      if config.action_clip:
        action = np.clip(action, -2, 2)
      act_n.append(action)
    
    # take the action
    next_obs_n, rew_n, done_n, _ = env.step(act_n)
      
    # add timestep to the replay buffer
    memory.remember(obs_n, act_n, rew_n, next_obs_n, done_n[0])
    # idx = replay_buffer.store_frame(obs_n)
    # replay_buffer.store_effect(idx, act_n, rew_n, done_n)
    obs_n = next_obs_n  
      
    # prep for next episode if we are starting a new one
    if any(done_n) or t % config.max_ep_len == 0:
      obs_n = env.reset()
         
def train():
  num_episodes = config.num_episodes
  batch_size = config.batch_size
  losses = []
  rewards = []
  collisions = []
  num_collisions = 0
  train_steps = 0
    
  for episode in range(num_episodes):
    obs_n = env.reset()
    current_ep_length = 0
    episode_loss = 0
    episode_reward = 0
    num_collisions = 0
    
    while True:
      train_steps += 1
      current_ep_length += 1
      
      if config.render and episode % config.render_frequency == 0:
        time.sleep(0.1)
        env.render()
        
      # figure out the next action to take
      act_n = []
      for i in range(env.n):
        action = agents[i].get_sampled_action(obs_n[i])
        if config.exploration:
          action = action + agents[i].noise()
        if config.action_clip:
          action = np.clip(action, -2, 2)
        act_n.append(action)
      
      # take the action and store in the replay buffer
      next_obs_n, rew_n, done_n, _ = env.step(act_n)
      memory.remember(obs_n, act_n, rew_n, next_obs_n, done_n[0])
      
      # update the metric variables for the taken action  
      episode_reward += np.sum(rew_n) # sum the rewards of all agents
      num_collisions += count_agent_collisions(env)
      
      # Every 100 timesteps, sample from the replaybuffer and update every agent
      if train_steps % config.train_freq == 0:
        size = memory.pointer
        batch = random.sample(range(size), batch_size)
        s, a, r, sn, _ = memory.sample(batch, env.n)
                
        critic_losses = []
        for i in range(env.n):
          critic_losses.append(critics[i].update(s, a, r, sn))
          agents[i].update(agents, s)
      
        # store the critic loss
        losses.append(np.mean(critic_losses))
              
      # if episode is over, do cleanup for next episode start
      if any(done_n) or current_ep_length > config.max_ep_len:
        rewards.append(episode_reward)
        collisions.append(num_collisions)

        if episode % 25 == 0 and train_steps > config.train_freq:
          logger.info("Episode " + str(episode) + " loss: " + str(np.mean(critic_losses)))
          logger.info("Episode " + str(episode) + " reward: " + str(episode_reward))
        break
      # otherwise, update current state and continue
      obs_n = next_obs_n
    
  
  # save stats to dictionary and return
  results = {}
  results['losses'] = losses
  results['rewards'] = rewards
  results['collisions'] = collisions  
  
  return results
  
if __name__ == '__main__':
  env = make_env(config.env_name)
  
  env.seed(config.seed)
  random.seed(config.seed)
  np.random.seed(config.seed)
  tf.set_random_seed(config.seed)
  
  sess = tf.Session()
  
  # directory for training outputs
  if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

  logger = get_logger(config.log_path)

  action_dim = env.action_space[0].n
  observation_dim = env.observation_space[0].shape[0]
  
  agents = []
  eval_actions = []
  target_actions = []
  state_ph = []
  state_next_ph = []
  critics = []
  # replay_buffer = ReplayBuffer(config.replay_buffer_size, observation_dim, action_dim, env.n)
  memory = Memory(config.replay_buffer_size)
  
  # initialize the agents
  for i in range(env.n):
    action_dim = env.action_space[i].n
    obs_dim = env.observation_space[i].shape[0]
    state = tf.placeholder(tf.float32, shape = [None, obs_dim])
    state_next = tf.placeholder(tf.float32, shape=[None, obs_dim])
    
    agents.append(Agent('a'+str(i), env, config, sess, logger, i, state, state_next))
    eval_actions.append(agents[i].mu_eval)
    target_actions.append(agents[i].mu_target)
    state_ph.append(state)
    state_next_ph.append(state_next)
  
  # initialize the critics
  
  for i in range(env.n):
    reward = tf.placeholder(tf.float32, [None])
    
    critics.append(Critic('c'+str(i), env, config, sess, logger, i, eval_actions, target_actions, state_ph, state_next_ph, reward))
    agents[i].update_mu_op(critics[i].action_grads[i])
  
  sess.run(tf.global_variables_initializer())
  
  start = time.time()
  prep_buffer()
  print('done prepping buffer')
  results = train()
  end = time.time()
  
  print("Finished {} episodes in {} seconds".format(config.num_episodes, end-start))
  save_obj(results, 'run_results')

  # results.dump(config.ouput_path + 'run-results.csv')