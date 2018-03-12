import tensorflow as tf

class config():
    # https://github.com/openai/maddpg
    env_name = "simple_spread"
    algo_name = "MADDPG"
    render = False # True
    render_frequency = 500 # render every this many episodes

    # output config
    output_path  = "results/" + env_name + "/" + algo_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 

        
    n_layers=2
    layer_size = 128
    discrete = False # if True, we use a single number to represent an action; else we use a vector of length action_dim
    max_ep_len = 25 # maximum episode length
    train_freq = 100 # do a training step after every train_freq samples added to replay buffer
    learning_rate = 0.001
    gamma              = .95 # the discount factor
    tau = 0.01
    replay_buffer_size = 2000 # 1000000 
    
    grad_clip = True # if true, clip the gradient using clip_val
    clip_val = .5
    exploration = True
    action_clip = True #if true, clip actions taken to be between -2 and 2 (used in maddpg.sample_n)
    action_clip_mag = 1
    output_activation = staticmethod(tf.nn.tanh)
    seed = 234
    num_episodes = 10000
    batch_size = 1024 #1024 # number of timesteps in each training period


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
