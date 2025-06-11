'''
Resources used:
[1] https://medium.com/nerd-for-tech/reinforcement-learning-deep-q-learning-with-atari-games-63f5242440b1
[2] https://arxiv.org/abs/1312.5602
[3] https://github.com/farama-Foundation/gymnasium

'''

import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D,Dense,Flatten,Lambda
from keras import Sequential
from collections import deque
import random

gym.envs.registry.keys()

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5',render_mode="human")
 
# env.action_space.seed(42)
# observation, info = env.reset(seed=42)

env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
# env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStackObservation(env, 4) #store the last 4 frames in a stack
print(env.observation_space.shape)

######### setting up the neural net

model = keras.Sequential(
    [
        #necessary for CPU implementation, from [1]
        Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1]), output_shape=(84, 84, 4), input_shape=(4, 84, 84)),

        #quotes from DeepMind paper [2]
        #"The first hidden layer convolves 32 filters of 8×8 with stride 4 with the input image and applies a rectifier nonlinearity." 
        Conv2D(32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(4, 84, 84)),
        #""
        Conv2D(64, kernel_size=(4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(env.action_space.n, activation="linear"),
    ]
)

rms = tf.keras.optimizers.RMSprop()
model.compile(loss="mse", optimizer=rms)

######### 


# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

#     if terminated or truncated:
#         observation, info = env.reset()

def choose_action(epsilon,cur_state):
    min_epsilon = .2
    
    if len(memory) < INIT_MEM:
        #prefill with random events
        action = env.action_space.sample() 
        return action,epsilon
    
    if random.random() < epsilon:
        #explore!
        action = env.action_space.sample()
        epsilon = max(min_epsilon,epsilon*.99)
    else:
        action = np.argmax(model.predict(np.expand_dims(cur_state, axis=0)))
        
    return action,epsilon

MAX_FRAMES = 1000
MAX_PLAYS = 2
all_scores = []

MAX_MEM = int(MAX_FRAMES / 10)
INIT_MEM = int(MAX_FRAMES / 100)
memory = deque(maxlen=MAX_MEM)

epsilon = .99

for i in range(MAX_PLAYS):
    prev_state = env.reset()
    cur_state = prev_state[0]
    dead = False
    score = 0

    while not dead:
        # env.render()
        
        action,epsilon = choose_action(epsilon,cur_state)
        prev_state = cur_state
        cur_state, reward, dead, _, _ = env.step(action)
        
        score += reward
        memory.append([np.expand_dims(prev_state[0], axis=0),np.expand_dims(cur_state[0], axis=0),reward,dead,action])
    
    # print(cur_state)
    print(score)
    all_scores.append(score)

# print(all_scores)

env.close()

