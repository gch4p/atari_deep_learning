'''
Resources used:
[1] https://medium.com/nerd-for-tech/reinforcement-learning-deep-q-learning-with-atari-games-63f5242440b1
[2] https://arxiv.org/abs/1312.5602
[3] https://github.com/farama-Foundation/gymnasium

'''

import gymnasium as gym
import ale_py
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,Dense,Flatten,Lambda
from keras import Sequential

gym.envs.registry.keys()

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5',render_mode="human")

# env.action_space.seed(42)
# observation, info = env.reset(seed=42)

env = gym.wrappers.FrameStackObservation(env, 4) #store the last 4 frames in a stack

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

MAX_FRAMES = 1000
MAX_PLAYS = 2
all_scores = []

for _ in range(MAX_PLAYS):
    env.reset()
    dead = False
    score = 0

    while not dead:
        # env.render()

        observation, reward, dead, _, _ = env.step(env.action_space.sample())
        score += reward
    
    print(score)
    all_scores.append(score)

# print(all_scores)

env.close()

