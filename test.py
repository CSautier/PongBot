#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:46:05 2019

@author: cstr
"""
import matplotlib.pyplot as plt
import gym

env = gym.make('Pong-v0')
env.reset()
for _ in range(1000):
    env.render()
    observation =env.step(3) # take a random action
plt.imshow(observation[0][:,:,0])
env.close()