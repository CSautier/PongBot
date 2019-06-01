#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:18:32 2019

@author: cstr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:58:04 2019

@author: cstr
"""

import gym
from tensorflow.keras import layers
from tensorflow.keras import Model
#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import random
from collections import deque

class DQN:
    def __init__(self, load, mode, epsilon, epsilon_min, epsilon_decay):
        self.env = gym.make('Pong-v0')
        self.memory = deque(maxlen=8000) #double-end list of fixed length to remember recent experiences
        self.learning_rate = 8e-5
        self.batch_size = 160
        self.gamma=0.95 #1-discount rate in the reward function
        self.epsilon = epsilon #initial probability for random movement if mode=0
        self.epsilon_min = epsilon_min #final probability for random movement if mode=0
        self.epsilon_decay = epsilon_decay
        self.mode=mode
        if(not load):
            self.model = self.create_model() #"fast" model, trained at each step
        else:
            self.model = load_model('pong_minimalist.h5')
            self.model.summary()
            self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        
        
    def create_model(self):
        input = layers.Input(shape=(80, 80,1))
        x = layers.Conv2D(filters=2, kernel_size=5, activation='relu', padding='same')(input)
        x= layers.MaxPooling2D(pool_size=(4,1), strides=None, padding='same', data_format=None)(x)
        x = layers.Flatten()(x)
        x= layers.Dense(2, activation="relu")(x)
        output = layers.Dense(2)(x)
        model = Model(input, output)
        model.summary()
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))     
        return model
    
    def remember(self, state, action, reward, new_state, done, obtained_reward):
        self.memory.append([state, action, reward, new_state, done, obtained_reward])
                    
            
    def replay(self): #create a batch of previous experience and train from them
        b=True
        if len(self.memory) < 20*self.batch_size: #very early step of the program, there is a risk of overfitting
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done, obtained_reward = sample
            target = self.model.predict(state.reshape((1,80,80,1))) #predicted score for each action for our network, obtained with the "stable" model for stability issues
            if done:
                print(target[0][0]- reward) #this is a very good indicator of whether the algorithm is converging or not. for pong both the value in target should be almost equal to the reward
                #target[0][action] = reward #this is what we would expect in a dqn, but in the case of pong, it can be replaced by the 2 next lines
                target[0][0] = reward
                target[0][1] = reward
            else:
                Q_future = max(self.model.predict(new_state.reshape((1,80,80,1)))[0])* self.gamma  #the estimated max score we can get from the next state, obtained with the "stable" model for stability issues
                if(Q_future<obtained_reward):
                    target[0][action] = obtained_reward
                    b=False
                else:
                    target[0][action] = reward + Q_future #the estimated score of our action is what we gained + what we can get next with a discount
            self.model.fit(state.reshape((1,80,80,1)), target, epochs=1, verbose=0)
        if b:
            print("model probably converged")
        
        
    def act(self, state, env):#chose the action to take
        predict=self.model.predict(state)[0]
        if self.mode==2: #mode where we always play the estimated best action
            return np.argmax(predict)
        elif self.mode==0: #mode where we sometime play randomly
            self.epsilon *= self.epsilon_decay #we decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon) #unless it is already at its min
            if np.random.random() < self.epsilon:
                return np.random.randint(0,2) #we play randomly one of the possible actions
            else:
                return np.argmax(predict) #we play the estimated best action
        elif self.mode==1: #mode where the exploration is given by a softmax function
            predict=np.exp(predict)
            predict/=sum(predict)
            aleatar=0.
            alea = np.random.random()
            for i in range(len(predict)):
                aleatar+=predict[i]
                if(alea<=aleatar):
                    return(i)
        print("error")
        return(0)
    
    def save_model(self, fn):
        print("saving, don't exit the program")
        self.model.save(fn)
        
def main(load=False, steps = 5000, mode=0, epsilon = 1., epsilon_min = 0.05, epsilon_decay = 0.999996, render=True): #the function to start the program. load = whether or not to load a previous network, mode 0 : classic exploration, 1 : softmax exploration, 2: argmax, render : show the game or not (can be slower)
    dqn_agent = DQN(load, mode, epsilon, epsilon_min, epsilon_decay)
    step=0
    while step<steps:
        done= False
        score=0
        obs=dqn_agent.env.reset()
        state=((obs[34:194,:,1]-72)*-1./164)[::2,::2] #we create an array containing the last two image of the game, cropped, rescaled and downsized
        step+=1
        while not done:
            reward=0
            tempmem=[]
            while reward==0: #for pong, everytime reward!=0 can be seen as the end of a cycle
                if render:
                    dqn_agent.env.render()
                action = dqn_agent.act(state.reshape(1,80,80,1), dqn_agent.env)
                obs, reward, done, info = dqn_agent.env.step(action+2) #see openai gym for information
                state2=((obs[34:194,:,1]-72)*-1./164)[::2,::2] #we create an array containing the last two image of the game, cropped, rescaled and downsized
                tempmem.append([state, action, state2])
                state=state2
            score+=reward
            for state, action, state2 in tempmem[::-1]:
                dqn_agent.remember(state, action,reward*(abs(reward)==1), state2, abs(reward)==1, reward) #done = True fix the reward to be applied instantly
                reward*=dqn_agent.gamma
            dqn_agent.replay() #train the model
            del tempmem
        dqn_agent.save_model("pong_minimalist.h5")
        print("epsilon :", dqn_agent.epsilon, "score :", score, "model saved")
        score=0
    del dqn_agent.memory
    dqn_agent.env.close()