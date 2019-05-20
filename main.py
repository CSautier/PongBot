#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:58:04 2019

@author: cstr
"""

import gym
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import numpy as np
import random
from collections import deque

class DQN:
    def __init__(self, load, mode, epsilon, epsilon_min, epsilon_decay):
        self.env = gym.make('Pong-v0')
        self.memory = deque(maxlen=2000) #double-end list of fixed length to remember recent experiences
        self.learning_rate = 5e-4 #ideally start with 1e-3 and end with 5e-4
        self.batch_size = 16
        self.gamma=0.9 #1-discount rate in the reward function
        self.epsilon = epsilon #initial probability for random movement if mode=0
        self.epsilon_min = epsilon_min #final probability for random movement if mode=0
        self.epsilon_decay = epsilon_decay
        self.mode=mode
        if(not load):
            self.model = self.create_model() #"fast" model, trained at each step
            self.target_model = self.create_model() #"stable" model, trained once in a while
        else:
            self.model = load_model('pong.h5')
            self.target_model = load_model('pong.h5')
            self.model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.learning_rate))
        
        
    def create_model(self):
        input = layers.Input(shape=(80, 80,2))
        x = layers.Conv2D(filters=10, kernel_size=20, activation='relu', padding='valid',strides=(4,4))(input)
        x = layers.Conv2D(filters=20, kernel_size=10, activation='relu', padding='valid',strides=(2,2))(x)
        x = layers.Conv2D(filters=40, kernel_size=3, activation='relu', padding='valid')(x)
        x = layers.Flatten()(x)
        output = layers.Dense(2)(x)
        model = Model(input, output)
        model.summary()
        model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.learning_rate))     
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
                    
            
    def replay(self): #create a batch of previous experience and train from them
        targets=[]
        states=[]
        if len(self.memory) < 2*self.batch_size: #very early step of the program, there is a risk of overfitting
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state.reshape((1,)+state.shape)) #predicted score for each action for our network, obtained with the "stable" model for stability issues
            if done:
                #print(target[0], reward) #this is a very good indicator of whether the algorithm is converging or not. for pong both the value in target should be almost equal to the reward
                #target[0][action] = reward #this is what we would expect in a dqn, but in the case of pong, it can be replaced by the 2 nex lines
                target[0][0] = reward
                target[0][1] = reward
            else:
                Q_future = max(self.target_model.predict(new_state.reshape((1,)+state.shape))[0]) #the estimated max score we can get from the next state, obtained with the "stable" model for stability issues
                target[0][action] = reward + Q_future * self.gamma #the estimated score of our action is what we gained + what we can get next with a discount
            targets.append(target[0])
            states.append(state)
        self.model.fit(np.array(states), np.array(targets), epochs=1, batch_size=self.batch_size, verbose=0)
        
    def quick_replay(self): #a version of replay for quickstart
        targets=[]
        states=[]
        if len(self.memory) < 2*self.batch_size: #very early step of the program, there is a risk of overfitting
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.model.predict(state.reshape((1,)+state.shape)) #predicted score for each action for our network, obtained with the "stable" model for stability issues
            target[0][action] = reward
            targets.append(target[0])
            states.append(state)
        self.model.fit(np.array(states), np.array(targets), epochs=1, batch_size=self.batch_size, verbose=0)
        
        
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
            predict=np.exp(np.multiply(predict,4))#this number is relatively arbitrary, the bigger it is, the more we "trust" our net
            predict/=sum(predict)
            aleatar=0.
            #print(predict)
            alea = np.random.random()
            for i in range(len(predict)):
                aleatar+=predict[i]
                if(alea<=aleatar):
                    return(i)
                
    def target_train(self):#copy the weight of the "fast" model to the "slow" one (see double q learning for references)
        self.target_model.set_weights(self.model.get_weights())
    
    def save_model(self, fn):
        print("saving, don't exit the program")
        self.model.save(fn)
        
def main(load=False, steps = 5000, mode=0, epsilon = 1., epsilon_min = 0.05, epsilon_decay = 0.999996, render=True, quickstart=False, steps_quickstart=500): #the function to start the program. load = whether or not to load a previous network, mode 0 : classic exploration, 1 : softmax exploration, 2: argmax, render : show the game or not (can be slower)
    dqn_agent = DQN(load, mode, epsilon, epsilon_min, epsilon_decay)
    step=0
    if quickstart: #quickstart gives an approximation of q-learning based on the actual score instead of the theoretical minimum, but is easier to train as long as the approximation holds
        while step<steps_quickstart:
            done= False
            score=0
            cur_obs=dqn_agent.env.reset()
            prev_obs=cur_obs
            state=np.concatenate((((prev_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis], ((cur_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis]), axis=2) #we create an array containing the last two image of the game, cropped, rescaled and downsized
            step+=1
            while not done:
                reward=0
                tempmem=[]
                while reward==0: #for pong, everytime reward!=0 can be seen as the end of a cycle
                    if render:
                        dqn_agent.env.render()
                    action = dqn_agent.act(state.reshape(1,80,80,2), dqn_agent.env)
                    cur_obs, reward, done, info = dqn_agent.env.step(action+2) #see openai gym for information
                    state2=np.concatenate((((prev_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis], ((cur_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis]), axis=2) #we create an array containing the last two image of the game, cropped, rescaled and downsized
                    tempmem.append([state, action, state2])
                    prev_obs=cur_obs
                    state=state2
                    dqn_agent.quick_replay() #train the model
                score+=reward
                for state, action, state2 in tempmem[::-1]:
                    dqn_agent.remember(state, action,10*reward, state2, True) #done = true fix the reward to be applied instantly
                    reward*=dqn_agent.gamma
                del tempmem
            dqn_agent.save_model("pong.h5")
            print("epsilon :", dqn_agent.epsilon, "score :", score, "model saved")
            score=0
        del dqn_agent.memory
        dqn_agent.memory=deque(maxlen=2000)
        dqn_agent.target_train()
        print("Exiting quickstart")
    while step<steps:
        done= False
        score=0
        cur_obs=dqn_agent.env.reset()
        prev_obs=cur_obs
        state=np.concatenate((((prev_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis], ((cur_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis]), axis=2) #we create an array containing the last two image of the game, cropped, rescaled and downsized
        step+=1
        while not done: #while the game is not over
            reward=0              
            while reward==0: #for pong, everytime reward!=0 can be seen as the end of a cycle
                if render:
                    dqn_agent.env.render()
                action = dqn_agent.act(state.reshape(1,80,80,2), dqn_agent.env)
                cur_obs, reward, done, info = dqn_agent.env.step(action+2) #see openai gym for information
                state2=np.concatenate((((prev_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis], ((cur_obs[34:194,:,1]-72)*-1./164)[::2,::2,np.newaxis]), axis=2) #we create an array containing the last two image of the game, cropped, rescaled and downsized
                dqn_agent.remember(state, action,10*reward, state2, reward!=0) #instead of giving done, giving reward!=0 is much easier for pong
                prev_obs=cur_obs
                state=state2
                dqn_agent.replay() #train the model
            score+=reward
            dqn_agent.target_train() #train the "stable model"
        dqn_agent.save_model("pong.h5")
        print("epsilon :", dqn_agent.epsilon, "score :", score, "model saved")
        score=0
    
    dqn_agent.env.close()