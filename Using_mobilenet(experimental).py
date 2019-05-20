#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:58:04 2019

@author: cstr
"""

import gym
from tensorflow.keras import layers
from tensorflow.keras import Model
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import numpy as np
import random
from collections import deque
from tensorflow.keras.applications import MobileNetV2

class DQN:
    def __init__(self, load, mode):
        self.env = gym.make('Pong-v0')
        self.memory = deque(maxlen=2000)
        self.learning_rate = 1e-7
        self.lenmemory=1000
        self.batch_size = 5
        self.gamma=0.9
        self.epsilon = 1.
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99993
        self.mode=mode
        if(not load):
            self.model = self.create_model()
            self.target_model = self.create_model()
        else:
            self.model = load_model('pong_mobilenet.h5')
            self.target_model = load_model('pong_mobilenet.h5')
            print("Model loaded")
            self.model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.learning_rate))  
        
        
    def create_model(self):
        mobilenet = MobileNetV2(weights='imagenet', input_shape=(160, 160, 3), include_top=False)
        for layer in mobilenet.layers:
            layer.trainable=False
        x =  mobilenet.layers[-2].output
        x = layers.Flatten()(x)
        output = layers.Dense(2)(x)
        model = Model(inputs= mobilenet.input, outputs=output)
        model.summary()
        model.compile(loss="mean_squared_error", optimizer=SGD(lr=self.learning_rate))        
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
                    
            
    def replay(self): 
        targets=[]
        states=[]
        if len(self.memory) < 2*self.batch_size: 
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state.reshape((1,)+state.shape))
            if done:
                print(target)
                target[0][action] = reward
                target[0][0]=reward
                target[0][0]=reward
            else:
                Q_future = max(self.target_model.predict(new_state.reshape((1,)+state.shape))[0])
                target[0][action] = reward + Q_future * self.gamma
            targets.append(target[0])
            states.append(state)
        self.model.fit(np.array(states), np.array(targets), epochs=1, batch_size=self.batch_size, verbose=0)
                  
        
    def act(self, state, env):
        if self.mode==2:
            return np.argmax(self.model.predict(state)[0])
        elif self.mode==0:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.random() < self.epsilon:
                return random.randint(0,1)
            else:
                return np.argmax(self.model.predict(state)[0])
        #case mode == 1
        predict=self.model.predict(state)[0]
        if (min(predict)==predict.all()):
            return random.randint(0,1)
        t=1.-sum(predict)
        predict+=t/len(predict)
        print(predict)
        aleatar=0.
        #print(predict)
        alea = np.random.random()
        for i in range(len(predict)):
            aleatar+=predict[i]
            if(alea<=aleatar):
                return(i)
                
    def target_train(self): ##copy pour remplacer ça ???
        self.target_model.set_weights(self.model.get_weights())
    
    def save_model(self, fn):
        self.model.save(fn)
        
def main(load=False, steps = 10000, mode=0): #mode 0 : epsilon, 1 : proba explo, 2 : argmax
    dqn_agent = DQN(load, mode)
    step=1
    while step<=steps:
        done= False
        score=0
        cur_obs=dqn_agent.env.reset()
        prev_obs2=cur_obs
        prev_obs1=cur_obs
        state=np.concatenate((((prev_obs2[34:194,:,1]-72)*-1./164)[:,:,np.newaxis], ((prev_obs1[34:194,:,1]-72)*-1./164)[:,:,np.newaxis], ((cur_obs[34:194,:,1]-72)*-1./164)[:,:,np.newaxis]), axis=2)
        step+=1
        while not done:
            reward=0
            while reward==0:
                #dqn_agent.env.render()
                action = dqn_agent.act(state.reshape(1,160,160,3), dqn_agent.env)
                cur_obs, reward, done, info = dqn_agent.env.step(action+2)
                state2=np.concatenate((((prev_obs2[34:194,:,1]-72)*-1./164)[:,:,np.newaxis], ((prev_obs1[34:194,:,1]-72)*-1./164)[:,:,np.newaxis], ((cur_obs[34:194,:,1]-72)*-1./164)[:,:,np.newaxis]), axis=2)
                dqn_agent.remember(state, action,100*reward, state2, reward!=0)
                prev_obs2=prev_obs1
                prev_obs1=cur_obs
                state=state2
                dqn_agent.replay()            
            score+=reward
            dqn_agent.target_train()
        dqn_agent.save_model("pong_mobilenet.h5")
        print("epsilon :", dqn_agent.epsilon, "score :", score, "saving model")
        score=0
        #print("Reward:", sum_reward, "Epsilon:", max(dqn_agent.epsilon, dqn_agent.epsilon_min))
        #game_length=0
        #sum_reward=0
    
    dqn_agent.save_model("pong_mobilenet.h5")
    dqn_agent.env.close()