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
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import os.path

LOSS_CLIPPING=0.2
ENTROPY_LOSS = 1e-6
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 2)), np.zeros((1, 1))

def proximal_policy_optimization_loss(advantage, old_prediction):#this is the clipped PPO loss function, see https://arxiv.org/pdf/1707.06347.pdf
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage + ENTROPY_LOSS * -(prob * K.log(K.clip(prob, 0.001, 0.999))))
    return loss


class PPO_agent:
    def __init__(self, load):
        self.env = gym.make('Pong-v0')
        self.critic_learning_rate = 1e-4
        self.actor_learning_rate = 1e-4
        self.gamma=0.95 #discount factor
        self.maxScore=-21
        if(not load):
            #there are 2 networks : actor and critic, as described in the PPO papers.
            self.actor, self.actor_weights = self.create_actor() #actor_weights allows us to save the network
            self.critic = self.create_critic()

        else:
            for score in range(21,-22, -1):
                if os.path.isfile("pong_ppo_critic_"+str(score)+".h5"):#check the best score and load it
                    self.critic = load_model("pong_ppo_critic_"+str(score)+".h5")
                    self.critic.compile(loss="mean_squared_error", optimizer=Adam(lr=self.critic_learning_rate))
                    self.actor_weights = load_model("pong_ppo_actor_"+str(score)+".h5")
                    advantage = layers.Input(shape=(1,))
                    obtained_prediction = layers.Input(shape=(2,))
    
                    self.actor = Model(inputs=[self.actor_weights.input, advantage, obtained_prediction], outputs=self.actor_weights.output) #keras does not load the loss function, so we have to recreate it
                    self.actor.compile(optimizer=Adam(lr=self.actor_learning_rate),loss=proximal_policy_optimization_loss(advantage,obtained_prediction))
                    self.actor.summary()
                    self.maxScore=score
                    break

    def create_actor(self): #create the actor model, to chose the action
        input = layers.Input(shape=(80, 80,2))
        x = layers.Conv2D(filters=8, kernel_size=5, activation='relu', padding='same')(input)
        #x= layers.MaxPooling2D(pool_size=(4,1), strides=None, padding='valid', data_format=None)(x) #the max pooling is along the height axis, so that we wont "miss" the ball
        x = layers.Flatten()(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        output = layers.Dense(2, activation='softmax')(x)
        advantage = layers.Input(shape=(1,))
        obtained_prediction = layers.Input(shape=(2,))
        weight_model=Model(inputs=input, outputs=output)
        model = Model(inputs=[input, advantage, obtained_prediction], outputs=output) #the loss_function requires advantage and prediction, so we feed them to the network but keep them unchanged
        model.compile(optimizer=Adam(lr=self.actor_learning_rate),loss=proximal_policy_optimization_loss(advantage,obtained_prediction))
        model.summary()
        return model, weight_model

    def create_critic(self): 
        input = layers.Input(shape=(80, 80,2))
        x = layers.Conv2D(filters=8, kernel_size=5, activation='relu', padding='same')(input)
        #x= layers.MaxPooling2D(pool_size=(4,1), strides=None, padding='valid', data_format=None)(x) #the max pooling is along the height axis, so that we wont "miss" the ball
        x = layers.Flatten()(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        output = layers.Dense(1)(x)
        model = Model(input, output)
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.critic_learning_rate))
        model.summary()
        return model

    def save_model(self, score):
        print("saving, don't exit the program")
        self.actor_weights.save("pong_ppo_actor_"+str(score)+".h5")#this way the actor model will save
        self.critic.save("pong_ppo_critic_"+str(score)+".h5")

    def process_frame(self, frame): #crop and renormalize a frame
        return ((frame[34:194,:,1]-72)*-1./164)[::2,::2]

def main(load=False, steps = 20000, render=False): #the function to start the program. load = whether or not to load a previously trained network, render : show the game or not (can be slower)
    ppo_agent = PPO_agent(load)
    step=0
    lastScore=-21
    ppo_agent.maxScore=-21
    while step<steps: #number of games to play
        done= False
        score=0
        observation=ppo_agent.env.reset()
        observation = ppo_agent.process_frame(observation)
        prev_observation=observation
        step+=1
        while not done:
            states_list = [] # shape = (x,80,80)
            up_or_down_action_list=[] # [0,1] or [1,0]
            predict_list=[]
            reward_pred=[]
            advantage_list=[]
            reward_list=[]
            reward=0

            while reward==0: #for pong, everytime reward!=0 can be seen as the end of a cycle, thus we train after them
                if render:
                    ppo_agent.env.render()
                state = np.concatenate((prev_observation[:,:,np.newaxis], observation[:,:,np.newaxis]), axis=2) #we create an array containing the 2 last images of the game
                states_list.append(state)
                predicted = ppo_agent.actor.predict([state.reshape(1,80,80,2), DUMMY_VALUE, DUMMY_ACTION])[0] #DUMMY sth are required by the network but never used, this is a hack
                predict_list.append(predicted)
                alea = np.random.random()
                aleatar=0
                action=2
                for i in range(len(predicted)): #chose randomly an action according to the probability distribution given by the softmax
                    aleatar+=predicted[i]
                    if(alea<=aleatar):
                        action=i+2
                        break;
                if action==2:
                    up_or_down_action_list.append([1,0])
                else:
                    up_or_down_action_list.append([0,1])
                prev_observation=observation
                observation, reward, done, info = ppo_agent.env.step(action) #compute the next step of the game, see openai gym for information
                observation = ppo_agent.process_frame(observation)
                reward_list.append(reward)
            score+=reward
            for i in range(len(states_list)-2, -1, -1):
                reward_list[i]+=reward_list[i+1] * ppo_agent.gamma #compute the discounted obtained reward for each step
            x=np.array(states_list)
            reward_array = np.reshape(np.array(reward_list), (len(reward_list), 1))
            reward_pred = ppo_agent.critic.predict(x)
            advantage_list=reward_array-reward_pred
#                print(reward_pred[-1])  #this print is useful to see if the net converges
#            print(predict_list[len(predict_list)//2]) #this print is useful to see if the actor is not too extreme, or not learning anything
            pr = np.array(predict_list)
            y_true = np.array(up_or_down_action_list) # 1 if we chose up, 0 if down
            ppo_agent.actor.fit(x=[x,advantage_list, pr],y=y_true, batch_size=16, verbose = False) #fit the networks
            ppo_agent.critic.fit(x=x, y=reward_list, batch_size=16, epochs=1, verbose = False)
        lastScore=int(score)
        if lastScore>ppo_agent.maxScore:
            ppo_agent.maxScore=lastScore
        ppo_agent.save_model(ppo_agent.maxScore)
        print("Score: ",lastScore)
        score=0
    ppo_agent.env.close()
