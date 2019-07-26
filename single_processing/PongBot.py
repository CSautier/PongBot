#this is a modification of the main code, to make the two networks share part of their weights
import gym
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import numpy as np
import os.path
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PPO training')
parser.add_argument('--load', default=False, help='Whether or not to load pretrained weights. You must have started an alread trained net for it to work',
                    dest='load', type=str2bool)
parser.add_argument('--render', default=True, help='Whether or not show the game being played (for all the playing processes). This slows the training a bit',
                    dest='render', type=str2bool)

LOSS_CLIPPING=0.1
ENTROPY_LOSS = 1e-3
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 3)), np.zeros((1, 1))

def proximal_policy_optimization_loss(advantage, old_prediction):#this is the clipped PPO loss function, see https://arxiv.org/pdf/1707.06347.pdf
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(K.clip(prob, K.epsilon(), 1-K.epsilon()))))
    return loss

class PPO_agent:
    def __init__(self, load):
        self.learning_rate = 1e-3
        self.maxScore=-21
        if(not load):
            self.ppo_net = self.create_model(self.learning_rate)
        else:
            for score in range(21,-22, -1):
                if os.path.isfile("pong_ppo_"+str(score)+".h5"):
                    self.ppo_net = self.create_model(self.learning_rate, "pong_ppo_"+str(score)+".h5")
                    self.maxScore=score
                    break
            try: self.ppo_net
            except: raise Exception('You need a pretrained net to do this')
    def create_model(self, lr, load_model=None):
        advantage = layers.Input(shape=(1,))
        obtained_prediction = layers.Input(shape=(3,))
        
        input = layers.Input(shape=(80, 80,2))
        x = layers.Conv2D(filters=8, kernel_size=5, activation='relu', padding='valid')(input)
        x = layers.Flatten()(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        mid_output= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(mid_output)
        x= layers.Dense(20, activation="relu")(x)
        actor = layers.Dense(3, activation='softmax', name='actor')(x)

        
        x= layers.Dense(20, activation="relu")(mid_output)
        x= layers.Dense(20, activation="relu")(x)
        critic = layers.Dense(1, name='critic')(x)
        
        
        ppo_net = Model(inputs=[input, advantage, obtained_prediction], outputs=[actor, critic]) #the loss_function requires advantage and prediction, so we feed them to the network but keep them unchanged
        ppo_net.compile(optimizer=Adam(lr), loss={'actor' : proximal_policy_optimization_loss(advantage,obtained_prediction), 'critic' : 'mean_squared_error'}, 
                        loss_weights={'actor': 1e-1, 'critic': 1.})
        if (load_model):
            ppo_net.load_weights(load_model)
        ppo_net.summary()
        
        
        return ppo_net

    def save_model(self, score):
        print("saving, don't exit the program")
        self.ppo_net.save_weights("pong_ppo_"+str(score)+".h5")
    
class Generator(Sequence):
    'Generates data for Keras'
    def __init__(self, render, ppo_agent):
        'Initialization'
        self.env = gym.make('Pong-v0')
        self.done = True
        self.render=render
        self.ppo_agent=ppo_agent
        self.gamma=0.95 #discount factor
        self.score=-21

    def process_frame(self, frame): #cropped and renormalized
        return ((frame[34:194,:,1]-72)*-1./164)[::2,::2]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.done:
            print(int(self.score))
            if int(self.score)>self.ppo_agent.maxScore:
                self.ppo_agent.maxScore = int(self.score)
            self.ppo_agent.save_model(self.ppo_agent.maxScore)
            self.score=0
            self.observation=self.env.reset()
            self.observation = self.process_frame(self.observation)
            self.prev_observation=self.observation
        states_list = [] # shape = (x,80,80)
        action_list=[] # [0,1] or [1,0]
        predict_list=[]
        reward_pred=[]
        advantage_list=[]
        reward_list=[]
        reward=0
        while reward==0: #for pong, everytime reward!=0 can be seen as the end of a cycle, thus we train after them
            if self.render:
                self.env.render()
            state = np.concatenate((self.prev_observation[:,:,np.newaxis], self.observation[:,:,np.newaxis]), axis=2) #we create an array containing the 2 last images of the game
            states_list.append(state)
            predicted = self.ppo_agent.ppo_net.predict([state.reshape(1,80,80,2), DUMMY_VALUE, DUMMY_ACTION])[0][0] #DUMMY sth are required by the network but never used, this is a hack
            predict_list.append(predicted)
            alea = np.random.random()
            aleatar=0
            action=1
            for i in range(len(predicted)): #chose randomly an action according to the probability distribution given by the softmax
                aleatar+=predicted[i]
                if(alea<=aleatar):
                    action=i+1
                    break;
            action_list.append([0,0,0])
            action_list[-1][action-1]=1
            self.prev_observation=self.observation
            self.observation, reward, self.done, info = self.env.step(action) #compute the next step of the game, see openai gym for information
            self.observation = self.process_frame(self.observation)
            reward_list.append(reward)
        self.score+=reward
        for i in range(len(states_list)-2, -1, -1):
            reward_list[i]+=reward_list[i+1] * self.gamma #compute the discounted obtained reward for each step
        x=np.array(states_list)
        reward_array = np.reshape(np.array(reward_list), (len(reward_list), 1))
        reward_pred = self.ppo_agent.ppo_net.predict([x, np.zeros((len(states_list), 1)), np.zeros((len(states_list), 3))])#[1]
        reward_pred=reward_pred[1]
        advantage_list=reward_array-reward_pred
        pr = np.array(predict_list)
        y_true = np.array(action_list) # 1 if we chose up, 0 if down
        X=[x,advantage_list, pr]
        y={'critic' : np.array(reward_list),'actor' :  np.array(y_true)}
        return X, y

def main(args): #the function to start the program. load = whether or not to load a previously trained network, render : show the game or not (can be slower)
    ppo_agent = PPO_agent(args.load)
    generator=Generator(args.render, ppo_agent)
    ppo_agent.ppo_net.fit_generator(generator=generator,  steps_per_epoch=1, epochs=20000, use_multiprocessing=False, workers=0, verbose=2)
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)