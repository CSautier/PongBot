import gym
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import numpy as np
import os.path
from multiprocessing import Pool, Manager

import time

LOSS_CLIPPING=0.1
ENTROPY_LOSS = 1e-3
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 2)), np.zeros((1, 1))

def proximal_policy_optimization_loss(advantage, old_prediction):#this is the clipped PPO loss function, see https://arxiv.org/pdf/1707.06347.pdf
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(K.clip(prob, K.epsilon(), 1-K.epsilon()))))
    return loss

class PPO_agent:
    def __init__(self, weights=None):
        self.learning_rate = 1e-3
        if weights is None:
            self.ppo_net = self.create_model(self.learning_rate)
        else:
            self.ppo_net = self.create_model(self.learning_rate, weights)

    def create_model(self, lr, weights=None):
        advantage = layers.Input(shape=(1,))
        obtained_prediction = layers.Input(shape=(2,))
        
        input = layers.Input(shape=(80, 80,2))
        x = layers.Conv2D(filters=8, kernel_size=5, activation='relu', padding='valid')(input)
        x = layers.Flatten()(x)
        x= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(x)
        mid_output= layers.Dense(20, activation="relu")(x)
        x= layers.Dense(20, activation="relu")(mid_output)
        x= layers.Dense(20, activation="relu")(x)
        actor = layers.Dense(2, activation='softmax', name='actor')(x)

        
        x= layers.Dense(20, activation="relu")(mid_output)
        x= layers.Dense(20, activation="relu")(x)
        critic = layers.Dense(1, name='critic')(x)
        
        
        ppo_net = Model(inputs=[input, advantage, obtained_prediction], outputs=[actor, critic]) #the loss_function requires advantage and prediction, so we feed them to the network but keep them unchanged
        ppo_net.compile(optimizer=Adam(lr), loss={'actor' : proximal_policy_optimization_loss(advantage,obtained_prediction), 'critic' : 'mean_squared_error'}, 
                        loss_weights={'actor': 1e-1, 'critic': 1.})
        if (weights):
            if type(weights)==str:
                ppo_net.load_weights(weights)
            else:
                ppo_net.set_weights(weights)
        ppo_net.summary()
        
        
        return ppo_net

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
        self.maxScore=-21

    def process_frame(self, frame): #cropped and renormalized
        return ((frame[34:194,:,1]-72)*-1./164)[::2,::2]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.done:
            if int(self.score)>self.maxScore:
                self.maxScore=int(self.score)
            print(int(self.score))
            self.score=0
            self.observation=self.env.reset()
            self.observation = self.process_frame(self.observation)
            self.prev_observation=self.observation
        states_list = [] # shape = (x,80,80)
        up_or_down_action_list=[] # [0,1] or [1,0]
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
            self.prev_observation=self.observation
            self.observation, reward, self.done, info = self.env.step(action) #compute the next step of the game, see openai gym for information
            self.observation = self.process_frame(self.observation)
            reward_list.append(reward)
        self.score+=reward
        for i in range(len(states_list)-2, -1, -1):
            reward_list[i]+=reward_list[i+1] * self.gamma #compute the discounted obtained reward for each step
        x=np.array(states_list)
        reward_array = np.reshape(np.array(reward_list), (len(reward_list), 1))
        reward_pred = self.ppo_agent.ppo_net.predict([x, np.zeros((len(states_list), 1)), np.zeros((len(states_list), 2))])#[1]
        reward_pred=reward_pred[1]
        advantage_list=reward_array-reward_pred
        pr = np.array(predict_list)
        y_true = np.array(up_or_down_action_list) # 1 if we chose up, 0 if down
        X=[x,advantage_list, pr]
        y={'critic' : np.array(reward_list),'actor' :  np.array(y_true)}
        return X, y

def train_proc(mem_queue, weight_dict):
    try:
        print('process started')
        import tensorflow as tf
        import tensorflow.keras.backend as K
        #this block enables GPU enabled multiprocessing
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.Session(config=core_config)
        K.set_session(session)
        
        update=0
        ppo_agent = PPO_agent(weight_dict['weights'])
        gen = Generator(True, ppo_agent)
        while True:
            if weight_dict['update']>update:
                print('updating net')
                update=weight_dict['update']
                ppo_agent.ppo_net.set_weights(weight_dict['weights'])
            mem_queue.put(gen[0])
            if gen.maxScore>weight_dict['maxScore']:
                weight_dict['maxScore']=gen.maxScore

        session.close()
        K.clear_session()
    except Exception as e: print(e)


def learn_proc(mem_queue, weight_dict, load, latency=10):
    try:
        print('process started')
        import tensorflow as tf
        import tensorflow.keras.backend as K
        #this block enables GPU enabled multiprocessing
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.Session(config=core_config)
        with tf.Session(config=core_config) as session:
            K.set_session(session)
            
            if(not load):
                ppo_agent = PPO_agent()
                weight_dict['maxScore']=-21
            else:
                for score in range(21,-22, -1):
                    if os.path.isfile("pong_ppo_multiproc_"+str(score)+".h5"):
                        ppo_agent = PPO_agent("pong_ppo_multiproc_"+str(score)+".h5")
                        weight_dict['maxScore']=score
                        break
            weight_dict['update']=0
            weights = ppo_agent.ppo_net.get_weights()
            weight_dict['weights']=weights
            while True:
                for i in range(latency):
                    batch, labels = mem_queue.get()
                    print('received batch')
                    ppo_agent.ppo_net.fit(batch, labels, verbose=0)
                weight_dict['weights']=ppo_agent.ppo_net.get_weights()
                weight_dict['update']+=1
                ppo_agent.ppo_net.save_weights("pong_ppo_multiproc_{}.h5".format(weight_dict['maxScore']))
    except Exception as e: print(e)
 
def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    processes=5
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(512)
    pool = Pool(processes+1, init_worker)
    
    try:
        pool.apply_async(learn_proc, (mem_queue, weight_dict, True, 10))
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        for i in range(processes):
            pool.apply_async(train_proc, (mem_queue, weight_dict))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    main()