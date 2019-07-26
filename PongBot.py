import gym
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import os.path
from multiprocessing import Pool, Manager
import time
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
parser.add_argument('--processes', default=5, help='Number of processes that plays the game. Note: there will always be a process to learn from it',
                    dest='processes', type=int)
parser.add_argument('--swap_freq', default=20, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience. Note, there is no need for it to be huge', dest='queue_size',
                    type=int)

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

def create_model(weights=None):
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
    ppo_net.compile(optimizer=Adam(1e-3), loss={'actor' : proximal_policy_optimization_loss(advantage,obtained_prediction), 'critic' : 'mean_squared_error'}, 
                    loss_weights={'actor': 1e-1, 'critic': 1.})
    if (weights):
        if type(weights)==str:
            ppo_net.load_weights(weights)
        else:
            ppo_net.set_weights(weights)
#    ppo_net.summary()
    return ppo_net

class Generator():
    #a 'generator' that produces a batch of data to train on
    def __init__(self, render, ppo_net):
        self.env = gym.make('Pong-v0')
        self.done = True
        self.render=render
        self.ppo_net=ppo_net
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
            print('Score: {}'.format(int(self.score)))
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
            predicted = self.ppo_net.predict([state.reshape(1,80,80,2), DUMMY_VALUE, DUMMY_ACTION])[0][0] #DUMMY sth are required by the network but never used, this is a hack
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
            if action==1:
                reward_list.append(reward*0.1+0.001)
            else:
                reward_list.append(reward*0.1)
        self.score+=reward
        for i in range(len(states_list)-2, -1, -1):
            reward_list[i]+=reward_list[i+1] * self.gamma #compute the discounted obtained reward for each step
        x=np.array(states_list)
        reward_array = np.reshape(np.array(reward_list), (len(reward_list), 1))
        reward_pred = self.ppo_net.predict([x, np.zeros((len(states_list), 1)), np.zeros((len(states_list), 3))])#[1]
        reward_pred=reward_pred[1]
        advantage_list=reward_array-reward_pred
        pr = np.array(predict_list)
        y_true = np.array(action_list) # 1 if we chose up, 0 if down
        X=[x,advantage_list, pr]
        y={'critic' : np.array(reward_list),'actor' :  np.array(y_true)}
        return X, y

def train_proc(mem_queue, weight_dict, render):
    #playing process it loads an instance of the model and use it to play the game, then sends the generated batch
    try:
        print('process started')
        import tensorflow as tf
        import tensorflow.keras.backend as K
        #this block enables GPU enabled multiprocessing
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True
        tf.logging.set_verbosity(tf.logging.ERROR)
        #allow_growth tells cuda not to use as much VRAM as it wants (as we nneed extra ram for all the other processes)
        with tf.Session(config=core_config) as session:
            K.set_session(session)
            
            #counter of the current version of the weights
            update=0
            #load the initial weights
            ppo_net = create_model(weight_dict['weights'])
            #a generator that plays a game and returns a batch
            gen = Generator(render, ppo_net)
            while True:
                #check for a new update
                if weight_dict['update']>update:
    #                print('updating player net')
                    #set the counter to the new version
                    update=weight_dict['update']
                    #update the weights
                    ppo_net.set_weights(weight_dict['weights'])
                #stores the weights
                mem_queue.put(gen[0])
                #if we otained the best score so far, store it in the shared memory
                if gen.maxScore>weight_dict['maxScore']:
                    weight_dict['maxScore']=gen.maxScore
    
            session.close()
            K.clear_session()
    except Exception as e: print(e)


def learn_proc(mem_queue, weight_dict, load, swap_freq=10):
    #learning process it creates or load the model, reads batchs from the player and fit the model with them
    try:
        print('process started')
        import tensorflow as tf
        import tensorflow.keras.backend as K
        #this block enables GPU enabled multiprocessing
        core_config = tf.ConfigProto()
        #allow_growth tells cuda not to use as much VRAM as it wants (as we nneed extra ram for all the other processes)
        core_config.gpu_options.allow_growth = True
        tf.logging.set_verbosity(tf.logging.ERROR)
        with tf.Session(config=core_config) as session:
            K.set_session(session)
            #whether or not to load a previous network
            if(not load):
                ppo_net = create_model()
                weight_dict['maxScore']=-21
            else:
                #load the network that scored the best so far
                for score in range(21,-22, -1):
                    if os.path.isfile("pong_ppo_"+str(score)+".h5"):
                        ppo_net = create_model("pong_ppo_"+str(score)+".h5")
                        weight_dict['maxScore']=score
                        break
                try: ppo_net
                except: raise Exception('You need a pretrained net to do this')
            #counter of the current update of the weights
            weight_dict['update']=0
            weights = ppo_net.get_weights()
            #stores weights in the global variable for the other processes to access
            weight_dict['weights']=weights
            while True:
                for i in range(swap_freq):
                    batch, labels = mem_queue.get()
#                    print('received batch')
                    ppo_net.fit(batch, labels, verbose=0)
                print('updating net')
                weight_dict['weights']=ppo_net.get_weights()
                weight_dict['update']+=1
                #save the weights in a file, to load it later. The file contains the best score ever obtained
                ppo_net.save_weights("pong_ppo_{}.h5".format(weight_dict['maxScore']))
    except Exception as e: print(e)
 
def init_worker():
    #Allows to pass a few signals to the processes, such as keyboardInterrupt
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main(args): 
    #create shared variables between all the processes
    manager = Manager()
    #contains information about the weights
    weight_dict = manager.dict()
    #a queue of batches to be fed to the training net
    mem_queue = manager.Queue(args.queue_size)
    
    #initializes all workers
    pool = Pool(args.processes+1, init_worker)
    try:
        #the learner set the weights and store them in the weight_dict
        pool.apply_async(learn_proc, (mem_queue, weight_dict, args.load, args.swap_freq))
        #wait for the learner to finish it's storing
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        for i in range(args.processes):
            #starts the player
            pool.apply_async(train_proc, (mem_queue, weight_dict, args.render))

        #never currently called, this would be usefull if the processes had an end
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)