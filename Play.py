"""
This script creates nice gifs from the game state
It does not use multiprocessing
It needs a pretrained network
"""

import gym
from PongBot import create_model
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


#import tensorflow as tf
#import tensorflow.keras.backend as K
#num_cores = 1
#num_CPU = 1
#num_GPU = 1
#config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
#                        inter_op_parallelism_threads=num_cores, 
#                        allow_soft_placement=True,
#                        device_count = {'CPU' : num_CPU,'GPU' : num_GPU})
#session = tf.Session(config=config)
#K.set_session(session)

for score in range(21,-22, -1):
    if os.path.isfile("pong_ppo_"+str(score)+".h5"):
        ppo_net = create_model("pong_ppo_"+str(score)+".h5")
        break
    
try: ppo_net
except: raise Exception('You need a pretrained net to do this')

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    try:
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=33)
        anim.save('gifs/gameplay.gif')
    except:
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=33)
        anim.save('gifs/gameplay.gif', writer=animation.PillowWriter(fps=40))
        
        
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 3)), np.zeros((1, 1))

def process_frame(frame): #cropped and renormalized
    return ((frame[34:194,:,1]-72)*-1./164)[::2,::2]

frames=[]
env=gym.make('Pong-v0')
frames.append(env.reset())
observation = process_frame(frames[-1])
prev_observation=observation
done = False
while not done: #for pong, everytime reward!=0 can be seen as the end of a cycle, thus we train after them
    env.render()
    state = np.concatenate((prev_observation[:,:,np.newaxis], observation[:,:,np.newaxis]), axis=2) #we create an array containing the 2 last images of the game
    predicted = ppo_net.predict([state.reshape(1,80,80,2), DUMMY_VALUE, DUMMY_ACTION])[0][0] #DUMMY sth are required by the network but never used, this is a hack
    alea = np.random.random()
    aleatar=0
    action=1
    for i in range(len(predicted)): #chose randomly an action according to the probability distribution given by the softmax
        aleatar+=predicted[i]
        if(alea<=aleatar):
            action=i+1
            break;
#    action=np.argmax(predicted)+1
    prev_observation=observation
    observation, reward, done, info = env.step(action) #compute the next step of the game, see openai gym for information
    frames.append(observation)
    observation = process_frame(observation)
display_frames_as_gif(frames)