# Multiprocessed Keras PPO bot playing Pong on OpenAI's gym

## This code is a **PPO** implementation of a bot playing Atari Pong from game images
### The net is written for Keras (using Tensorflow backend, but this is easily modified)

![Actual Result of the bot](https://github.com/CSautier/PongBot/gifs/demo.gif)
My bot plays in green (right)

## Requirements

* Tensorflow (GPU)
* Numpy
* gym (Atari)
* multiprocessing
* ffmpeg or Pillow if you want to create gifs of the game

### The code has only been tested with
* Python 3.7
* Tensorflow-gpu 1.13

## How to begin the training

* Clone this repository: `git clone `
* Launche the game: `python PongBot.py`

## How to resume the training

Launch the game with the *load* option: `python PongBot.py --load True`

## What to expect of the training

I've got a GTX 1060, and can run 5 processes at once. If your GPU has more memory than mine (3Gb) you can increase the number of processes: `python PongBot.py --processes 10` (for instance).
You will get the feeling of some progress in about an hour, but it takes approximately a day to really train the net.

## I get a *Cuda Memory Error*

This means your GPU is not powerful enough to run as many processes. Try launching the training with fewer processes: `python PongBot.py --processes 2`.
If you still have this problem you should try the non-multiprocessing version, or you can force Keras to run the processes with a CPU (not recommanded and not tested).

## I get a *You need a pretrained net to do this*

This means you are trying to load a pretrained weight without actually having one.

## Useful resources

* https://openai.com/
* https://github.com/Grzego/async-rl
* https://arxiv.org/pdf/1707.06347.pdf


**Feel free to use as much of this code as you want but mention my github if you found this useful**.  
**For more information, you can contact me on my github.**
