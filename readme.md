This code is a dqn implementation of a bot playing Atari Pong from game images.
The net is written for Tensorflow's implementation of Keras.

It uses some clever speedups such as a "quickstart" to train the model much faster in the beginning.
The "mode 1" allows to run the exploration according to a softmax function, which gives a nice probabilistic way to adress the best output.

minimalist is a tiny network, that should still be able to score pretty decently

minimalist algorithm contains a few improvements that could be added to the main

For more information, you can contact me on my github.

Note: If you run short on RAM, you can reduce self.memory maxlen
