layout: post 
comments: true
title: Playing Doom with Deep Reinforcement Learning
excerpt: "" 
date: 2016-07-28 20:00:00 
mathjax: true
---

While deep learning (DL) made a revolution in computer vision (<a href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">Link 1</a>, <a href="https://arxiv.org/pdf/1512.03385v1.pdf">Link 2</a>), a group of smart people from DeepMind decided to combine current reinforcement learning approaches with deep learning neural networks, more precisely to replace Q-Learning function approximator with convolutional neural network. Their paper recieves a lot of attention in AI community, since it is the first time, when a single algorithm successfully learns how to survive in absolutely different environments with different rules and objectives, and in some games, it even outperforms human!
In this topic we will try to solve a very simple Doom (VizDoom port) level using <a href="https://gym.openai.com/envs/DoomBasic-v0">OpenAI's Gym</a> for accessing Doom API, but if you will get the basic idea of how RL works in practice it wouldn't be a such a big deal to move to a more interesting and complicated problems. So let's get started!


## Input data preprocessing
**Resolution.** The original game screen of Doom is 640x480. Since we are building relarively small ConvNet, there is no need in such high resolution, so to speed up our training time, let's scale it down to 64x48.

**Register in-game movements by frame substraction.** CNNs, unlike recurrent neural networks, does not have their own "memory". That means, our agent won't remember where the monster was 1 frame ago, it will only see it's current position. The common hack is to substract current frame from the previous one


## Convolutional Neural Network as a function approximator
CNN; Current Model; Leaky ReLU; Adam.


## Q-learning vs Policy Gradients
TODO


## Diving into implementation details
For implementation I've used a high-level deep learning library <a href="http://keras.io">Keras</a>. 
By default, Anaconda comes with installed Keras. But you can always do it by hand through `pip install keras`.
For those, who prefer more low-level control of what's going on, here it is an alternative implementation of the same script using Google's TensorFlow.
TODO

### CUDA
It's not mandatory (especially for this tutorial) but if you are advanterous and want to speed up your training process it is recommended to install CUDA drivers to get your GPU work (if you have a proper NVIDIA GPU).

## Training
TODO

## Results
TODO

---
## Where should I start learning more about RL?
I would suggest you to start from <a href="http://karpathy.github.io/2016/05/31/rl/">Andrej Karpathy's post</a> about Deep RL applied to a pong game.
Then, to get some intuition you may read through <a href="https://www.nervanasys.com/demystifying-deep-reinforcement-learning">this post</a>, simualteniously watching <a href="http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html">10 David Silver's lectures about RL</a>.
And, of course, a classic <a href="https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html">Richard Sutton's RL book</a>.

For **deep learning** I would recommend you to start from <a href="http://neuralnetworksanddeeplearning.com/">Nielsen's online book</a>.
Later, watch, <a href="http://cs231n.github.io/">read and work through CS231n Stanford lectures</a> (unfortunately official video lectures were removed from youtube, but probably, somewhere, there might be an unofficial one :) ).

# Some usefull paper links
1. <a href="http://arxiv.org/abs/1312.5602">Playing Atari with Deep Reinforcement Learning</a>, 2013, Mnih et al. - DQN.
2. <a href="http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf">Human-level control through deep reinforcement learning</a>, 2015, Mnih et al. - DQN.
3. <a href="http://arxiv.org/pdf/1602.01783v2.pdf">Asynchronous Methods for Deep Reinforcement Learning</a>, 2016, Mnih et. al. - Policy Gradients.
3. <a href="https://webdocs.cs.ualberta.ca/~sutton/papers/SMSM-NIPS99.pdf>Policy Gradient Methods for Reinforcement Learning with Function Approximation</a>, 2000, Sutton et al. - Policy Gradients.
4. <a href="http://jmlr.org/proceedings/papers/v32/silver14.pdf>Deterministic Policy Gradient Algorithms</a>, 2014, Silver et al. - Policy Gradients.
---------------------------