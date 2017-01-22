---
layout: post 
comments: true
title: Asynchronous Deep Reinforcement Learning from pixels
excerpt: "I'll implement and explain an idea of Asynchronous one-step Q-Learning. As an example will be trained an agent to play in Atari SpaceInvaders game, using just a raw pixels!" 
date: 2016-11-03 22:00:00 
mathjax: true
---

**Deep Reinforcement Learning** has recently become a really hot area of research, due to the huge amount of breakthroughs in last couple of years. Such explosion started by a group of scientists from a start-up company called DeepMind (later it was acquired by Google), who decided to apply current deep learning progress to existing reinforcement learning (RL) approaches. The result paper [Playing Atari with Deep Reinforcement Learning", Mnih et al., 2013](https://arxiv.org/abs/1312.5602) recieves a lot of attention in Artificial Intelligence (AI) community, since it is the first time, when a single algorithm, using only raw pixels observations, successfully learns how to survive in absolutely different evironments, with different rules and objectives, and in some of the games, it even outperforms human!

**Many improvements** have been made to Deep Q-Network (DQN) since 2013. In this topic we will implement Google DeepMind's asynchronous one-step Q-Learning method, presented in [Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016.](https://arxiv.org/abs/1602.01783), with [OpenAI's Gym](https://gym.openai.com/) classic Atari 2600 SpaceInvaders game (however it can work with any OpenAI Gym environment with raw visual input).  
Although, the main breakthrough of their paper policy-based *Asynchronous Advantage Actor-Critic Network (A3C)*, which outperforms value-based Q-Learning methods in both data efficiency and accuracy, it won't be covered in current post.

*Update 18.11.2016:* current state-of-the-art introduced in their new paper ["Reinforcement Learning with Unsupervised Auxiliary Tasks"](https://arxiv.org/pdf/1611.05397.pdf) (UNREAL), which is also based on A3C method.

**For implementation** were used a deep learning [TensorFlow](http://tensorflow.org) and [Keras](https://keras.io/) libraries.
Code used in this topic can be found at my [github repository](https://github.com/dbobrenko/async-deeprl). All requirements are listed [here](https://github.com/dbobrenko/async-deeprl#requirements).

{% include image.html
    img="/assets/posts/async-deeprl/si_sample.gif"
    title="Trained agent plays SpaceInvaders Atari 2600 game"
    caption="Figure 1: Example of agent playing OpenAI Gym SpaceInvaders."
%}

## Basic theory
Since purpose of this post is to overview and gain intuition in Deep RL basics, all deep learning stuff will be discussed very briefly, instead focusing on reinforcement learning ([skip this boring theory!](#Implementation)).


**Rewards**. Usually, all reinfocement learning problems are based on rewards. The higher reward you recieve, the better you are doing. Though, rewards are not always immediate - there might be a delay between correct action and reward in a few milliseconds, seconds or even hours (in our case timesteps). And here comes first challenge of reinforcement learning called **credit assignment problem** - how can we decide what exactly action leads to the received reward? One of the most used methods to solve this problem called **discounted future rewards**. The main idea is to discount all future rewards by the factor of $$\gamma$$:

$$R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{n-1} r_n,$$

which can be rewritten as:  

$$R_t = r_t + \gamma R_{t+1},$$

where $$\gamma$$ usually equals to 0.9 or 0.99 - the further reward from current time step the more it will be discounted.

You may ask:

1. Why does the future rewards are important? Can't we just take into account only **immediate rewards** (i.e. $$\gamma = 0$$)?

   *Firstly, predicting future rewards gives agent an ability to think "three steps ahead".*
*Secondly, in almost all games and scenarios, first actions are more important than later ones. That's why rewards for earlier actions include all rewards for further actions.*
*And finally, rewards in many games are delayed to the end of the game, so without reward discounting, all actions except last one will have zero reward values.*
2. Then why we just dont take **total discounted future reward** (i.e. $$\gamma = 1$$)?

   *The more we go in the future, the more uncertain we get. Imagine you are playing poker or any card game - there will be no guarantee that actions that lead you to the states in the past will lead you to the same states in the future, and the further you go, the smaller probability to repeat the same sequence of states.*

**Deep Q Network (DQN)** is probably one of the most famous deep reinforcement learning algorithms nowadays, which uses a core idea of **Q-learning** ([Watkins and Dayan, 1992](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)).  
Classic Q-learning algorithm contains a function approximator $$Q(s_t, a_t) = \mathbb E[R_t\|s_t, a_t]$$, which predicts *maximum discounted reward if we will perform action `a` in state `s`*. In Q-learning given function approximator represented as a table (called Q-table), where rows - all possible states, columns - all available in-game actions. During learning, such table fills with *maximum discounted rewards* for each action in each state.  
Since we will learn from raw screen pixels, even with resizing and preprocessing game screen there will be an extremely huge number of all possible states in Q-table. Concretely, in our case, where will be $$256^{84 \cdot 84 \cdot 4}$$ $$\approx  1.4\times10^{67970}$$ possible states in table, multiplied by the number of actions, which is $$\approx 10^{67961}$$ GB of RAM memory (4 byte float).  
And that is where comes Deep Q-Network, replacing huge and hulking Q-table with relatively small deep neural network. The main idea of DQN is to compress Q-table by learning to recognize in-game objects and their behavior, in order to predict **reward for each action** given the *state* (example is shown in figure 5). When rewards for all possible actions in current state recieved, it becomes really easy to play - just choose an action with the highest expected reward!  
Q-function can be represented as a recurrent equation, also called **Bellman equation**:

$$Q(s_t, a_t) = r_t + \gamma max_{a_{t+1}} Q(s_{t+1}, a_{t+1}),$$

where $$s_t$$ - state (in our case game screen),  
$$a_t$$ - action to execute (in our case it's one of the {no operation, left, right} actions),  
$$r_t$$ - immediate reward from environment after performing action $$a_t$$ in state $$s_t$$,  
$$\gamma$$ - discount factor.  
Expression $$max_{a_{t+1}} Q(s_{t+1}, a_{t+1})$$ means "choose maximum reward value over predicted rewards per each action by Q-function for given next state".

**Loss function.** Since DQN learns to predict continuous reward values for each action in the action space - it can be interpreted as a regression task. That's why we will define mean squared error loss function for our neural network:

$$L = (r + \gamma max_{a_{t+1}} Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))^2,$$

where $$r + \gamma max_{a_{t+1}} Q(s_{t+1}, a_{t+1})$$ is ground-truth $$y$$,  
$$Q(s_t, a_t)$$ is our current prediction $$\hat y$$.  
Intuitively, current loss function optimizes neural network so it's predictions will be equal to the reward $$r_t$$ for given state $$t$$ **plus** maximum **expected** discounted reward $$R_{t+1}$$ for the next state $$t+1$$.  
And now, if you will think about maximum discounted reward for the next state $$t+1$$, you will find that it also includes maximum discounted reward $$R_{t+2}$$ for the next state $$t+2$$ and so on up to the terminal state.

**Ok. But how it can work?** That seems to be insane, especially for those, who are familiar with supervised learning. Of course, at early iterations, an approximation of $$Q(s_{t+1}, a_{t+1})$$ will return an absolute garbage, however, over a long time of training (with right hyperparameters and other DQN tricks), the prediction of future expected rewards will become more and more accurate, and finally, it will converge ([a proof of shallow Q-learning convergence](http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf)).

**Asynchronous one-step and n-step Q-Learning**. The main change they made to DQN since 2013, as you might might guess - is asynchronous training in multiple game environments at the same time. Such approach significantly speeds-up convergence, and allows us to train it on a single multicore CPU instead of GPU (compared to vanilla DQN and other deep RL methods). Another important change is in replay memory - there is no need in replaying past experience anymore, since parallel actor-learners happens to be in a different independent environments, which leads to a similar stabilizing effect, as in case of training on random samples from past experience (such hack is necessary for reducing temporal correlations between consecutive updates; for details of experience replay techniques you may read through [Prioritized Experience Replay, T. Schaul et al., 2016](https://arxiv.org/abs/1511.05952)).

They have presented two versions of asynchronous deep Q-Learning: *one-step* and *n-step* Q-learning. The main difference, is that n-step explicitly computes n-step returns by predicting expected discounted future reward only after `n` steps, backpropagating predicted value on earlier actions, instead of predicting it after each step.  
In this topic I will walk through one-step version.
{% include image.html
    img="/assets/posts/async-deeprl/onestep_alg.jpg"
    title="Asynchronous Q-Learning algorithm pseudo-code"
    caption="Figure 2: Asynchronous Q-Learning algorithm pseudo-code (Mnih et al., 2016)."
%}

## Tips and Tricks
**Action repeat** is a nice feature that will help to speed-up training process. Since neighbour frames are almost identical to each other, we will repeat last action on the next 4 frames.  
*Keep in mind, that some games have "rounds" (most Atari games do), in order to avoid repeating actions from last game in new one, and not to predict expected future rewards for current game, based on state from the next game, we should handle end of these rounds as terminal states*.  
Action repeat implementation code ([full code](https://github.com/dbobrenko/async-deeprl/blob/master/asyncrl/environment.py#L143)):

```python
def step(env, action_index, action_repeat=4):
"""Executes an action in OpenAI Gym environment
   and repeats it on the next X frames"""
accum_reward = 0
for _ in range(action_repeat):
    s, r, terminal, info = env.step(action_index)
    accum_reward += r
    if terminal:
        break
    # some preprocess function with resizing, turning to grayscale, etc
    # s = preprocess(s, w, h, channels)
return s, accum_reward, terminal, info
```


**Preprocessing input screen.** Since we are using ConvNets - they have no internal memory, unlike recurrent neural networks. Without having information about previous frames - agent won't be able to infer the velocity of game objects.  
In DeepMind paper they solve this problem by taking last four screen images, converting them to grayscale, resizing to 84x84 and stacking together. As shown on figure 3 and 4, having such history about previous frames gives us an information about objects velocity. Combined with action repeat approach, we will stack only every 4th frame, so the input to the network will be: 1st, 5th, 9th and 13th frame ([implementation](https://github.com/dbobrenko/async-deeprl/blob/master/asyncrl/environment.py#L126)).  
{% include image.html
    img="/assets/posts/async-deeprl/input_pong.png"
    title="Pong input"
    caption="Figure 3: Example of Pong input screen."
%}
{% include image.html
    img="/assets/posts/async-deeprl/input_si.png"
    title="SpaceInvaders input"
    caption="Figure 4: Example of SpaceInvaders input screen."
%}


**Exploration vs. Exploitation** is yet another well-known challenge in reinforcement learning. It's about a struggle between **following already discovered strategy** or **exploring new one**, maybe better that current. In current paper, they sampled the minimum exploration rate $$\epsilon$$ from a distribution taken three values: `{0.1, 0.01, 0.5}` with probabilities `{0.4, 0.3, 0.3}` respectively, separately for each learner thread. During course of training, inital $$\epsilon$$ anneals from 1 to sampled minimum epsilon value over 4 millions of global frames.

## TensorFlow implementation<a name="Implementation"></a>
TensorFlow sometimes feels a bit low level and verbose. There are a lot of wrappers to reduce code, few of them: [keras](https://keras.io/) (used in this post), [slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim), [tflearn](http://tflearn.org/getting_started/).

**Agent** is the first thing we should start from our implementation. It consists of two models - **online model** and **target model**. First one predicts, and learns to predict rewards per action for given state; second one predicts expected future rewards for the next state, used for future reward discounting. Periodically, online model updates target model by copying it's weights. Such approach was introduced in [Deep Reinforcement Learning with Double Q-learning, van Hasselt et al. (2015)](https://arxiv.org/abs/1509.06461) paper, and aims to impove DQN performance. 

First, let's define network architecture ([full code](https://github.com/dbobrenko/async-deeprl/blob/master/asyncrl/agent.py)):

```python
import tensorflow as tf
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
action_size = 3 # depends on the environment settings

def build_model(h, w, channels, fc3_size=256):
    state = tf.placeholder('float32', shape=(None, h, w, channels))
    inputs = Input(shape=(h, w, channels,))
    model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4),
                          activation='relu', border_mode='same')(inputs)
    model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2),
                          activation='relu', border_mode='same')(model)
    model = Flatten()(model)
    model = Dense(output_dim=fc3_size, activation='relu')(model)
    # dropout was skipped for sake of simplicity
    out = Dense(output_dim=action_size, activation='linear')(model)
    model = Model(input=inputs, output=out)
    qvalues = model(state)
    return model, state, qvalues
```


In the original implementation they've used [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) optimizer with shared statistics across threads, `decay=0.99`, `epsilon=0.1` and linearly annealing learning rate over the course of training. To simplify, I've replaced all of this stuff with [Adam](https://arxiv.org/abs/1412.6980) optimizer (however it can be more unstable compared to shared RMSProp):

```python
w = 84 # screen width
h = 84 # screen height
channels = 4 # number of stacked frames, used as network's memory
with tf.variable_scope('network'):
    action = tf.placeholder('int32', [None], name='action')
    reward = tf.placeholder('float32', [None], name='reward')
    model, state, q_values = build_model(h, w, channels)
    weights = model.trainable_weights
with tf.variable_scope('optimizer'):
    action_onehot = tf.one_hot(action, action_size,
                               1.0, 0.0, name='action_onehot')
    action_q = tf.reduce_sum(tf.mul(q_values, action_onehot),
                             reduction_indices=1)
    loss = tf.reduce_mean(tf.square(reward - action_q))
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.minimize(loss, var_list=weights)
with tf.variable_scope('target_network'):
    t_model, target_state, target_q_values = build_model(h, w, channels)
    t_w = t_model.trainable_weights
with tf.variable_scope('target_update'):
    target_update = [t_w[i].assign(weights[i]) for i in range(len(t_w))]
```


Now let's wrap all TensorFlow operations into easy-to-read functions:

```python
def update_target():
    sess.run(target_update)

def predict_rewards(states):
    return sess.run(q_values, {state: states}).flatten()

def predict_target(states):
    return np.max(sess.run(target_q_values, {target_state: states}).flatten())

def train(states, actions, rewards):
    sess.run(train_op, feed_dict={
        state: states,
        action: actions,
        reward: rewards
    })
```


And finally, **training loop** python pseudo-code (defined in [run_dqn.py](https://github.com/dbobrenko/async-deeprl/blob/master/run_dqn.py)).

```python
T = 0
def learner_thread():
    # global shared frame step
    global T
    # randomly sample minimum epsilon from given distribution
    eps_min = random.choice(4 * [0.1] + 3 * [0.01] + 3 * [0.5])
    while T < total_frames:
        batch_states, batch_rewards, batch_actions = [], [], []
        while not terminal and len(batch_states) < batch_size:
            T += 1
            # Explore with epsilon probability:
            if random.random() < epsilon:
                action_index = random.randrange(action_size)
            else: 
                action_index = predict_rewards(screen)
            new_state, reward, terminal = step(env, action_index)
            # Clip reward in [-1; 1] range
            reward = np.clip(reward, -1, 1)
            # Apply future reward discounting
            if not terminal:
                reward += gamma * predict_target(new_state)
            # Accumulate gradients
            batch_rewards.append(reward)
            batch_actions.append(action_index)
            batch_states.append(state)
            state = new_state
        # Apply asynchronous gradient update to shared agent
        train(np.vstack(batch_states), batch_actions, batch_rewards)
        # Linearly anneal epsilon
        epsilon = update_epsilon(T, epsilon_anneal_steps, epsilon_min)
        # Logging and target network update.
        # thread_index == 0 means to do it only in 1st (chief) thread
        if thread_index == 0 and T % UPDATE_INTERVAL == 0: 
            update_target()
            # testing, logging, etc...
```


**Asynchronization** was implemented using standard python *threading* module. Despite python Global Interpreter Lock, all main work is done by TensorFlow, which parallelizes training process (benchmarks are shown in Table 1). So threads are created just like:

```python
# Run multiple learner threads asynchronously (in e.g. 8 threads):
import threading
thds = [threading.Thread(target=learner_thread) for i in range(8)]
for t in thds:
    t.start()
```

## Results

Here it is some results obtained after training on 80 millions of frames (shown on figure 5 and 6 - agent's progress during training and demonstation of it's play respectively):



{% include image.html
    img="/assets/posts/async-deeprl/si-momentum-lr3-reward.png"
    title="Agent's evaluation score during training"
    caption="Figure 5: Average episode reward during course of training on SpaceInvaders (light cyan curve - actual average episode reward, cyan curve - smoothed average episode reward)."
%}


<div class="video">
<iframe width="560" height="315" src="https://www.youtube.com/embed/rz2qWeMaqtw?autoplay=1&loop=1&rel=0&showinfo=0&color=white&iv_load_policy=3&playlist=rz2qWeMaqtw" frameborder="0" allowfullscreen></iframe>
Figure 6: Agent, trained over 80 millions of frames, plays Atari SpaceInvaders.
</div>


**Pretrained model** on SpaceInvaders can be downloaded from [here](https://drive.google.com/file/d/0By6rAKVSThTxRGYwRWlfM09MZTg/view). The model was trained asynchronously in 8 threads over 40 hours on GTX 980 Ti GPU, in total of 80 millions of frames.  
After model is downloaded and unpacked, you can evaluate it by running (by default result will be saved to *eval/SpaceInvaders-v0/*):

```bash
python run_dqn.py --logdir 'model/folder/path' --eval
```


<div style="text-align: right"><b>Table 1.</b> Current implementation of Asynchronous one-step Q-Learning algorithm benchmarks.</div>

|   **Device**                                        |   **Input shape**     |   **FPS**   |
|:----------------------------------------------------|:---------------------:|:-----------:|
| GPU **GTX 980 Ti**                                  | $$84\times84\times4$$ |   **530**   |
| CPU **Core i7-3770 @ 3.40GHz** (4 cores, 8 threads) | $$84\times84\times4$$ |   **300**   |

## Where can I learn more about Deep Reinforcement Learning?

I would suggest you to start from [Andrej Karpathy's post](http://karpathy.github.io/2016/05/31/rl/) - an awesome explanation of *stochastic Policy Gradients* applied to a pong game.  
To get more intuition about classic Deep Q-Network you may read through [this post](https://www.nervanasys.com/demystifying-deep-reinforcement-learning), and watch [10 David Silver's lectures about RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).  
And of course, [Richard Sutton's RL book](https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf). 

To gain some understanding in **deep learning**, I would recommend [Nielsen's online book](http://neuralnetworksanddeeplearning.com/). Later, work through [CS231n Stanford lectures](http://cs231n.github.io/) (unfortunately official video lectures were removed from youtube, but probably, somewhere, there might be an unofficial ones ;) ).

Finally, recorded A3C speech from NIPS workshop can be found [here](https://youtu.be/9sx1_u2qVhQ).

**Some awesome RL papers:**

1. A3C: [Asynchronous Methods for Deep Reinforcement Learning, Mnih et al., 2016](https://arxiv.org/abs/1602.01783).
2. UNREAL: [Reinforcement Learning with Unsupervised Auxiliary Tasks, Jaderberg et al., 2016](https://arxiv.org/pdf/1611.05397.pdf). 
3. DQN: [Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013](http://arxiv.org/pdf/1312.5602v1.pdf).
4. Deterministic Deep Policy Gradients: [Continuous control with deep reinforcement learning, Lillicrap, Hunt et al., 2016](http://arxiv.org/pdf/1509.02971v5.pdf).
5. Deterministic Policy Gradients: [Deterministic Policy Gradient Algorithms, Silver et al., 2014](http://jmlr.org/proceedings/papers/v32/silver14.pdf).

And that's it! Any feedback will be highly appreciated.  
**Thank you for reading, hope you enjoy it!**
