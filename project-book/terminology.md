# Terminology

## POMDPs

Partially Observable Markov Decision Processes (POMDPs[^POMDPs]) are a special case of a Markov Decision Process (MDP) where the agent does not have direct observability to the state, but rather gets their own, possibly unique, observation. Formally, it is a tuple consisting of:
- $S$, a set of states
- $A$, a set of actions
- $T$, a set of transition probabilities between states
- $R: S \times A \rightarrow \mathbb{R}$ is the rewaard function
- $\Omega$, a set of observations
- $O$, a set of observation probabilities
- $\gamma$ is a discount factor, bounded by $[0,1)$

Given a current state $s \in S$, an agent will take an action $a \in A$ based on some observation $o \in O$ and will transition to a new state $s'$ with probability $T(s'|s,a)$ and receive a reward $r = R(s,a)$

RL and MARL enviornments usually satisfy the Markov property that future states depend only on current state and action pairs, and thus can be formulated as MDPs.

## RL \& MARL

[Reinforcement Learning](Single-RL) is a training method wherein an agent takes an action in an environment and is either rewarded or punished while transitioning to a new state. This is repeated until the agent reaches a terminal state. Ideally an agent will learn an optimal *policy* that maps observations to actions through many interactions in their environment. 

When it comes to [Multi-Agent Learning](Multi-Agent-RL), multiple agents are either leaarning in a decentralized manner, or being directed by a central policy. The dynamic nature of of Multi-Agent systems make learning much more challenging, although certain algorithms such as PPO and DQN still work.

Agents can learn using value-based, policy-based and model-based algorithms. A good taxonomy of different types of algorithms can be seen in the diagram below.

```{figure-md} RL-Taxonomy
<img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" alt="Taxonomy of RL algorithms">

Taxonomy of RL Algorithms[^rl-tax]
```

A good reference on RL is the timeless book *Reinforcement Learning: An Introduction*[^Sutton-Barto-Book] by Sutton & Barto.

```{figure-md} Single-RL
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*Ews7HaMiSn2l8r70eeIszQ.png" alt="single agent RL">

Single Agent Reinforcement Learning[^rl-pictures]
```
```{figure-md} Multi-Agent-RL
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*1o1oeH3vpzsfJukLbFsekw.png" alt="multi agent RL">

Multi-Agent Reinforcement Learning[^rl-pictures]
```

(ppo-alg-header)=
## PPO

Proxmial Policy Optimization (PPO) was chosen because it stil seems to be the most widely used on-policy algorithm and it was also used in a similar setting as our work {cite}`Brandizzi2021RLupusCT, Matsunami2020-wt`.

Because we are also implementing an LSTM, the action $a_t$ selected by the policy $\pi_{\theta}$ depends on both the observation $o_t$ and the hidden state $h_t$.

We implemented our own following some works using a truncated BPTT {cite}`pleines2023memory` and CLeanRL, however relying on a framework might have been a better choice.

```{prf:algorithm} Proximal Policy Optimization w/ Clipped Surrogate
:label: ppo-alg

**Inputs** Initial policy parameters $\theta_0$, clipping threshold $\epsilon$
1. for $k=0,1,2,...$ do
    1. Collect set of trajectories $\tau$ on policy $\pi_k = \pi(\theta_k)$
    2. Estimate advatanges $A^{\pi_k}$ using GAE {cite}`schulman2015high`
    3. Compute policy update {cite}`schulman2017proximal`

    $$
    \theta_{k+1} = argmax \mathcal{L}^{CVH} 
    $$

    by taking $K$ steps of minibatch SGD (via Adam), where

    $$
        \mathcal{L}^C(\theta) & = E [\sum_{t=0}^{\tau}[min(q_t(\theta)\hat{A}_t, clip(q_t(\theta), 1-\epsilon, 1+ \epsilon)A_t)]] \\
        q_t(\theta) & = \frac{\pi_{\theta}(a_t | o_t, h_t)}{\pi_{\theta-old}(a_t | o_t, h_t)}
    $$

    $$
        q_t(\theta) = \frac{\pi_{\theta}(a_t | o_t, h_t)}{\pi_{\theta-old}(a_t | o_t, h_t)}
    $$
    
    $$
    \mathcal{L}^V_t(\theta) = (V_{\theta}(o_t,h_t) - V^{target}_t)^2
    $$

    Coefficients $c_1, c_2$ weigh the value fucntion loss and the entropy bonus
    $$
    L^{CVH}_t(\theta) = E [\mathcal{L}^C(\theta) - c_1\cdot \mathcal{L}^V_t(\theta) + c_2 \mathcal{H}[\pi_{\theta}](o_t, h_t)]
    $$

```


```{note}
The 37 implementation details of PPO[^37-details-ppo] is a great blog post detailing many more intricacies of PPO implementations
```

## LSTMs

Long-short Term Memory (LSTM) networks are a type of deep neural network that is tailored for sequential data. They are an evolution of Recurrent Neural Networks (RNNs) that have been designed to deal with backpropagation issues and can handle much longer sequences {cite}`hochreiter1997long`.

These types of networks have loops in them, allowing information  to persist. In the below [figure](Unrolled-RNN), we can see what an unrolled RNN looks like, and in following [figure](LSTM-Internal-Cell), we see the LSTM,along with its inner workings and different gates that the cells are composed of.

For our PPO Implementation, we store the hidden state $h_n$ and cell state $c_n$ from our LSTM output for each state/action pair taken by an agent so we can use it again the next time we want to call our model or calculate losses.

```{note}
A great resource to understanding LSTMs and RNNs is [this famous blogpost](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.
```

```{figure-md} Unrolled-RNN
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" alt="unrolled RNN">

An unrolled RNN highlights it's sequential nature[^understand-lstm]
```

```{figure-md} LSTM-Internal-Cell
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" alt="lstm cell">

LSTM Cells and their internals[^understand-lstm]
```



[^37-details-ppo]:https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
[^POMDPs]:https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process
[^Sutton-Barto-Book]:http://incompleteideas.net/book/the-book-2nd.html
[^rl-pictures]:https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b
[^rl-tax]:https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html
[^understand-lstm]:https://colah.github.io/posts/2015-08-Understanding-LSTMs/