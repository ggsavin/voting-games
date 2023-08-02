# Methodology

## POMDPs

Partially Observable Markov Decision Processes (POMDPs[^POMDPs]) are a special case of a Markov Decision Process (MDP) where the agent does not have direct observability to the state, but rather gets their own, possibly unique, observation. Formally, it is a tuple consisting of:
- $S$, a set of states
- $A$, a set of actions
- $T$, a set of transition probabilities between states
- $R: S \cross A \rightarrow \mathbb{R}$ is the rewaard function
- $\omega$, a set of observations
- $O$, a set of observation probabilities
- $\gamma$ is a discount factor, bounded by $[0,1)$

## RL \& MARL

Reference pictures here : https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b

| Single Agent RL | Multi Agent RL | 
| ---- | --- |
| ![Single-Agent-RL](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Ews7HaMiSn2l8r70eeIszQ.png) | ![Multi-Agent-RL](https://miro.medium.com/v2/resize:fit:720/format:webp/1*1o1oeH3vpzsfJukLbFsekw.png) |


## PPO

Proxmial Policy Optimization (PPO) was chosen because it stil seems to be the most widely used on-policy algorithm.

Because we are implementing an LSTM, the action $a_t$ selected by the policy $\pi_{\theta}$ depends on both the observation $o_t$ and the hidden state $h_t$.

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
    \mathcal{L}^C(\theta) & = E [\sum_{t=0}^{\tau}[min(q_t(\theta)\hat{A}_t, clip(q_t(\theta), 1-\epsilon, 1+ \epsilon)A_t)]]
            $q_t(\theta) & = \frac{\pi_{\theta}(a_t | o_t, h_t)}{\pi_{\theta-old}(a_t | o_t, h_t)}
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

### Why did we choose PPO

Other works involving the werewolf game we looked at used PPO {cite}`Matsunami2020-wt, Brandizzi2021RLupusCT`, , as it has nice learning properties and good overall success.

We implemented our own following some works in truncated BPTT and CLeanRL, however relying on a framework might have been a better choice.



[^37-details-ppo]:https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
[^POMDPs]:https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process