# Methodology

## MDPs

Markov Decision Process

## POMDPs

## PPO

PPO was chosen because it stil seems to be the most widely used on-policy algorithm.

INitializations and hyper-parameter choices were greatly influenced by the exhaustive work in {cite}`Andrychowicz2020-fs`

```{prf:algorithm} Proximal Policy Optimization
:label: ppo-alg

**Inputs** Initial policy parameters $theta_0$, clipping threshold $\epsilon$
1. for $k=0,1,2,...$ do
    1. Collect set of trajectories $\Tau$ on policy $\pi_k = \pi(\theta_k)$
    2. Estimate advatanges $A^{\pi_k}$ using GAE (Schulman 2016)
    3. Compute policy update
            $\theta_{k+1} = argmax \mathcal{L} $
        by taking $K$ steps of minibatch SGD (via Adam), where

        (Schulman 2017)
```


```{note}
The 37 implementation details of PPO[^37-details-ppo] is a great blog post detailing many more intricacies of PPO implementations
```

## Agent Implementation

Use pytorchviz or torchview to visualize the pytorch graph

### Utilities

(convert-obs)=
#### Convert observation to vector

(game-analysis-methodology)=
## Analysis


[^37-details-ppo]:https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/