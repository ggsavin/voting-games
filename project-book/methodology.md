# Methodology

## MDPs

## POMDPs

## PPO

```{prf:algorithm} Proximal Policy Optimization
:label: ppo-alg

**Inputs** Given a Network $G=(V,E)$ with flow capacity $c$, a source node $s$, and a sink node $t$

**Output** Compute a flow $f$ from $s$ to $t$ of maximum value

1. $f(u, v) \leftarrow 0$ for all edges $(u,v)$
2. While there is a path $p$ from $s$ to $t$ in $G_{f}$ such that $c_{f}(u,v)>0$
	for all edges $(u,v) \in p$:

	1. Find $c_{f}(p)= \min \{c_{f}(u,v):(u,v)\in p\}$
	2. For each edge $(u,v) \in p$

		1. $f(u,v) \leftarrow f(u,v) + c_{f}(p)$ *(Send flow along the path)*
		2. $f(u,v) \leftarrow f(u,v) - c_{f}(p)$ *(The flow might be "returned" later)*
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