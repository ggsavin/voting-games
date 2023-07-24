# Literature Review

Reward designing to try to shape behavior and outcomes is very hard, and can usally lead to unintended behavior that still optimizes *cite* (look at inverse RL paper for these citations, use the upside-down helicopter picture)

## Voting RL

{cite}`Dodevska2019ComputationalSC` computational social choice and challenges of voting in multi-agent systems:



{cite}`Le_Gleau2020-ye` use RL to both learn how to offer and vote in _The Ultimatum Game_ where partitions of limited resources are voted upon. They evaluate their agents learned performances against different voting behaviors. There is no temporal aspect to teh game


{cite}`Xu2019-vy` and {cite}`Matsunami2020-wt` have their agents learn to vote for a joint action that maximize total average rewards. {cite}`Xu2019-vy` explored voting-based coordination of IoT devices. They only exchange vote information for collaboration, with no need to exchange value functions, and conclude that distributed decision making does not slow down the process of obtaining global consensus, and voting-based learning is more efficient than letting agents behave individually. This is a fully cooperative setting. {cite}`Matsunami2020-wt` considered an enhanced reward function that implements a VCG mechanism to incentivize truthful behavior and stabalize learning. They found that this mechanism produced the highest grossing profit for their game, constrasted with local and global reward settings. Given that our reward function is essentially their local reward case, it would be interesting to see how a VCG mechanism would change the way our agents converge. 


Similar to MD VCG is the idea of counterfactuals by Forester *cite* transition to work on hidden role games.




In our problem space, we have competition between subsets of agents that are not actually known to one group. Agents also die, and cannot continue.


## Why did we choose PPO

A majority of the works we looked at used PPO {cite}`Matsunami2020-wt, Brandizzi2021RLupusCT`, , as it has nice learning properties and good overall success.

We implemented our own following some works in truncated BPTT and CLeanRL, however relying on a framework might have been a better choice.



## Game theoretic work done in mafia

## RL

AIWolf {cite}`toriumi2017ai` is a popular Werewolf game framework and yearly competition run out of Japan *cite the competition and the framework*, which pits agents against eachother in full NLP or protocol limited communication modes. Agents have been trained *cite*


RLupus {cite}`Brandizzi2021RLupusCT` explored explicit communication in Werewolf, by adding a signal channel to the action. A follow up {cite}`Lipinski2022EMP` showed that a single word vocabulary is a good enough heuristic. They also explored longer communication rounds, much like we allow multiple accusation rounds.


Our work takes the voting mechanism and replaces it with an approval variant, to see if there is any difference. 



## SGDs

Organize around different approaches taken on hidden role, elimination games.


## Questions we try to answer?

 Can we characterize the emerging agent behavior even in the absence of human notions of strategy, learning or equilibria?



