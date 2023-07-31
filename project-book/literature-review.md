# Literature Review

Reward designing to try to shape behavior and outcomes is very hard, and can usally lead to unintended behavior that still optimizes *cite* (look at inverse RL paper for these citations, use the upside-down helicopter picture)

## Voting RL

Earlier work by {cite}`Partalas2007-sz` used a voting process amongst a set of joint actions to choose the one that all agents follow in the predator-prey game. The reasoning is that each agent perceives only their part, so merging each agents partial knowledge may lead to better coordination and results. _Voting is used as the mechanism to combine the individual decisions of the agents and output a global decision (or strategy) that the agents as a team must follow_


{cite}`Airiau2017-ny` look to use RL in iterative voting to see if they observe convergence, and if the winner lines up with theoretic expectations. They use Condorcet efficiency and Borda Scores as their metrics, 


{cite}`Dodevska2019ComputationalSC` brings at the forefront the focuses on computational challenges of voting algorithms or different social issues of voting in the scope of multi-agent decentralized systems.



{cite}`Le_Gleau2020-ye` use RL to both learn how to offer and vote in _The Ultimatum Game_ where partitions of limited resources are voted upon. They evaluate their agents learned performances against different voting behaviors. There is no temporal aspect to teh game


{cite}`Xu2019-vy` and {cite}`Matsunami2020-wt` have their agents learn to vote for a joint action that maximize total average rewards. {cite}`Xu2019-vy` explored voting-based coordination of IoT devices. They only exchange vote information for collaboration, with no need to exchange value functions, and conclude that distributed decision making does not slow down the process of obtaining global consensus, and voting-based learning is more efficient than letting agents behave individually. This is a fully cooperative setting. {cite}`Matsunami2020-wt` considered an enhanced reward function that implements a VCG mechanism to incentivize truthful behavior and stabalize learning. They found that this mechanism produced the highest grossing profit for their game, constrasted with local and global reward settings. Given that our reward function is essentially their local reward case, it would be interesting to see how a VCG mechanism would change the way our agents converge. 


Similar to MD VCG is the idea of counterfactuals by Forester *cite* transition to work on hidden role games.




In our problem space, we have competition between subsets of agents that are not actually known to one group. Agents also die, and cannot continue.


## Why did we choose PPO

A majority of the works we looked at used PPO {cite}`Matsunami2020-wt, Brandizzi2021RLupusCT`, , as it has nice learning properties and good overall success.

We implemented our own following some works in truncated BPTT and CLeanRL, however relying on a framework might have been a better choice.



## Game theoretic work done in mafia



{cite}`braverman2008mafia` answer the question of how large should a sub-group be (the werewolves/mafia) to dominate the game. They explore games with and without a seer or detective that is fed information on the true roles of players.

## Mafia and Werewolf RL

AIWolf {cite}`toriumi2017ai` is a popular Werewolf game framework and yearly competition run out of Japan *cite the competition and the framework*, which pits agents against eachother in full NLP or protocol limited communication modes. Agents have been trained *cite*

{cite}`Eger2018-hx` explore commitment to an apparent goal through messaging in agents playing ultaimte werewolf.

RLupus {cite}`Brandizzi2021RLupusCT` explored explicit communication in Werewolf, by adding a signal channel to the action. A follow up {cite}`Lipinski2022EMP` showed that a single word vocabulary is a good enough heuristic. They also explored longer communication rounds, much like we allow multiple accusation rounds.


Other works {cite}`Ibraheem2020-mw` looked to classify roles in mafia based on their textual interactions and properties.

Our work takes the voting mechanism and replaces it with an approval variant, to see if there is any difference. 

{cite}`Serrino2019-ge` develop a neural network model based on counterfactual regret minimization to train agents to play The Resistance Avalon.

{cite}`Reinhardt2020-bs` developped a variant of Monte-Carlo Tree Search to play Secret Hitler, and showed that it can play as well as simpler rule based agents.


## Psychology

Initial Werewolf work, 
Werewolf was used {cite}`Conner2022-iu` to teach symbolic interaction and to help students overcome anxiety when confronted with sociological theory by using active learning approaches. {cite}`Ri2022-ih` looked at the different dynamics at play between the minority and majority teams in Mafia and tried to give it an entertainment score based on these dynamics.



## SDGs

Organize around different approaches taken on hidden role, elimination games.

A paired down clone of Among Us was explored in {cite}`Kopparapu2022-bm`. how individuals might learn to synthesize potentially unreliable information about others. Given the richness of the 2D environment, different strategic behaviors were observed such as camping and chasing and ability to vote properly.

Some work in SGDs explore algorithms for better play {cite}`Serrino2019-ge`
_J. Reinhardt, Finding friend and foe in multi-agent games, Application of deep reinforcement learning in werewolf game agents._, communication for better play _RLupus, wdad_


## Questions we try to answer?

 Can we characterize the emerging agent behavior even in the absence of human notions of strategy, learning or equilibria?



