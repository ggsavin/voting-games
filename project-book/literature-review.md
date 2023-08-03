# Literature Review

While the idea of using different voting mechanisms in an SDG seems to be a novel permutation, research involving SDGs is not, and there is a lot of work done in this regard as well as specific research done on the mafia/werewolf SDG. 


## Game theorertic work on mafia/werewolf

Early iterative game theoretic research in mafia/werewolf {cite}`braverman2008mafia, yao2008theoretical, migdal2010mathematical` answered questions such as how large does the minority group (greater than or equal to the square root of total players) need to be to dominate a game aand what strategies should they follow (random policy). Although the game model was kept simple, and advanced behavior was not considered, it provided very good baseline insights for which further work could leverage. 

## AIWolf and other Werewolf environments

A more complete werewolf environment, AIWolf {cite}`toriumi2017ai`, was proposed to study communication in the Werewolf game. AIWolf splits the comeptition in two tracks, one where communication is based off of a protocol, and one where the full english vocabulary can be used. This communication occurs during communication phases, and need to be handled in tandem with choosing appropriate actions.

To train agents, Q-learning has been shown to work {cite}`hagiwara2019using, zhang2021designing, Wang2018-yn`, as it was used in both 5-player and 10+ AIWolf game settings. Some of these {cite}`Wang2018-yn, hagiwara2019using` even used multiple models for various in-game tasks. However, using AIWolf would have required changing a lot of internal server code, and would have been too much for a project, with minimal impact on the actual desired topic of voting mechanisms. 

RLupus {cite}`Brandizzi2021RLupusCT` is a simplified environment to focus on emergent communication in Werewolf, and we considered extending it, however too much would have had to be changed, and ultimately we went with a focused custom environment.

## Other SDGs

Hidden Agenda {cite}`Kopparapu2022-bm` is simplified version of Amongst us where agents use visual information to synthesize and learn from potentially unreliable information about other agents. Being a 2D environment, visualizing behavior such as camping and chasing were observed, as well as learning to spatially vote (using plurality as the underlying mechanism). Other research developped a neural network model based on counterfactual regret minimization to train agents to play The Resistance Avalon {cite}`Serrino2019-ge`. Variants of Monte-Carlo Tree Search were used to play Secret Hitler, and showed that it can play as well as simpler rule based agents {cite}`Reinhardt2020-bs`.


## Communication

The nature of duplicitous communication in Werewolf and more generally SDGs, leads to a wide range of research focusing solely on various aspects of it. Determining player roles or itentions based on their communication {cite}`Eger2018-hx, ibraheem2022putting, azaria2015agent, nishizaki2016behavior, 8916754, Hirata2016-bu, sun2021prediction` is a common avenue of research, with interesting variations such as tracking commitment via speech {cite}`Eger2018-hx` or using sentiment analysis {cite}`sun2021prediction`. Explicitly studying impacts of communication in Werewolf was also explored by allowing agents to enhance actions with variable length integer vectors {cite}`Brandizzi2021RLupusCT`, and a follow-up in that environment showed that a single word vocabulary was a good enough heuristic. They also explored longer communication rounds, much like we allow multiple accusation rounds. This matches the idea in {cite}`8916754` that a single utterance impacts the learning agents the most.
While communication is inherently critical to such games, we need to cut it out to focus solely on the elucidation the implicit vote signaling can provide to agents. 

## Voting in RL

Most of the voting in RL research can be split into two categories, the first being voting to select a joint action out of individual agent submissions {cite}`Partalas2007-sz, Xu2019-vy, Matsunami2020-wt, chen2022deep`. The usecases are varied, such as Joint Action Learning for IoT device coordination {cite}`Xu2019-vy` that compete with individual learners, to implenting VCG mechanisms {cite}`Matsunami2020-wt` to incentivize truthful behavior in games maximizing total profit. Given that our reward function is essentially their local reward case, it would be an interesting problem to see how a VCG mechanism would impact players in Hidden Role games. A similar idea to VCG mechanisms is the implementation of counterfactuals {cite}`foerster2018counterfactual` that have been used in hidden role games.


The other research avenue with a focus on voting is the use of RL and ML to intuite and classify voting outcomes {cite}`Airiau2017-ny, Le_Gleau2020-ye, burka2022voting`. The work in {cite}`Le_Gleau2020-ye` also designed novel Neural Nets to have agents learn to output discrete partitions of finite resources. While not directly applicable to the work we are doing, the novel outputting could help when implementing different voting mechanisms in the future. 



