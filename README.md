# voting-games

A custom simplified ultimate werewolf environment is available, along with notebooks and scripts trying to observe emergent behavior


### TODOs:
    - [ ] Cleanup Environment
    - [ ] Improve Rendering for human observations
    - [ ] Update the environment to have two rounds of voting, 
          allowing villagers a chance to be influenced by eachother
    - [ ] Experiment with Different RL Libraries & Algorithms
        - [ ] Stable-Baselines3
        - [ ] Ray/RLlib
        - [ ] Tianshou
        - [ ] CleanRL
    - [ ] Set up experimentation metrics
        - [ ] Weights and Biases
        - [ ] MLFlow
        - [ ] Neptune
        - [ ] AIM.io

### Directions to explore:

    - [ ] Update environment
        - [ ] How ? 
    - [ ] Re-train agents with 
        - [ ] different model
        - [ ] different rewards (what should these be?)
        - [ ] Environment now shows dead werewolves actual roles
    - [ ] Visualize games even better


https://www.reddit.com/r/reinforcementlearning/comments/o5o0b7/agent_taking_multiple_actions/
https://github.com/henrycharlesworth/multi_action_head_PPO

https://www.sciencedirect.com/science/article/pii/S037722172100850X


- change reward structures (verify that rewards and being given properly)
- look at model differences
- try to do combined action space of 3^n 
- get a VM and run it for a day + instead of the amount of time I have been (1-3 hours)
- sum critic value vs mean right now
- different architectures
- ** USE AN LSTM, or something to allow history in observation? **
- look at possibly keeping a counterfactual model of what the agent might be
- change accusation phases



____
Need to revamp environment
maybe keep player positions as roles, but change IDs around randomly?

- wolves need to vote the same way during the accusation and actual voting

---
https://npitsillos.github.io/blog/2021/recurrent-ppo/
https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO-aka-PPO-LSTM-on-environments-with-masked-velocity-SB3-Contrib---VmlldzoxOTI4NjE4
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
https://cnvrg.io/pytorch-lstm/



---
For LSTM even better :

https://npitsillos.github.io/blog/2021/recurrent-ppo/
https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/blob/main/buffer.py
https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt/blob/main/model.py


### Other stuff to do

Track KL Divergence

- [ ] Create tests for the wolf environments

Behavioral Agents
- [ ] Revenge Villager (votes for up to x people who voted for them)
- [ ] Punishing Villager (votes for people with the most dissaprovals)
- [ ] Forgiving Villager (puts another villager neutral if that villager made them neutral or positive after voting for them)


- Do we care about rounds being in the observation to users?


## Current Problems

Training PPO frequently leads to NaNs and this instability is something we have been exploring by trying to change hyperparameters of the model, the training and the environment. There are quite a few posts about this [here](https://github.com/hill-a/stable-baselines/issues/340), [here](https://stable-baselines3.readthedocs.io/en/master/guide/checking_nan.html) and more if you look up Nans RL PPO training
- Maybe Entropy Coefficient is way too high? Needs to be lowered significantly?
