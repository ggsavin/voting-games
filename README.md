# Voting Mechanisms in Werewolf

## What is this project and what is it trying to achieve?

This project explores how agents learn to play the game werewolf by solely using different voting mechanisms to select candidates for elimination aand execution. In games with hidden roles and motives, learning to communicate, and identification of traitors via communication has been at the forefront of research. Voting, and its abilities to force 

Questions we would like to answer?

- [ ] Can agents learn in an implicit communication style SGD?
- [ ] How does the number of accusation rounds impact their ability to learn?
- [ ] Are there differences in agents between voting mechanisms?


This repo is essentially a monolith with the following:
- The PettingZoo[^petting-zoo] environment created to explore voting mechanisms in werewolf
- Jupyter-Book[^jup-book] to present masters project findings for approval and plurality voting

### Custom Environment

#### Training

Do not forget to connect the devcontainer to the mlflow container so that tracking the training works. Training scripts actually require a connection to mlflow

Do not forget to change trainer.py `tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)` to True to disable the tqdm output

`nohup python /my/folder/abc.py >> abc.log &`

To view the mlflow instance running on our remote server, run this command with the appropriate information and then 
`ssh -L 9999:localhost:5000 A.B.C.D`

### Project Book
#### View book

The repo comes with a pre-compiled jupyter-book that can be viewed by opening `project-book/_build/html/index.html`

#### Build the book

Run `jupyter-book build --all project-book/` in the docker container at the project root level, making sure that `apt-get install graphviz` has been run in this docker container. This should probably be added in the dockerfile.


### TODOs

- Commitment as another indicator. On average how often do agents change their votes across phases?. In approval, does it go from like <-> neutral <-> dislike ? 
- Add ranked choice.
- Play with penalties
- Re-factor environment to remove voting, rewarding, etc and have it so that this can be customizzed and just added to the game in a modular fashion. 
- Allow the gaame environment to work with training infrastructure such as Tianshou[^tianshou]
- COMA, QMIX and MADDPG algorithms?
- Why is training so unstable? Why do Nans almost always happen at the LSTM layer?
- experiment with reward structure and  reward functions
- add testing to environment, in case we want to make env improvements that do not affect outcomes.
- consolidate static policy logic to fit and work within a class mimicking trained agents.


## Ranked choice?

How do we implement this? 

https://towardsdatascience.com/learning-to-rank-a-complete-guide-to-ranking-using-machine-learning-4c9688d370d4
https://en.wikipedia.org/wiki/Learning_to_rank



- Do we care about rounds being in the observation to users?


## Current Problems

Training PPO frequently leads to NaNs and this instability is something we have been exploring by trying to change hyperparameters of the model, the training and the environment. There are quite a few posts about this [here](https://github.com/hill-a/stable-baselines/issues/340), [here](https://stable-baselines3.readthedocs.io/en/master/guide/checking_nan.html) and more if you look up Nans RL PPO training
- Maybe Entropy Coefficient is way too high? Needs to be lowered significantly?


Another Comment found:
· Learned initial state: Large loss terms are caused in the first few time steps as a result of initializing the hidden state as zeroes thereby rendering the model to focus less on the actual sequence. Training the initial state as a variable can improve performance.

[This example detaches it](https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py), also uses detaching in the hidden states, and also uses a smooth_l1_loss to the loss, it also uses a fully connected layer before the LSTM


https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem


[^petting-zoo]: https://pettingzoo.farama.org/
[^jup-book]: https://jupyterbook.org/en/stable/intro.html
[^tianshou]:https://github.com/thu-ml/tianshou