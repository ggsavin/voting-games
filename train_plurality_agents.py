
import torch 
import numpy as np 
import copy
import mlflow
import sys
sys.path.append('../')
import tqdm
import random

from notebooks.learning_agents.buffer import ReplayBuffer
from notebooks.learning_agents.actor_critic_model import ActorCriticAgent

from voting_games.werewolf_env_v0 import plurality_env, plurality_Phase, plurality_Role

def random_coordinated_wolf(env, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    target = random.choice(list(villagers_remaining))
    return {wolf: int(target.split("_")[-1]) for wolf in wolves_remaining}

def random_wolfs(env):
    return {wolf: env.action_space(wolf).sample() for
            wolf in set(env.world_state["werewolves"]) & set(env.world_state['alive'])}

def revenge_coordinated_wolf(env, actions = None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    wolves_remaining = set(env.world_state["werewolves"]) & set(env.world_state['alive'])

    # who tried to vote out a wolf last time?
    
    target = random.choice(list(villagers_remaining))
    # pick 
    for wolf in wolves_remaining:
        actions[wolf] = [0] * len(env.possible_agents)
        actions[wolf][int(target.split("_")[-1])] = -1
        for curr_wolf in wolves_remaining:
            actions[wolf][int(curr_wolf.split("_")[-1])] = 1
    # for wolf in env.werewolves_remaining:

def random_single_target_villager(env, agent):
    targets = set(env.world_state["alive"]) - set([agent])
    return int(random.choice(list(targets)).split("_")[-1])

# random_coordinated_wolf(env)
def random_agent_action(env, agent, action=None):
   return env.action_space(agent).sample()

def random_coordinated_single_wolf(env, agent, action=None):
    villagers_remaining = set(env.world_state["villagers"]) & set(env.world_state['alive'])
    return action if action != None else int(random.choice(list(villagers_remaining)).split("_")[-1])


@torch.no_grad()
def fill_recurrent_buffer(buffer, env, config:dict, wolf_policy, villager_agent):

    buffer.reset(gamma=config["training"]["gamma"], gae_lambda=config["training"]["gae_lambda"])
    
    for _ in range(config["training"]["buffer_games_per_update"]):
        ## Play the game 
        next_observations, rewards, terminations, truncations, infos = env.reset()
        # init recurrent stuff for actor and critic to 0 as well
        magent_obs = {agent: {'obs': [], 
                              'rewards': [], 
                              'actions': [], 
                              'logprobs': [], 
                              'values': [], 
                              'terms': [],

                              # obs size, and 1,1,64 as we pass batch first
                              'hcxs': [(torch.zeros((1,1,config["model"]["recurrent_hidden_size"]), dtype=torch.float32), 
                                        torch.zeros((1,1,config["model"]["recurrent_hidden_size"]), dtype=torch.float32))]
                    } for agent in env.agents if not env.agent_roles[agent]}
        
        wolf_brain = {'day': 1, 'phase': 0, 'action': None}
        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villager steps
                # villagers actions
            for villager in villagers:
                #torch.tensor(env.convert_obs(observations['player_0']['observation']), dtype=torch.float)
                torch_obs = torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float)
                obs = torch.unsqueeze(torch_obs, 0)

                # TODO: Testing this, we may need a better way to pass in villagers
                recurrent_cell = magent_obs[villager]["hcxs"][-1]
                
                # ensure that the obs is of size (batch,seq,inputs)
                # we only have one policy, but to keep the same agent structure, we have "policies"
                policies, value, recurrent_cell = villager_agent(obs, recurrent_cell)
                policy_action, game_action = villager_agent.get_action_from_policies(policies, voting_type="plurality")
                
                actions[villager] = game_action.item()

                # can store some stuff 
                magent_obs[villager]["obs"].append(obs)
                magent_obs[villager]["actions"].append(policy_action)

                # how do we get these
                magent_obs[villager]["logprobs"].append(torch.stack([policy.log_prob(action) for policy, action in zip(policies, policy_action)], dim=1).reshape(-1))
                magent_obs[villager]["values"].append(value)

                #store the next recurrent cells
                magent_obs[villager]["hcxs"].append(recurrent_cell)


            # wolf steps
            day = observations[list(observations)[0]]['observation']['day']
            phase = observations[list(observations)[0]]['observation']['phase']

            if wolf_brain['day'] != day or wolf_brain['phase'] == plurality_Phase.NIGHT:
                wolf_brain = {'day': day, 'phase': phase, 'action': None}

            for wolf in wolves:
                action = wolf_policy(env, wolf, action=wolf_brain['action'])
                wolf_brain['action'] = action
                actions[wolf] = action

            # actions = actions | wolf_policy(env)
        
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for villager in villagers:
                magent_obs[villager]["rewards"].append(rewards[villager])
                magent_obs[villager]["terms"].append(terminations[villager])

        ## Fill bigger buffer, keeping in mind sequence
        for agent in magent_obs:
            buffer.add_replay(magent_obs[agent])
    
    return buffer

@torch.no_grad()
def play_recurrent_game(env, wolf_policy, villager_agent, num_times=10, hidden_state_size=None):
    
    wins = 0
    # loop = tqdm(range(num_times))
    for _ in range(num_times):
        ## Play the game 
        next_observations, rewards, terminations, truncations, infos = env.reset()
        # init recurrent stuff for actor and critic to 0 as well
        magent_obs = {agent: {'obs': [], 
                              # obs size, and 1,1,64 as we pass batch first
                              'hcxs': [(torch.zeros((1,1,hidden_state_size), dtype=torch.float32), torch.zeros((1,1,hidden_state_size), dtype=torch.float32))],
                    } for agent in env.agents if not env.agent_roles[agent]}
        

        wolf_brain = {'day': 1, 'phase': 0, 'action': None}

        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villagers actions
            for villager in villagers:
                #torch.tensor(env.convert_obs(observations['player_0']['observation']), dtype=torch.float)
                torch_obs = torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float)
                obs = torch.unsqueeze(torch_obs, 0)

                # TODO: Testing this, we may need a better way to pass in villagers
                recurrent_cell = magent_obs[villager]["hcxs"][-1]
                
                # ensure that the obs is of size (batch,seq,inputs)
                # we only have one policy, but to keep the same agent structure, we have "policies"
                policies, value, recurrent_cell = villager_agent(obs, recurrent_cell)
                _, game_action = villager_agent.get_action_from_policies(policies, voting_type="plurality")
                actions[villager] = game_action.item()

                #store the next recurrent cells
                magent_obs[villager]["hcxs"].append(recurrent_cell)

            # wolf steps
            day = observations[list(observations)[0]]['observation']['day']
            phase = observations[list(observations)[0]]['observation']['phase']
            
            if wolf_brain['day'] != day or wolf_brain['phase'] == plurality_Phase.NIGHT:
                wolf_brain = {'day': day, 'phase': phase, 'action': None}
            
            for wolf in wolves:
                action = wolf_policy(env, wolf, action=wolf_brain['action'])
                wolf_brain['action'] = action
                actions[wolf] = action

            # actions = actions | wolf_policy(env)
        
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

        ## Fill bigger buffer, keeping in mind sequence
        winner = env.world_state['winners']
        if winner == plurality_Role.VILLAGER:
            wins += 1

        # loop.set_description(f"Villagers won {wins} out of a total of {num_times} games")
    
    return wins

def calc_minibatch_loss(agent: ActorCriticAgent, samples: dict, clip_range: float, beta: float, v_loss_coef: float, optimizer):

    # TODO:Consider checking for NAans anywhere. we cant have these. also do this in the model itself
    # if torch.isnan(tensor).any(): print(f"{label} contains NaN values")
    policies, values, _ = agent(samples['observations'], (samples['hxs'].detach(), samples['cxs'].detach()))
    
    log_probs, entropies = [], []
    for i, policy_head in enumerate(policies):
        # append the log_probs for 1 -> n other agent opinions
        log_probs.append(policy_head.log_prob(samples['actions'][:,i]))
        entropies.append(policy_head.entropy())
    log_probs = torch.stack(log_probs, dim=1)
    entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)
    
    ratio = torch.exp(log_probs - samples['logprobs'])

    # normalize advantages
    norm_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
    norm_advantage = norm_advantage.unsqueeze(1).repeat(1, 1)

    # policy loss w/ surrogates
    surr1 = norm_advantage * ratio
    surr2 = norm_advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = torch.min(surr1, surr2)
    policy_loss = policy_loss.mean()

    # Value  function loss
    clipped_values = samples["values"] + (values - samples["values"]).clamp(min=-clip_range, max=clip_range)
    vf_loss = torch.max((values - samples['returns']) ** 2, (clipped_values - samples["returns"]) ** 2)
    vf_loss = vf_loss.mean()

    # Entropy Bonus
    entropy_loss = entropies.mean()

    # Complete loss
    loss = -(policy_loss - v_loss_coef * vf_loss + beta * entropy_loss)


    # TODO : do i reset the LR here? do I want to?

    
    # Compute gradients
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    optimizer.step()

    
    return [policy_loss.cpu().data.numpy(),     # policy loss
            vf_loss.cpu().data.numpy(),         # value loss
            loss.cpu().data.numpy(),            # total loss
            entropy_loss.cpu().data.numpy()]    # entropy loss

class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu"), mlflow_uri:str=None) -> None:
        """Initializes all needed training components.
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.device = device
        self.run_id = run_id
        self.mlflow_uri = mlflow_uri
        self.env = None

        # we are not using schedules yet
        # self.lr_schedule = config["learning_rate_schedule"]
        # self.beta_schedule = config["beta_schedule"]
        # self.cr_schedule = config["clip_range_schedule"]

        # Initialize Environment
        env = plurality_env(num_agents=10, 
                            werewolves=2, 
                            num_accusations=self.config["config_game"]['gameplay']["accusation_phases"], 
                            #rewards=self.config["config_game"]['rewards']
                            )
        self.env = env
        
        observations, rewards, terminations, truncations, infos = env.reset()
        obs_size= env.convert_obs(observations['player_0']['observation']).shape[-1]

        # Initialize Buffer
        self.buffer = ReplayBuffer(buffer_size=10, gamma=0.99, gae_lambda=0.95)

        # Initialize Model & Optimizer
        # needs to be set appropriately for plurality, where approval states lines up
        self.agent = ActorCriticAgent({"rec_hidden_size": self.config["config_training"]["model"]["recurrent_hidden_size"], 
                                        "rec_layers": self.config["config_training"]["model"]["recurrent_layers"], 
                                        "hidden_mlp_size": self.config["config_training"]["model"]["mlp_size"],
                                        "num_votes": self.config["config_training"]["model"]["num_votes"],
                                        "approval_states": self.config["config_training"]["model"]["approval_states"]},
                                        num_players=self.config["config_game"]["gameplay"]["num_agents"],
                                        obs_size=obs_size)
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), 
                                          lr=self.config["config_training"]["training"]["learning_rate"], 
                                          eps=self.config["config_training"]["training"]["adam_eps"])

        # setup mlflow run if we are using it

    def train(self):
        if self.mlflow_uri:
            mlflow.set_tracking_uri(self.mlflow_uri)

        name = f'{self.run_id}'
        with mlflow.start_run(run_name=name):
            
            mlflow.log_params(self.config["config_training"]["training"])
            mlflow.log_params(self.config["config_training"]["model"])
            mlflow.log_params(self.config["config_game"]['gameplay'])

            loop = tqdm.tqdm(range(self.config["config_training"]["training"]["updates"]), position=0)

            # if the average wins when we do periodic checks of the models scoring is above the save threshold, we save or overwrite the model
            model_save_threshold = 50.0

            for tid, _ in enumerate(loop):

                if tid % 10 == 0:
                    # print(f'Playing games with our trained agent after {epid} epochs')
                    loop.set_description("Playing games and averaging score")
                    wins = []
                    score_gathering = tqdm.tqdm(range(10), position=1, leave=False)
                    for _ in score_gathering:
                        wins.append(play_recurrent_game(self.env, 
                                                        random_coordinated_single_wolf, 
                                                        self.agent, 
                                                        num_times=100,
                                                        hidden_state_size=self.config["config_training"]["model"]["recurrent_hidden_size"]))
                        score_gathering.set_description(f'Avg wins with current policy : {np.mean(wins)}')

                    mlflow.log_metric("avg_wins/100", np.mean(wins))
                    if np.mean(wins) > model_save_threshold:
                        model_save_threshold = int(np.mean(wins))
                        torch.save(self.agent.state_dict(), f'plurality_agent_{self.config["config_game"]["gameplay"]["num_agents"]}_score_{model_save_threshold}')

                loop.set_description("Filling buffer")
                # fill buffer
                buff = fill_recurrent_buffer(self.buffer, 
                                             self.env,
                                             self.config["config_training"],
                                             random_coordinated_single_wolf, 
                                             self.agent)

                # train info will hold our metrics
                train_info = []
                loop.set_description("Epoch Training")
                for _ in range(self.config['config_training']["training"]['epochs']):
                    # run through batches and train network
                    for batch in buff.get_minibatch_generator(self.config['config_training']["training"]['batch_size']):
                        train_info.append(calc_minibatch_loss(self.agent, 
                                                              batch, 
                                                              clip_range=self.config['config_training']["training"]['clip_range'], 
                                                              beta=self.config['config_training']["training"]['beta'], 
                                                              v_loss_coef=self.config['config_training']["training"]['value_loss_coefficient'], 
                                                              optimizer=self.optimizer))

                train_stats = np.mean(train_info, axis=0)
                mlflow.log_metric("policy loss", train_stats[0])
                mlflow.log_metric("value loss", train_stats[1])
                mlflow.log_metric("total loss", train_stats[2])
                mlflow.log_metric("entropy loss", train_stats[3])
            # one more run

        # torch.save(self.agent, f"rnn_agent_{self.run_id}")

config_training = {
    "model": {
        "recurrent_layers": 1, # 1,2 (2)
        "recurrent_hidden_size": 128, # 64-128-256-512 (4)
        "mlp_size": 128, # 64-128-256-512 (4)
        "num_votes": 1,
        "approval_states": 10 # this is tied to the number of players
    },
    "training" : {
        "batch_size": 128, # 32-64-128-256-512-1024 (6)
        "epochs": 3, # 4,5,6,7,8,9,10 (7)
        "updates": 301, # 1000 (1)
        "buffer_games_per_update": 200, # 50-100-200 (3)
        "clip_range": 0.1, # 0.1,0.2,0.3 (3)
        "value_loss_coefficient": 0.1, # 0.1, 0.05, 0.01, 0.005, 0.001 (5)
        "max_grad_norm": 0.5, 
        "beta": 0.01, # entropy loss multiplier # 0.1, 0.05, 0.01, 0.005, 0.001
        "learning_rate": 0.0001, # 0.001, 0.0005, 0.0001, 0.00005, 0.00001
        "adam_eps": 1e-5, # 1e-8, 1e-7. 1e-6, 1e-5
        "gamma": 0.99, # 0.99
        "gae_lambda": 0.95, #0.95-0.99
    }
}


config_game = {
    "rewards": {
        "day": -1,
        "player_death": -1,
        "player_win": 10,
        "player_loss": -5,
        "self_vote": -1,
        "dead_villager": -1,
        "dead_vote": -1,
        "dead_wolf": 5,
        "no_viable_vote": -1,
        "no_sleep": -1,
    },
    "gameplay": {
        "accusation_phases": 2, # 2,3
        "num_agents": 10,
        "num_werewolves": 2,
    }
}

config = {
    "config_game": config_game,
    "config_training": config_training,
}

finished_one = False
for _ in range(50):
    try:
        trainer = PPOTrainer(config=config,run_id="Plurality", mlflow_uri="http://mlflow:5000")
        trainer.train()
        finished_one = True
    except ValueError as e:
        print("Probably a nan error")
        if ("nan" in str(e)):
            print("It was value errors")
        print("Trying again")
    finally:
        if finished_one == True:
            break