import torch
import numpy as np
from notebooks.learning_agents.models import ActorCriticAgent
from notebooks.learning_agents.buffer import ReplayBuffer
from notebooks.learning_agents.utils import convert_obs
import mlflow
import tqdm
import copy
import enum

class Roles(enum.IntEnum):
    VILLAGER = 0
    WEREWOLF = 1

class Phase(enum.IntEnum):
    ACCUSATION = 0
    VOTING = 1
    NIGHT = 2

class PPOTrainer:
    def __init__(self, env, config:dict, wolf_policy, run_id:str="run", device:torch.device=torch.device("cpu"), mlflow_uri:str=None, voting_type:str=None) -> None:
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
        self.wolf_policy = wolf_policy

        # we are not using schedules yet
        # self.lr_schedule = config["learning_rate_schedule"]
        # self.beta_schedule = config["beta_schedule"]
        # self.cr_schedule = config["clip_range_schedule"]

        # Initialize Environment
        self.env = env
        
        observations, _, _, _, _ = self.env.reset()
        # obs_size= env.convert_obs(observations['player_0']['observation']).shape[-1]
        obs_size = convert_obs(observations['player_0']['observation'], voting_type=voting_type).shape[-1]

        # Initialize Buffer
        self.buffer = ReplayBuffer(buffer_size=10, 
                                   gamma=self.config["config_training"]["training"]["gamma"], 
                                   gae_lambda=self.config["config_training"]["training"]["gae_lambda"])

        # Initialize Model & Optimizer
        # needs to be set appropriately for plurality, where approval states lines up
        self.agent = ActorCriticAgent({"rec_hidden_size": self.config["config_training"]["model"]["recurrent_hidden_size"], 
                                        "rec_layers": self.config["config_training"]["model"]["recurrent_layers"], 
                                        "joint_mlp_size": self.config["config_training"]["model"]["joint_mlp_size"],
                                        "split_mlp_size": self.config["config_training"]["model"]["split_mlp_size"], 
                                        "num_votes": self.config["config_training"]["model"]["num_votes"],
                                        "approval_states": self.config["config_training"]["model"]["approval_states"]},
                                        num_players=self.config["config_game"]["gameplay"]["num_agents"],
                                        obs_size=obs_size)
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), 
                                          lr=self.config["config_training"]["training"]["learning_rate"], 
                                          eps=self.config["config_training"]["training"]["adam_eps"])

        # setup mlflow run if we are using it

    def train(self, voting_type=None, save_threshold=50.0):
        if self.mlflow_uri:
            mlflow.set_tracking_uri(self.mlflow_uri)

        name = f'{self.run_id}'
        with mlflow.start_run(run_name=name):
            
            mlflow.log_params(self.config["config_training"]["training"])
            mlflow.log_params(self.config["config_training"]["model"])
            mlflow.log_params(self.config["config_game"]['gameplay'])

            loop = tqdm.tqdm(range(self.config["config_training"]["training"]["updates"]), position=0)

            # if the average wins when we do periodic checks of the models scoring is above the save threshold, we save or overwrite the model
            # model_save_threshold = 50.0

            for tid, _ in enumerate(loop):

                if tid % 10 == 0:
                    # print(f'Playing games with our trained agent after {epid} epochs')
                    loop.set_description("Playing games and averaging score")
                    wins = []
                    score_gathering = tqdm.tqdm(range(10), position=1, leave=False)
                    for _ in score_gathering:
                        wins.append(play_recurrent_game(self.env, 
                                                        self.wolf_policy, 
                                                        self.agent, 
                                                        num_times=100,
                                                        hidden_state_size=self.config["config_training"]["model"]["recurrent_hidden_size"],
                                                        voting_type=voting_type))
                        score_gathering.set_description(f'Avg wins with current policy : {np.mean(wins)}')

                    mlflow.log_metric("avg_wins/100", np.mean(wins))
                    if np.mean(wins) > save_threshold:
                        save_threshold = int(np.mean(wins))
                        torch.save(self.agent.state_dict(), f'{voting_type}_agent_{self.config["config_game"]["gameplay"]["num_agents"]}_score_{save_threshold}')

                loop.set_description("Filling buffer")
                # fill buffer
                buff = fill_recurrent_buffer_scaled_rewards(self.buffer, 
                                             self.env,
                                             self.config["config_training"],
                                             self.wolf_policy, 
                                             self.agent,
                                             voting_type=voting_type)

                # train info will hold our metrics
                train_info = []
                # TODO List how many items we are training on
                loop.set_description(f'Epoch Training on {self.buffer.games} games')
                for _ in range(self.config['config_training']["training"]['epochs']):
                    # run through batches and train network
                    for batch in buff.get_minibatch_generator(self.config['config_training']["training"]['batch_size']):
                        train_info.append(calc_minibatch_loss(self.agent, 
                                                              batch, 
                                                              clip_range=self.config['config_training']["training"]['clip_range'], 
                                                              beta=self.config['config_training']["training"]['beta'], 
                                                              v_loss_coef=self.config['config_training']["training"]['value_loss_coefficient'],
                                                              grad_norm=self.config['config_training']["training"]['max_grad_norm'],
                                                              optimizer=self.optimizer))

                train_stats = np.mean(train_info, axis=0)
                mlflow.log_metric("policy loss", train_stats[0])
                mlflow.log_metric("value loss", train_stats[1])
                mlflow.log_metric("total loss", train_stats[2])
                mlflow.log_metric("entropy loss", train_stats[3])


@torch.no_grad()
def play_recurrent_game(env, wolf_policy, villager_agent, num_times=10, hidden_state_size=None, voting_type=None):
    
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
    
        wolf_action = None
        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villagers actions
            for villager in villagers:
                #torch.tensor(env.convert_obs(observations['player_0']['observation']), dtype=torch.float)
                # torch_obs = torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float)
                torch_obs = convert_obs(observations[villager]['observation'], voting_type=voting_type)
                obs = torch.unsqueeze(torch_obs, 0)

                # TODO: Testing this, we may need a better way to pass in villagers
                recurrent_cell = magent_obs[villager]["hcxs"][-1]
                
                # ensure that the obs is of size (batch,seq,inputs)
                policies, _, recurrent_cell = villager_agent(obs, recurrent_cell)
                _, game_action = villager_agent.get_action_from_policies(policies, voting_type=voting_type)

                if voting_type == "plurality":
                    actions[villager] = game_action.item()
                elif voting_type == "approval":
                    actions[villager] = game_action.tolist()

                #store the next recurrent cells
                magent_obs[villager]["hcxs"].append(recurrent_cell)

            # wolf steps
            phase = env.world_state['phase']
            for wolf in wolves:
                wolf_action = wolf_policy(env, wolf, action=wolf_action)
                actions[wolf] = wolf_action
        
            next_observations, _, _, _, _ = env.step(actions)
            
            # clear the wolf action if needed
            if env.world_state['phase'] == Phase.NIGHT:
                wolf_action = None
            
            if env.world_state['phase'] == Phase.ACCUSATION and phase == Phase.NIGHT:
                wolf_action = None

        ## Fill bigger buffer, keeping in mind sequence
        winner = env.world_state['winners']
        if winner == Roles.VILLAGER:
            wins += 1

        # loop.set_description(f"Villagers won {wins} out of a total of {num_times} games")
    
    return wins

def calc_minibatch_loss(agent, samples: dict, clip_range: float, beta: float, v_loss_coef: float, grad_norm: float, optimizer):

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

    # need to repeat for amount of shape of policies (this way we know how many policy heads we need to watch out for)
    norm_advantage = norm_advantage.unsqueeze(1).repeat(1, agent.num_votes)

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
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=grad_norm)
    optimizer.step()

    
    return [policy_loss.cpu().data.numpy(),     # policy loss
            vf_loss.cpu().data.numpy(),         # value loss
            loss.cpu().data.numpy(),            # total loss
            entropy_loss.cpu().data.numpy()]    # entropy loss

@torch.no_grad()
def fill_recurrent_buffer_scaled_rewards(buffer, env, config:dict, wolf_policy, villager_agent, voting_type=None):

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
        
        wolf_action = None
        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villager steps
                # villagers actions
            for villager in villagers:
                #torch.tensor(env.convert_obs(observations['player_0']['observation']), dtype=torch.float)
                #torch_obs = torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float)
                torch_obs = convert_obs(observations[villager]['observation'], voting_type=voting_type)
                obs = torch.unsqueeze(torch_obs, 0)

                # TODO: Testing this, we may need a better way to pass in villagers
                recurrent_cell = magent_obs[villager]["hcxs"][-1]
                
                # ensure that the obs is of size (batch,seq,inputs)
                # we only have one policy, but to keep the same agent structure, we have "policies"
                policies, value, recurrent_cell = villager_agent(obs, recurrent_cell)
                policy_action, game_action = villager_agent.get_action_from_policies(policies, voting_type=voting_type)
                
                # only difference is the game_action here between this and approval
                if voting_type == "plurality":
                    actions[villager] = game_action.item()
                elif voting_type == "approval":
                    actions[villager] = game_action.tolist()

                # can store some stuff 
                magent_obs[villager]["obs"].append(obs)
                magent_obs[villager]["actions"].append(policy_action)

                # how do we get these
                magent_obs[villager]["logprobs"].append(torch.stack([policy.log_prob(action) for policy, action in zip(policies, policy_action)], dim=1).reshape(-1))
                magent_obs[villager]["values"].append(value)

                #store the next recurrent cells
                magent_obs[villager]["hcxs"].append(recurrent_cell)


            # wolf steps
            phase = env.world_state['phase']
            for wolf in wolves:
                wolf_action = wolf_policy(env, wolf, action=wolf_action)
                actions[wolf] = wolf_action

            # actions = actions | wolf_policy(env)
        
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for villager in villagers:
                # dividing rewards by 100
                magent_obs[villager]["rewards"].append(rewards[villager]/100.0)
                magent_obs[villager]["terms"].append(terminations[villager])

            # update wolf_action appropriately
            if env.world_state['phase'] == Phase.NIGHT:
                wolf_action = None
            
            if env.world_state['phase'] == Phase.ACCUSATION and phase == Phase.NIGHT:
                wolf_action = None

        ## Update the end_game rewards for villagers that died before the end
        max_game_rounds = max([len(magent_obs[villager]['rewards']) for villager in magent_obs.keys()])
        a_villager_who_made_it_to_end = [villager for villager in magent_obs.keys() if len(magent_obs[villager]['rewards']) == max_game_rounds][0]
        reward_at_max_round = magent_obs[a_villager_who_made_it_to_end]['rewards'][-1]
        for villager in villagers:
            magent_obs[villager]['rewards'][-1] += reward_at_max_round * (0.9**(max_game_rounds - len(magent_obs[villager]['rewards'])))

        ## Fill bigger buffer, keeping in mind sequence
        for agent in magent_obs:
            buffer.add_replay(magent_obs[agent])

        
    
    return buffer


@torch.no_grad()
def fill_recurrent_buffer(buffer, env, config:dict, wolf_policy, villager_agent, voting_type=None):

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
        
        wolf_action = None
        while env.agents:
            observations = copy.deepcopy(next_observations)
            actions = {}

            villagers = set(env.agents) & set(env.world_state["villagers"])
            wolves = set(env.agents) & set(env.world_state["werewolves"])

            # villager steps
                # villagers actions
            for villager in villagers:
                #torch.tensor(env.convert_obs(observations['player_0']['observation']), dtype=torch.float)
                #torch_obs = torch.tensor(env.convert_obs(observations[villager]['observation']), dtype=torch.float)
                torch_obs = convert_obs(observations[villager]['observation'], voting_type=voting_type)
                obs = torch.unsqueeze(torch_obs, 0)

                # TODO: Testing this, we may need a better way to pass in villagers
                recurrent_cell = magent_obs[villager]["hcxs"][-1]
                
                # ensure that the obs is of size (batch,seq,inputs)
                # we only have one policy, but to keep the same agent structure, we have "policies"
                policies, value, recurrent_cell = villager_agent(obs, recurrent_cell)
                policy_action, game_action = villager_agent.get_action_from_policies(policies, voting_type=voting_type)
                
                # only difference is the game_action here between this and approval
                if voting_type == "plurality":
                    actions[villager] = game_action.item()
                elif voting_type == "approval":
                    actions[villager] = game_action.tolist()

                # can store some stuff 
                magent_obs[villager]["obs"].append(obs)
                magent_obs[villager]["actions"].append(policy_action)

                # how do we get these
                magent_obs[villager]["logprobs"].append(torch.stack([policy.log_prob(action) for policy, action in zip(policies, policy_action)], dim=1).reshape(-1))
                magent_obs[villager]["values"].append(value)

                #store the next recurrent cells
                magent_obs[villager]["hcxs"].append(recurrent_cell)


            # wolf steps
            phase = env.world_state['phase']
            for wolf in wolves:
                wolf_action = wolf_policy(env, wolf, action=wolf_action)
                actions[wolf] = wolf_action

            # actions = actions | wolf_policy(env)
        
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for villager in villagers:
                magent_obs[villager]["rewards"].append(rewards[villager])
                magent_obs[villager]["terms"].append(terminations[villager])

            # update wolf_action appropriately
            if env.world_state['phase'] == Phase.NIGHT:
                wolf_action = None
            
            if env.world_state['phase'] == Phase.ACCUSATION and phase == Phase.NIGHT:
                wolf_action = None

        ## Fill bigger buffer, keeping in mind sequence
        for agent in magent_obs:
            buffer.add_replay(magent_obs[agent])
    
    return buffer