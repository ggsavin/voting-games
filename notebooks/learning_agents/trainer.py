import torch
import numpy as np
import Approval

class PPOTrainer:
    def __init__(self, env, config:dict, run_id:str="run", device:torch.device=torch.device("cpu"), mlflow_uri:str=None) -> None:
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
        self.env = env

        # we are not using schedules yet
        # self.lr_schedule = config["learning_rate_schedule"]
        # self.beta_schedule = config["beta_schedule"]
        # self.cr_schedule = config["clip_range_schedule"]

        # Initialize Environment
        env = pare(num_agents=self.config["config_game"]["gameplay"]["num_agents"], werewolves=self.config["config_game"]["gameplay"]["num_werewolves"])
        self.env = env
        
        observations, rewards, terminations, truncations, infos = env.reset()
        obs_size= env.convert_obs(observations['player_0']['observation']).shape[-1]

        # Initialize Buffer
        self.buffer = RolloutBuffer(buffer_size=10, gamma=0.99, gae_lambda=0.95)

        # Initialize Model & Optimizer
        self.agent = ApprovalRecurrentAgent({"rec_hidden_size": self.config["config_training"]["model"]["recurrent_hidden_size"], 
                                                "rec_layers": self.config["config_training"]["model"]["recurrent_layers"], 
                                                "hidden_mlp_size": self.config["config_training"]["model"]["mlp_size"],
                                                "approval_states": self.config["config_training"]["model"]["approval_states"]},
                                                num_players=self.config["config_game"]["gameplay"]["num_agents"],
                                                obs_size=obs_size)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.0001, eps=1e-5)

        # setup mlflow run if we are using it

    def train(self, idx: int):
        if self.mlflow_uri:
            mlflow.set_tracking_uri(self.mlflow_uri)

        name = f'{self.run_id}_{idx}'
        with mlflow.start_run(run_name=name):
            
            mlflow.log_params(self.config["config_training"]["training"])
            mlflow.log_params(self.config["config_training"]["model"])

            loop = tqdm(range(self.config["config_training"]["training"]["updates"]))

            for tid, _ in enumerate(loop):
                # train 100 times
                if tid % 2 == 0:
                    # print(f'Playing games with our trained agent after {epid} epochs')
                    loop.set_description("Playing games and averaging score")
                    wins = []
                    for _ in range(10):
                        wins.append(play_recurrent_game(self.env, 
                                                        random_wolf, 
                                                        self.agent, 
                                                        num_times=50,
                                                        hidden_state_size=self.config["config_training"]["model"]["recurrent_hidden_size"]))
                    
                    mlflow.log_metric("avg_wins/50", np.mean(wins))

                loop.set_description("Filling buffer")
                # fill buffer
                buff = fill_recurrent_buffer(self.buffer, 
                                             self.env,
                                             self.config["config_training"],
                                             random_wolf, 
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