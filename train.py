from common.logger import Logger
from common.storage import Storage, ConceptStorage
from common.model import NatureModel, ImpalaModel, ConceptMlpModel, ConceptNatureModel
from common.policy import CategoricalPolicy, ConceptPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import random
import torch
import wandb
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import AutoResetWrapper, RecordEpisodeStatistics, NormalizeReward, NormalizeObservation
from common.env.atari_wrappers import AttentionGuider

torch.autograd.set_detect_anomaly(True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'AG_Policy_Pong', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'ALE/Pong-v5', help='environment ID')
    parser.add_argument('--param_name',       type=str, default = 'concept-ppo', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'gpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--num_timesteps',    type=int, default = int(50000000), help = 'number of training timesteps')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(100), help='number of checkpoints to store')
    parser.add_argument('--patch_size',       type=int, default = int(10), help='patch_size that is the common divisor of height and weight of the input screen')

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    patch_size = args.patch_size

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/atari/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    device = torch.device('cuda')

    #################
    ## ENVIRONMENT ##
    #################
    def make_envs(n_env, env_id, concept_dict=None, search_area=None, patch_size=None):
        def make_env():
            env = gym.make(env_id)
            # env = NormalizeReward(env)
            env = AttentionGuider(env, concept_dict, search_area, patch_size)
            env = AutoResetWrapper(env)
            # env = NormObservation(env)
            return env
        return gym.vector.SyncVectorEnv([make_env for _ in range(n_env)])
    if env_name == 'ALE/Pong-v5':
        concept_dict = {
            "player": np.array([92, 186, 92]),
            "enemy": np.array([213, 130, 74]),
            "ball": np.array([236, 236, 236]),
        }
        search_area = [[34, 194],[]]
    elif env_name == 'ALE/SpaceInvaders-v5':
        concept_dict = {
            "enemies": np.array([134, 134, 29]),
            "player": np.array([50, 132, 50]),
            "shield": np.array([181, 83, 40]),
            "bullet": np.array([142, 142, 142]),
        }
        search_area = [[31, 195], []]
    print('INITIALIZAING ENVIRONMENTS...')
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    # We use Atari Envs this time
    env = make_envs(n_envs, env_name, concept_dict, search_area, patch_size)
    env = NormalizeObservation(env)
    env = RecordEpisodeStatistics(env)

    ############
    ## LOGGER ##
    ############
    # Tensorboard
    print('INITIALIZAING LOGGER...')
    logdir = 'atari/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    # # Wandb
    wandb.init(project='atari',
               name=exp_name,
               config={**hyperparameters, **vars(args)})

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape[1:]
    algo = hyperparameters.get('algo', 'ppo')
    architecture = hyperparameters.get('architecture', 'impala')
    output_dim = hyperparameters.get('output_dim', 32)
    in_channels = observation_shape[-1]
    # print("in_channels: ", in_channels)
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    elif architecture == 'mlp':
        model = ConceptMlpModel(in_channel=in_channels, patch_size=patch_size, output_dim=output_dim)
    elif architecture == 'concept-nature':
        model = ConceptNatureModel(in_channels=in_channels, patch_size=patch_size, output_dim=output_dim)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    action_size = action_space[0].n
    if algo == 'ppo':
        policy = CategoricalPolicy(model, recurrent, action_size)
    elif algo == 'conceptPPO':
        param_dict = {'num_heads': 2, 'n_concepts': len(concept_dict.keys())}
        policy = ConceptPolicy(model, recurrent, action_size, **param_dict)
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    num_pathes = (210//patch_size)*(160//patch_size)
    mask_shape = (num_pathes, 1)
    concept_shape = (num_pathes, len(concept_dict))
    if algo == 'ppo':
        storage = Storage((3, 210, 160), hidden_state_dim, n_steps, n_envs, device)
    elif algo == 'conceptPPO':
        storage = ConceptStorage(observation_shape, hidden_state_dim, n_steps, n_envs, device,mask_shape, concept_shape)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    elif algo == 'conceptPPO':
        from agents.ppo import ConceptPPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)
    agent.policy.cuda()
    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train()
