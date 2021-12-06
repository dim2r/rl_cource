import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gym
from pg_utils import get_device
from datetime import timedelta
import os
import collections


class Simulator:
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter=None,
                **env_args):
        ### EMPLOY MANY ENVS AT ONCE ####
        self.env = np.asarray([gym.make(env_name, **env_args) for i in range(n_trajectories)])

        for env in self.env:
            env._max_episode_steps = trajectory_len

        self.action_space = self.env[0].action_space
        self.policy = policy
        self.n_trajectories = n_trajectories
        self.trajectory_len = trajectory_len
        self.state_filter = state_filter
        self.device = get_device()

    def sample_trajectories(self):
        self.policy.eval()

        with torch.no_grad():
            memory = np.asarray([collections.defaultdict(list) for i in range(self.n_trajectories)])
            done = [False] * self.n_trajectories
            for trajectory in memory:
                trajectory['done'] = False

            for env, trajectory in zip(self.env, memory):
                state = torch.tensor(env.reset()).float()

                if self.state_filter:
                    state = self.state_filter(state)

                trajectory['states'].append(state)

            while not np.all(done):
                continue_mask = [i for i, trajectory in enumerate(memory) if not trajectory['done']]
                trajs_to_update = [trajectory for trajectory in memory if not trajectory['done']]
                continuing_envs = [env for i, env in enumerate(self.env) if i in continue_mask]

                policy_input = torch.stack([torch.tensor(trajectory['states'][-1]).to(self.device)
                                            for trajectory in trajs_to_update])

                action_dists = self.policy(policy_input)
                actions = action_dists.sample()
                actions = actions.cpu()

                for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                    state, reward, done, info = env.step(action.numpy())

                    state = torch.tensor(state).float()
                    reward = torch.tensor(reward, dtype=torch.float)

                    if self.state_filter:
                        state = self.state_filter(state)

                    trajectory['actions'].append(action)
                    trajectory['rewards'].append(reward)
                    trajectory['done'] = done

                    if not done:
                        trajectory['states'].append(state)

                done = [trajectory['done'] for trajectory in memory]

        return memory

# config = yaml.load(open('config.yaml', 'r'))
config={}

# TO DO: Model name issue

class TRPO:
    '''
    Optimizes the given policy using Trust Region Policy Optization (Schulman 2015)
    with Generalized Advantage Estimation (Schulman 2016).

    Attributes
    ----------
    policy : torch.nn.Sequential
        the policy to be optimized

    value_fun : torch.nn.Sequential
        the value function to be optimized and used when calculating the advantages

    simulator : Simulator
        the simulator to be used when generating training experiences

    max_kl_div : float
        the maximum kl divergence of the policy before and after each step

    max_value_step : float
        the learning rate for the value function

    vf_iters : int
        the number of times to optimize the value function over each set of
        training experiences

    vf_l2_reg_coef : float
        the regularization term when calculating the L2 loss of the value function

    discount : float
        the coefficient to use when discounting the rewards

    lam : float
        the bias reduction parameter to use when calculating advantages using GAE

    cg_damping : float
        the multiple of the identity matrix to add to the Hessian when calculating
        Hessian-vector products

    cg_max_iters : int
        the maximum number of iterations to use when solving for the optimal
        search direction using the conjugate gradient method

    line_search_coef : float
        the proportion by which to reduce the step length on each iteration of
        the line search

    line_search_max_iters : int
        the maximum number of line search iterations before returning 0.0 as the
        step length

    line_search_accept_ratio : float
        the minimum proportion of error to accept from linear extrapolation when
        doing the line search

    mse_loss : torch.nn.MSELoss
        a MSELoss object used to calculating the value function loss

    value_optimizer : torch.optim.LBFGS
        a LBFGS object used to optimize the value function

    model_name : str
        an identifier for the model to be used when generating filepath names

    continue_from_file : bool
        whether to continue training from a previous saved session

    save_every : int
        the number of training iterations to go between saving the training session

    episode_num : int
        the number of episodes already completed

    elapsed_time : datetime.timedelta
        the elapsed training time so far

    device : torch.device
        device to be used for pytorch tensor operations

    mean_rewards : list
        a list of the mean rewards obtained by the agent for each episode so far

    Methods
    -------
    train(n_episodes)
        train the policy and value function for the n_episodes episodes

    unroll_samples(samples)
        unroll the samples generated by the simulator and return a flattend
        version of all states, actions, rewards, and estimated Q-values

    get_advantages(samples)
        return the GAE advantages and a version of the unrolled states with
        a time variable concatenated to each state

    update_value_fun(states, q_vals)
        calculate one update step and apply it to the value function

    update_policy(states, actions, advantages)
        calculate one update step using TRPO and apply it to the policy

    surrogate_loss(log_action_probs, imp_sample_probs, advantages)
        calculate the loss for the policy on a batch of experiences

    get_max_step_len(search_dir, Hvp_fun, max_step, retain_graph=False)
        calculate the coefficient for search_dir s.t. the change in the function
        approximator of interest will be equal to max_step

    save_session()
        save the current training session

    load_session()
        load a previously saved training session

    print_update()
        print an update message that displays statistics about the most recent
        training iteration
    '''
    def __init__(self, policy, value_fun, simulator, max_kl_div=0.01, max_value_step=0.01,
                vf_iters=1, vf_l2_reg_coef=1e-3, discount=0.995, lam=0.98, cg_damping=1e-3,
                cg_max_iters=10, line_search_coef=0.9, line_search_max_iter=10,
                line_search_accept_ratio=0.1, model_name=None, continue_from_file=False,
                save_every=1):
        '''
        Parameters
        ----------

        policy : torch.nn.Sequential
            the policy to be optimized

        value_fun : torch.nn.Sequential
            the value function to be optimized and used when calculating the advantages

        simulator : Simulator
            the simulator to be used when generating training experiences

        max_kl_div : float
            the maximum kl divergence of the policy before and after each step
            (default is 0.01)

        max_value_step : float
            the learning rate for the value function (default is 0.01)

        vf_iters : int
            the number of times to optimize the value function over each set of
            training experiences (default is 1)

        vf_l2_reg_coef : float
            the regularization term when calculating the L2 loss of the value function
            (default is 0.001)

        discount : float
            the coefficient to use when discounting the rewards (discount is 0.995)

        lam : float
            the bias reduction parameter to use when calculating advantages using GAE
            (default is 0.98)

        cg_damping : float
            the multiple of the identity matrix to add to the Hessian when calculating
            Hessian-vector products (default is 0.001)

        cg_max_iters : int
            the maximum number of iterations to use when solving for the optimal
            search direction using the conjugate gradient method (default is 10)

        line_search_coef : float
            the proportion by which to reduce the step length on each iteration of
            the line search (default is 0.9)

        line_search_max_iters : int
            the maximum number of line search iterations before returning 0.0 as the
            step length (default is 10)

        line_search_accept_ratio : float
            the minimum proportion of error to accept from linear extrapolation when
            doing the line search (default is 0.1)

        model_name : str
            an identifier for the model to be used when generating filepath names
            (default is None)

        continue_from_file : bool
            whether to continue training from a previous saved session (default is False)

        save_every : int
            the number of training iterations to go between saving the training session
            (default is 1)
        '''

        self.policy = policy
        self.value_fun = value_fun
        self.simulator = simulator
        self.max_kl_div = max_kl_div
        self.max_value_step = max_value_step
        self.vf_iters = vf_iters
        self.vf_l2_reg_coef = vf_l2_reg_coef
        self.discount = discount
        self.lam = lam
        self.cg_damping = cg_damping
        self.cg_max_iters = cg_max_iters
        self.line_search_coef = line_search_coef
        self.line_search_max_iter = line_search_max_iter
        self.line_search_accept_ratio = line_search_accept_ratio
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.value_optimizer = torch.optim.LBFGS(self.value_fun.parameters(), lr=max_value_step, max_iter=25)
        self.model_name = model_name
        self.continue_from_file = continue_from_file
        self.save_every = save_every
        self.episode_num = 0
        self.elapsed_time = timedelta(0)
        self.device = get_device()
        self.mean_rewards = []

        if not model_name and continue_from_file:
            raise Exception('Argument continue_from_file to __init__ method of ' \
                            'TRPO case was set to True but model_name was not ' \
                            'specified.')

        if not model_name and save_every:
            raise Exception('Argument save_every to __init__ method of TRPO ' \
                            'was set to a value greater than 0 but model_name ' \
                            'was not specified.')

        if continue_from_file:
            self.load_session()



    def unroll_samples(self, samples):
        q_vals = []

        for trajectory in samples:
            rewards = torch.tensor(trajectory['rewards'])
            reverse = torch.arange(rewards.size(0) - 1, -1, -1)
            discount_pows = torch.pow(self.discount, torch.arange(0, rewards.size(0)).float())
            discounted_rewards = rewards * discount_pows
            disc_reward_sums = torch.cumsum(discounted_rewards[reverse], dim=-1)[reverse]
            trajectory_q_vals = disc_reward_sums / discount_pows
            q_vals.append(trajectory_q_vals)

        states = torch.cat([torch.stack(trajectory['states']) for trajectory in samples])
        actions = torch.cat([torch.stack(trajectory['actions']) for trajectory in samples])
        rewards = torch.cat([torch.stack(trajectory['rewards']) for trajectory in samples])
        q_vals = torch.cat(q_vals)

        return states, actions, rewards, q_vals

 



    def get_max_step_len(self, search_dir, Hvp_fun, max_step, retain_graph=False):
        num = 2 * max_step
        denom = torch.matmul(search_dir, Hvp_fun(search_dir, retain_graph))
        max_step_len = torch.sqrt(num / denom)

        return max_step_len

    def save_session(self, config_save_dir):
        if not os.path.exists(config_save_dir):
            os.mkdir(config_save_dir)

        save_path = os.path.join(config_save_dir, self.model_name + '.pt')

        ckpt = {'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value_fun.state_dict(),
                'mean_rewards': self.mean_rewards,
                'episode_num': self.episode_num,
                'elapsed_time': self.elapsed_time}

        if self.simulator.state_filter:
            ckpt['state_filter'] = self.simulator.state_filter

        torch.save(ckpt, save_path)

    def load_session(self):
        load_path = os.path.join(config_save_dir, self.model_name + '.pt')
        ckpt = torch.load(load_path)

        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.value_fun.load_state_dict(ckpt['value_state_dict'])
        self.mean_rewards = ckpt['mean_rewards']
        self.episode_num = ckpt['episode_num']
        self.elapsed_time = ckpt['elapsed_time']

        try:
            self.simulator.state_filter = ckpt['state_filter']
        except KeyError:
            pass

    def print_update(self):
        update_message = '[EPISODE]: {0}\t[AVG. REWARD]: {1:.4f}\t [ELAPSED TIME]: {2}'
        elapsed_time_str = ''.join(str(self.elapsed_time).split('.')[0])
        format_args = (self.episode_num, self.mean_rewards[-1], elapsed_time_str)
        print(update_message.format(*format_args))

        

