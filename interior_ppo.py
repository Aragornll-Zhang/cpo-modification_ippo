import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical
import json

class IPO:
    """
        This is the IPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        # assert(type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = 5  # TODO env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim,
                                  probs=True)  # 离散空间输出概率                                                   # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

        # IPO new
        self.barrier_t_inv = 0.3 # TODO: 后续可变
        self.barrier_t_upper = 1000
        self.ipo_penalty_MAXI = -49.0 # 1e-21 - 1e-22 , 2 倍 lower bound 1e-9, TODO 不再 line search， 而是 J_C > 0 直接截断，设置一个大惩罚 -> log(0) （log( -1e10 ) -> -23 ）

        self.constrain_hole_prob = 0.1


    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens , batch_constrain_dict = self.rollout()  # ALG STEP 3

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            constrain_hole = np.mean(batch_constrain_dict['hole'])  # MC 算撞洞概率
            win_rate = np.mean(batch_constrain_dict['find_optimal'])
            print('撞洞率: ', constrain_hole)
            print('win : ', win_rate )

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # performance function maximizes it.
                clip_loss = (-torch.min(surr1, surr2)).mean()
                # print( _ , clip_loss )

                # + 惩罚项 / interior-barrier
                # # V1. wrong 算 batch_cost

                # V2.
                interior_loss = None
                if 'hole_last_step' in batch_constrain_dict:
                    before_holes_StateAndAction = batch_constrain_dict['hole_last_step'] # list[ tuple(state , action) ]
                    wrong_action_probs = self.calculate_FallIntoHoles_prob(before_holes=before_holes_StateAndAction)
                    # select unsatisfiable action_prob
                    # 分段一波

                    penalty_probs = wrong_action_probs[wrong_action_probs >= self.constrain_hole_prob]
                    if penalty_probs.shape[0]>0:
                        interior_loss_penalty = torch.mean( self.barrier_t_inv / ( 1 - penalty_probs + 1e-2 ) )
                        print('penalty: ',interior_loss_penalty)
                        interior_loss = interior_loss_penalty
                    # else:
                    #     interior_loss = 0
                    # # good
                    reward_probs = wrong_action_probs[ (wrong_action_probs < self.constrain_hole_prob) & (wrong_action_probs > min(self.constrain_hole_prob/2 , 0.06) ) ]
                    if reward_probs.shape[0]:
                        interior_loss_reward = self.barrier_t_inv * torch.mean( -torch.log( self.constrain_hole_prob - reward_probs + 0.35 )  )
                        print('reward',interior_loss_reward)
                        if interior_loss is None:
                            interior_loss = interior_loss_reward
                        else:
                            interior_loss += interior_loss_reward

                if interior_loss is None:
                    interior_loss = 0 # penalty_hole # TODO

                actor_loss = clip_loss + interior_loss # -clip_loss + (-log(-u)/t)
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # save_results
            save_data_dict = {}
            save_data_dict['iteration'] = i_so_far
            save_data_dict['times_step'] = str( t_so_far)
            save_data_dict['optimal_win'] = float(win_rate)
            save_data_dict['hole'] = float(constrain_hole)

            avg_ep_lens = np.mean(self.logger['batch_lens'])
            avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
            avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
            # Round decimal places for more aesthetic logging messages
            avg_ep_lens = str(round(avg_ep_lens, 2))
            avg_ep_rews = str(round(avg_ep_rews, 2))
            avg_actor_loss = str(round(avg_actor_loss, 5))
            save_data_dict['avg_actor_loss'] = avg_actor_loss
            save_data_dict['avg_ep_rews'] = avg_ep_rews

            # Print a summary of our training so far
            self._log_summary()

            # write save_data_dict into jsonl
            # ...
            with open('./results_ippo.jsonl', 'a+', encoding='utf-8') as jsonl_file:
                json.dump( save_data_dict , jsonl_file, indent=4, ensure_ascii=False)
                jsonl_file.write('\n')

            # Save our model if it's time
            if i_so_far % self.save_freq == 0 or (win_rate > 0.95 and constrain_hole < self.constrain_hole_prob ):
                torch.save(self.actor.state_dict(), f'./model_results/ippo_actor_epoch_{i_so_far}.pth')
                torch.save(self.critic.state_dict(), f'./model_results/ippo_critic_{i_so_far}.pth')
                print('save...')
                with open('./ippo_save_torch_info.txt' , 'a+' ,encoding='utf-8') as f:
                    f.write( str(i_so_far)  + '  saved. \n' )
                if (win_rate > 0.95 and constrain_hole < self.constrain_hole_prob and i_so_far > 100):
                    break

    def calculate_FallIntoHoles_prob(self,before_holes):
        states_before = torch.tensor( [ item[0] for item in before_holes ] , dtype=torch.float )
        wrong_action_idx = torch.tensor( [ item[1] for item in before_holes ] , dtype=torch.int64 ).unsqueeze(1)
        probs = self.actor(states_before)
        return probs.gather(-1, wrong_action_idx).squeeze(1)






    def rollout(self):
        """
            Parameters:
                None
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
                batch_constrain_list - [ [number of constrain1 occured] , [number of constrain2 record] }
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        batch_constrain_dict = {} # record diff constrains...
        ep_rews = []

        t = 0

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode
            ep_constrain_dict = {}
            # Reset the environment. Note that obs is short for observation.
            obs, _ = self.env.reset()
            done = False


            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode): # TODO env.max_timesteps
                # If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1  # Increment timesteps ran this batch so far

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done, _, ep_constrain_dict = self.env.step(action)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break


            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

            if ep_constrain_dict['hole'] == 1:
                if 'hole_last_step' in batch_constrain_dict:
                    batch_constrain_dict['hole_last_step'].append( (batch_obs[-1] , batch_acts[-1]) )
                else:
                    batch_constrain_dict['hole_last_step'] = [(batch_obs[-1] , batch_acts[-1])] # 记录上一个 (state , action) , 算概率，让该概率越小越好

            for constrain_key, v in ep_constrain_dict.items():
                if constrain_key in batch_constrain_dict:
                    batch_constrain_dict[constrain_key].append(v)
                else:
                    batch_constrain_dict[constrain_key] = [v]


        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens , batch_constrain_dict

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


    def get_action(self, obs):
        # 离散型， softmax
        probs = self.actor(obs)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()


    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        probs = self.actor(batch_obs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 1 # 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")


    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
