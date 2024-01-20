import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import Dict_Hyperparams as P
import Util as U


class SubNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_units, output_size, seed):
        super(SubNetwork, self).__init__()
        dims = (input_size,) + hidden_units        
        self.layers = nn.ModuleList([U.layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]
        self.output_layer = U.layer_init(nn.Linear(self.feature_dim, output_size), 1e-3)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.tanh(layer(x))
        x = self.output_layer(x)    
        return x    
            
class ActorAndCritic(nn.Module):
    
    def __init__(self, num_agents, state_size, action_size, seed, device):
        super(ActorAndCritic, self).__init__()
        self.device = device
        self.seed = random.seed(seed)
        self.actor = SubNetwork(state_size, (P.HIDDEN_LAYERS, P.HIDDEN_LAYERS), action_size, seed)
        self.critic = SubNetwork(state_size, (P.HIDDEN_LAYERS, P.HIDDEN_LAYERS), 1, seed)
        self.std = nn.Parameter(torch.zeros(action_size))
        #self.to(Config.DEVICE)
        
    def forward(self, obs, action=None):
        obs = U.tensor(obs, self.device)
        a = self.actor(obs)
        v = self.critic(obs)
        mean = F.tanh(a)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        return (v, dist)
        
class Rollout():
    
    def __init__(self, device):
        # Stored values
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.episode_not_dones = []
        self.states = []
        # Calculated values
        self.returns = [0.0] * P.ROLLOUT_LENGTH
        self.advantages = [0.0] * P.ROLLOUT_LENGTH
        self.device = device
        
    def save_prediction(self, actions, log_prob_actions, values):
        self.actions.append(actions)
        self.log_prob_actions.append(log_prob_actions)
        self.values.append(values)

    def save_consequences(self, rewards, episode_not_dones, states):
        self.rewards.append(rewards)
        self.episode_not_dones.append(episode_not_dones)
        self.states.append(states)
        
    def calculate_returns_and_advantages(self, final_reward, num_agents):
        self.rewards.append(None)
        self.episode_not_dones.append(None)
        self.calculate_future_returns(final_reward)
        self.estimate_advantages(num_agents)

    def calculate_future_returns(self, returns):
        for i in reversed(range(P.ROLLOUT_LENGTH)):
            returns = self.rewards[i] + P.DISCOUNT * self.episode_not_dones[i] * returns
            self.returns[i] = returns.detach() 

    def estimate_advantages(self, num_agents):
        advantages = U.tensor(np.zeros((num_agents, 1)), self.device)
        # Go backwards through rollout steps and calculate advantages for each state action pair
        # Use GAE for PPO. (Schulman, Moritz, Levine et al. 2016)
        for i in reversed(range(P.ROLLOUT_LENGTH)):
            td = self.rewards[i] + (P.DISCOUNT * self.episode_not_dones[i] * self.values[i + 1]) - self.values[i]
            advantages = advantages * P.GAE_LAMBDA * P.DISCOUNT * self.episode_not_dones[i] + td
            self.advantages[i] = advantages.detach()               

    def stack_tensor(self, some_list):
        return torch.cat(some_list[:P.ROLLOUT_LENGTH], dim=0)
            
    def get_sample_data(self):
        states = self.stack_tensor(self.states)
        actions = self.stack_tensor(self.actions) 
        log_prob_actions = self.stack_tensor(self.log_prob_actions)
        returns = self.stack_tensor(self.returns)
        # Normalize advantages
        advantages = self.stack_tensor(self.advantages)
        advantages = (advantages - advantages.mean()) / advantages.std()        
        return (states, actions, log_prob_actions, returns, advantages)
    
class MasterAgent():   
    
    def __init__(self, num_agents, state_size, action_size, seed, device):
        self.device = device
        self.network = ActorAndCritic(num_agents, state_size, action_size, seed, self.device)
        self.first_states = True
        self.total_steps = 0
        self.state_normalizer = U.MeanStdNormalizer()
        self.num_agents = num_agents
        
    def evaluate_actions_against_states(self, states, actions):
        value, action_distribution = self.network(states, actions)
        log_prob = self.get_log_prob(action_distribution, actions)
        return (log_prob, value)
    
    def get_log_prob(self, action_distribution, actions):
        return action_distribution.log_prob(actions).sum(-1).unsqueeze(-1)
    
    def get_prediction(self, states):
        if self.first_states:
            self.states = states
            self.first_states = False
        #self.latest_actions, self.latest_log_prob, self.latest_values = self.get_prediction_from_states(self.states)
        self.latest_values, action_distribution = self.network(self.states)
        self.latest_actions = action_distribution.sample()
        self.latest_log_prob = self.get_log_prob(action_distribution, self.latest_actions)
        return self.latest_actions
    
    def step(self, states, actions, rewards, next_states, dones):
        rewards = np.asarray(rewards)
        next_states = self.state_normalizer(next_states)
        self.rollout.save_prediction(self.latest_actions, self.latest_log_prob, self.latest_values)
        dones = np.asarray(dones).astype(int)
        rewards = U.tensor(rewards, self.device).unsqueeze(-1)
        episode_not_dones = U.tensor(1 - dones, self.device).unsqueeze(-1)
        states = U.tensor(self.states, self.device)        
        self.rollout.save_consequences(rewards, episode_not_dones, states)

        self.states = next_states
                
    def start_rollout(self):
        self.rollout = Rollout(self.device)
            
    def process_rollout(self, states):
        self.save_final_results(states)
        self.rollout.calculate_returns_and_advantages(self.latest_values.detach(), self.num_agents)
        self.optimize()
        self.first_states = True
        
    def save_final_results(self, states):    
        self.get_prediction(states)
        self.rollout.save_prediction(self.latest_actions, self.latest_log_prob, self.latest_values)
   
    def save_weights(self):
        print("======== Saving weights ==========")
        torch.save(self.network.state_dict(), "trained_weights.pth")

    def optimize(self):
        # Now use tensors for 's', 'a', 'log_pi_a', 'ret', 'adv' for training
        # states, actions, log prob actions, returns, advantages (1 row / timestep, 1 column per worker)

        states, actions, log_probs_old, returns, advantages = self.rollout.get_sample_data()
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        optimizer = torch.optim.Adam(self.network.parameters(), 3e-4, eps=1e-5)
        for i in range(P.OPTIMIZATION_EPOCHS):
            number_timesteps = states.size(0)
            timesteps_to_sample = U.random_sample(np.arange(number_timesteps), P.MINI_BATCH_SIZE) 
            for timestep in timesteps_to_sample:
                t = U.tensor(timestep, self.device).long()
                # Get data for all workers from sampled timestep 
                sampled_states = states[t]
                sampled_actions = actions[t]
                sampled_log_probs_old = log_probs_old[t]
                sampled_returns = returns[t]
                sampled_advantages = advantages[t]
                self.optimize_with_sampled_worker_data(optimizer, sampled_states,
                                                                  sampled_actions,
                                                                  sampled_log_probs_old,
                                                                  sampled_returns,
                                                                  sampled_advantages)
        steps = P.ROLLOUT_LENGTH * self.num_agents
        # Total steps used to train network
        self.total_steps += steps
        
    def optimize_with_sampled_worker_data(self, optimizer, sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages):
        # Get log_prob(actions) and value given states
        # Pass in states for all workers x batch_size.
        log_prob_action, value = self.evaluate_actions_against_states(sampled_states, sampled_actions)
        
        policy_loss = self.get_policy_loss(log_prob_action, sampled_log_probs_old, sampled_advantages)
        value_loss = self.get_value_loss(value, sampled_returns)
        
        # Do the actual optimization
        optimizer.zero_grad()
        # Overall loss function for training both networks at once. Get gradients on weights.
        (policy_loss + value_loss).backward()
        # Clip weight gradients 
        nn.utils.clip_grad_norm_(self.network.parameters(), P.GRADIENT_CLIP) 
        # Run actual optimization
        optimizer.step()
        
    def get_policy_loss(self, log_prob_action, sampled_log_probs_old, sampled_advantages):
        # This is the core of PPO
        # ratio = new prob / old prob for all workers
        ratio = (log_prob_action - sampled_log_probs_old).exp() 
        # Clip loss on the upside
        clamped_ratio = ratio.clamp(1.0 - P.PPO_RATIO_CLIP, 1.0 + P.PPO_RATIO_CLIP)
        obj = ratio * sampled_advantages
        obj_clipped = clamped_ratio * sampled_advantages
        policy_loss = -torch.min(obj, obj_clipped).mean() 
        return policy_loss
    
    def get_value_loss(self, value, sampled_returns):
        # Mean squared error
        value_loss = 0.5 * (sampled_returns - value).pow(2).mean()
        return value_loss
