from numpy import ndarray
from cyberbattle._env import cyberbattle_env
import numpy as np
from typing import List, NamedTuple, Optional, Tuple, Union
import random

# deep learning packages
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions import Normal
from torch.optim import Adam

import cyberbattle.agents.baseline.agent_wrapper as w

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
# init
# policy params theta
# q-function params phi1, phi2
#empty replay buffer):

#Initializing all the values in all the layers to 0
# gain=1 means no additional scaling
def initialize_weights(l):
    if isinstance(l, nn.Linear):
        nn.init.xavier_uniform_(l.weight, gain=1)
        nn.init.constant_(l.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        #q1
        self.linear1_1 = nn.Linear(num_inputs+num_actions, hidden_dim)
        self.linear1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear1_3 = nn.Linear(hidden_dim, 1)
     
        #q2
        self.linear2_1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_3 = nn.Linear(hidden_dim, 1)

        self.apply(initialize_weights)

    def forward(self, x, a):
        # concating the state and action
        xa = torch.cat([x, a],1)
        x1 = F.relu(self.linear1_1(xa))
        x1 = F.relu(self.linear1_2(xa))
        x1 = self.linear1_3(xa)

        x2 = F.relu(self.linear2_1(xa))
        x2 = F.relu(self.linear2_2(xa))
        x2 = self.linear2_3(xa)
        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy,self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(initialize_weights)

    def forward(self, x):
        """"Forward pass through the network. 
        Given a state x, compute the mean and log standard deviation for the Gaussian distribution
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x) #output mean of gaussian distribution
        log_std = self.log_std_linear(x) #output log standard dev
        log_std = torch.clamp(log_std,min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    # Sample from the gaussian distribution
    def sample(self, x):
        """Sample an action from the Gaussian distribution using the reparameterization trick (used to allow diffierentiation for backpropagation)
        Returns the log probability of the sampled action
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        # accounts for the nonlinearity
        action = torch.tanh(x_t)
        #action bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std
    
        
class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)


class SAC:
    """Soft Actor Critic
     Policy Network - Actor - outputs a probability distribution over actions for a given state - learns to maximize expected rewards + entropy (encouraging exploration)
     Two Q-value Networks (Critics) - estimate the expected return for a given state-action pair, during updates uses the minimum of the two q-values
     Two Target Q value Networks - use soft updates to stabilize training
     alpha - entropy tuning term
       """
    def __init__(self,action_space, state_space, gamma, alpha, hidden_size, lr=1e-3, use_cuda=False, target_update_interval=10):
        self.gamma = gamma
        self.alpha = alpha # entropy tuning
        self.action_range = [min(action_space), max(action_space)]
        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if use_cuda else "cpu") 

        self.critic = QNetwork(state_space.shape[0], action_space.shape[0], hidden_size).to(device=self.device)
        self.target = QNetwork(state_space.shape[0], action_space.shape[0], hidden_size).to(device=self.device)
        self.q_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.hard_update(self.target, self.critic)
        self.policy = GaussianPolicy(state_space.shape[0], action_space.shape[0], hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)


    # θtarget =  τθ+(1−τ)θtarget
    def soft_update(target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def update(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        # # run one gradient descent step for q1 and q2
        # self.critic_optim.zero_grad()
        # loss_q, q_info = compute_loss_q(data)
        # loss_q.backward()
        # self.critic_optim.step()

        # pi, log_pi, mean, log_std = self.policy.sample(batch)
        
        #target q value computation
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.policy.sample(next_state_batch)
            q1_target_val, q2_target_val = self.target(next_state_batch, next_action)
            min_target_q = torch.min(q1_target_val, q2_target_val)
            next_q = reward_batch + (1 - mask_batch) * self.gamma * (min_target_q - self.alpha * next_log_prob)

        # critic update
        q1_critic, q2_critic = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        q1_loss = F.mse_loss(q1_critic, next_q) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2_critic, next_q) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        critic_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        #policy update
        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # Regularization Loss
        reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            self.soft_update(self.target, self.critic, tau=0.005)

        return q1_loss.item(), q2_loss.item(), policy_loss.item()
    
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
                (self.action_range[1] + self.action_range[0]) / 2.0

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        if value_path is None:
            value_path = "models/sac_value_{}_{}".format(env_name, suffix)
        print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.value.state_dict(), value_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path, value_path):
        print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
        if value_path is not None:
            self.value.load_state.dict(torch.load(value_path))

    # s, a - state and action spaces
    # gamma, reward
    # P - state transition dynamics, mu - initial state distributions 
    # aim of entropy regularized rl 0 learn a policy pi_theta: s x a -> [0,1]


import logging

class SafeSACLogger:
    def __init__(self):
        self.logger = logging.getLogger("SafeSAC")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

logger = SafeSACLogger()

class SafeSAC(SAC):
    def __init__(self, *args, gamma_safe=0.99, epsilon=0.2, safe_lr=1e-3, tau=0.005, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma_safe = gamma_safe
        self.epsilon = epsilon
        self.tau = tau

        self.qsafe = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.qsafe_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.qsafe_optim = Adam(self.qsafe.parameters(), lr=safe_lr)
        self.hard_update(self.qsafe_target, self.qsafe)

        self.safe_replay = ReplayMemory(capacity=100000, seed=0)

    def is_unsafe(self, state):
        return False

    def find_safe_action(self, state_tensor, num_candidates=10):
        for _ in range(num_candidates):
            a_sampled, _, _, _ = self.policy.sample(state_tensor)
            q_safe = self.qsafe(state_tensor, a_sampled)
            if q_safe.item() <= self.epsilon:
                return a_sampled.detach().cpu().numpy()[0]
        return None

    def rollout(self, env, T):
        s, _ = env.reset()
        trajectory = []
        for _ in range(T):
            state_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            a = self.find_safe_action(state_tensor)
            if a is None:
                a, _, _, _ = self.policy.sample(state_tensor)
                a = a.detach().cpu().numpy()[0]
            s_prime, r, done, _ = env.step(a)
            trajectory.append((s, a, r, s_prime, done))
            s = s_prime
            if done:
                break
        self.safe_replay.buffer.extend(trajectory)
        logger.info(f"Collected {len(trajectory)} safe rollout steps.")
        return trajectory

    def update_safety_critic(self, batch_size):
        if len(self.safe_replay) < batch_size:
            logger.warning("Not enough samples in D_safe to update safety critic.")
            return None
        state, action, reward, next_state, done = self.safe_replay.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        label = torch.FloatTensor([self.is_unsafe(s) for s in next_state]).unsqueeze(1).to(self.device)
        with torch.no_grad():
            q_next = self.qsafe_target(next_state, action)
            target = label + (1 - done) * self.gamma_safe * q_next
        q_pred = self.qsafe(state, action)
        loss = F.mse_loss(q_pred, target)
        self.qsafe_optim.zero_grad()
        loss.backward()
        self.qsafe_optim.step()
        self.soft_update(self.qsafe_target, self.qsafe, self.tau)
        logger.info(f"Updated safety critic. Loss: {loss.item():.4f}")
        return loss.item()

    def hard_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    def soft_update(self, target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)



class SafeSACAgent:
    def __init__(self, env):
        self.env = env
        self.agent = SafeSAC(env.observation_space.shape[0], env.action_space.n, env)

    def select_action(self, state):
        if isinstance(state, dict):
            state = state['vector'] if 'vector' in state else list(state.values())[0]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.agent.select_action(state_tensor)
        return action

    def update_parameters(self):
        self.agent.update()
        return

    @property
    def replay_buffer(self):
        return self.agent.replay_buffer

    def is_unsafe(self, state):
        return self.agent.is_unsafe(state)

    def find_safe_action(self, state):
        return self.agent.find_safe_action(state)

    def update_safety_critic(self):
        self.agent.update_safety_critic()


def finetune(agent, env, n_target, batch_size, lambda_coef=1e-3):
    """
    Finetune the agent using the SQRL fine-tuning algorithm (Algorithm 2).
    Implements the full J_target update from the paper.
    """
    D_offline = ReplayMemory(capacity=100000, seed=42)

    # Dual variables for constraints in SQRL (Eq. 4)
    alpha = torch.nn.Parameter(torch.tensor(1.0, device=agent.device))
    nu = torch.nn.Parameter(torch.tensor(1.0, device=agent.device))
    dual_optimizer = torch.optim.Adam([alpha, nu], lr=lambda_coef)
    failure = 0

    s, _ = env.reset()
    for step in range(n_target):
        # Rejection sampling for safe action selection: a ~ Γ(πθ)
        state_tensor = torch.FloatTensor(s).unsqueeze(0).to(agent.device)
        a = agent.find_safe_action(state_tensor)
        if a is None:
            a, _, _, _ = agent.policy.sample(state_tensor)
            a = a.detach().cpu().numpy()[0]

        s_prime, r, done, _ = env.step(a)
        D_offline.push(s, a, r, s_prime, done)

        if len(D_offline) >= batch_size:
            state, action, reward, next_state, done = D_offline.sample(batch_size)
            state = torch.FloatTensor(state).to(agent.device)
            action = torch.FloatTensor(action).to(agent.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(agent.device)
            next_state = torch.FloatTensor(next_state).to(agent.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(agent.device)

            # Compute expected objective: E[min Q - α logπ - ν Q_safe] (Eq. 4)
            pi_action, log_pi, _, _ = agent.policy.sample(state)
            q1_pi, q2_pi = agent.critic(state, pi_action)
            min_q_pi = torch.min(q1_pi, q2_pi)
            q_safe = agent.qsafe(state, pi_action)
            J_target = (min_q_pi - alpha * log_pi - nu * q_safe).mean()

            # Update policy parameters via gradient ascent on J_target
            agent.policy_optim.zero_grad()
            (-J_target).backward()
            agent.policy_optim.step()

            # Update dual variables (gradient descent on -J_target)
            dual_optimizer.zero_grad()
            (-J_target).backward()
            dual_optimizer.step()

        # If state is unsafe, reset; else, proceed
        if agent.is_unsafe(s_prime):
            failure += 1
            s = env.reset()[0] 
        else:
            s = s_prime

    logger.info("Finetuning complete.")
    return failure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.logits(x)

    def sample(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def flatten_action(node_id, action_type, n_action_types):
    return node_id * n_action_types + action_type

def unflatten_action(flat_action, n_action_types):
    return flat_action // n_action_types, flat_action % n_action_types

class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.2, lr=3e-4, target_entropy=None):
        self.gamma = gamma
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.target_entropy = target_entropy or -np.log(1.0 / action_dim)

        self.policy = DiscretePolicy(state_dim, action_dim)
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        self.target_q1 = QNetwork(state_dim, action_dim)
        self.target_q2 = QNetwork(state_dim, action_dim)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    def update(self, memory, batch_size):
        state, action, reward, next_state, done = memory.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        with torch.no_grad():
            next_logits = self.policy(next_state)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            target_q1 = self.target_q1(next_state)
            target_q2 = self.target_q2(next_state)
            min_q = torch.min(target_q1, target_q2)
            entropy_term = self.alpha * next_log_probs
            next_q = (next_probs * (min_q - entropy_term)).sum(dim=1)
            q_target = reward + (1 - done) * self.gamma * next_q

        current_q1 = self.q1(state).gather(1, action.unsqueeze(1)).squeeze(1)
        current_q2 = self.q2(state).gather(1, action.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(current_q1, q_target)
        q2_loss = F.mse_loss(current_q2, q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q1_vals = self.q1(state)
        q2_vals = self.q2(state)
        min_q_vals = torch.min(q1_vals, q2_vals)
        policy_loss = (probs * (self.alpha * log_probs - min_q_vals)).sum(dim=1).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

    def act(self, state, eval=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        if eval:
            return torch.argmax(probs, dim=-1).item()
        dist = Categorical(probs)
        return dist.sample().item()



