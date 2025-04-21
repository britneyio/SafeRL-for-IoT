'''Stable-baselines agent for CyberBattle Gym environment'''


# +
from typing import cast
import sys,os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from cyberbattle._env.cyberbattle_toyctf import CyberBattleToyCtf
from final_project.iot_env import CyberBattleIoT
import logging

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3 import SAC
from cyberbattle._env.flatten_wrapper import (
    FlattenObservationWrapper,
    FlattenActionWrapper,
)
from stable_baselines3 import DQN
import os
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.learner as learner
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
retrain = ["sac", "safesac", "dqn"]

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")


# +
class DictToFlatBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        flat_sample = self.flatten_observation(self.env.reset()[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_sample.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return self.flatten_observation(observation)

    def flatten_observation(self, obs_dict):
        # Flatten each value (array/scalar) in the dict and concatenate
        flat_obs = []
        for k, v in obs_dict.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                flat_obs.append(np.array([v], dtype=np.float32))
            elif isinstance(v, np.ndarray):
                flat_obs.append(v.flatten().astype(np.float32))
            else:
                raise ValueError(f"Unsupported observation type: {type(v)} for key: {k}")
        return np.concatenate(flat_obs)

class DiscreteToBoxActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)

        self.original_action_space = env.action_space
        self.nvec = self.original_action_space.nvec

        # Create continuous action space with shape = original MultiDiscrete
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.nvec),),
            dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert continuous Box action ∈ [-1, 1] to MultiDiscrete action.
        Maps each element separately based on its sub-action size.
        """
        scaled = ((action + 1.0) / 2.0) * self.nvec
        discrete_action = np.floor(scaled).astype(np.int32)

        # Clip to ensure we remain within bounds
        return np.clip(discrete_action, 0, self.nvec - 1)


# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit, Schedule
from typing import Any, Dict, Optional, Type, Union
import numpy as np

class SafetyCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(obs_dim) + int(action_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))

class SafeReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_flags = np.zeros((self.buffer_size,), dtype=np.bool_)

    def add(self, obs, next_obs, action, reward, done, info):
        idx = self.pos
        self.safety_flags[idx] = info.get("unsafe", False)
        super().add(obs, next_obs, action, reward, done, info)

    def sample(self, batch_size, env=None):
        data = super().sample(batch_size, env)
        data["safety_flags"] = torch.tensor(
            self.safety_flags[data["indices"]], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        return data

class SafeSAC(SAC):
    def __init__(self, policy, env, safety_threshold=0.5, **kwargs):
            super().__init__(policy, env, **kwargs)
            self.safety_threshold = safety_threshold
    
            # Wait until env is properly set
            obs_space = self.observation_space
            action_space = self.action_space
    
            if isinstance(obs_space, gym.spaces.Box):
                obs_dim = obs_space.shape[0]
            elif isinstance(obs_space, gym.spaces.Dict):
                obs_dim = sum([np.prod(v.shape) for v in obs_space.spaces.values()])
            else:
                raise NotImplementedError("Unsupported observation space type")
    
            if isinstance(action_space, gym.spaces.Box):
                action_dim = action_space.shape[0]
            elif isinstance(action_space, gym.spaces.Discrete):
                action_dim = 1  # or action_space.n if you're modeling logits
            else:
                raise NotImplementedError("Unsupported action space type")
            self.safety_critic = SafetyCritic(obs_dim, action_dim).to(self.device)
            self.safety_critic_target = SafetyCritic(obs_dim, action_dim).to(self.device)
            self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
            self.safety_critic_optimizer = torch.optim.Adam(self.safety_critic.parameters(), lr=self.learning_rate)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with torch.no_grad():
                next_action, _ = self.policy.sample(replay_data.next_observations)
                next_q = self.safety_critic_target(replay_data.next_observations, next_action)
                safety_target = replay_data.safety_flags
            current_q = self.safety_critic(replay_data.observations, replay_data.actions)
            safety_loss = F.binary_cross_entropy_with_logits(current_q, safety_target)
            self.safety_critic_optimizer.zero_grad()
            safety_loss.backward()
            self.safety_critic_optimizer.step()
            super().train(1, batch_size)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # Get action from policy
        actions, _states = self.policy.predict(observation, deterministic=deterministic)
    
        # Preprocess obs and action for safety critic
        #if isinstance(observation, dict):
        obs_tensor = self.policy.obs_to_tensor(observation)[0]
        if isinstance(obs_tensor, dict):
            # Flatten all tensors to 1D before concatenating
            obs_tensor = torch.cat([
                v.view(-1) if v.dim() > 1 else v for v in obs_tensor.values()
            ], dim=0).unsqueeze(0)
        # else:
        #     obs_tensor = torch.tensor(observation).float().to(self.device).unsqueeze(0)
    
        action_tensor = torch.tensor(actions).float().to(self.device).unsqueeze(0)
        #action_tensor = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(0)  # shape: [1, action_dim]

        # Evaluate safety critic
        safety_scores = self.safety_critic(obs_tensor, action_tensor)
        mask = (torch.sigmoid(safety_scores) > self.safety_threshold).squeeze()
    
        # If safe, return action, otherwise return default/fallback (e.g., random or zero)
        if mask:
            return actions, _states
        else:
            print("Unsafe action blocked. Returning safe default.")
            safe_action = np.zeros_like(actions)  # replace with a better fallback if needed
            return safe_action, _states
    def safe_predict(self, observation, deterministic=False, max_trials=10):
        """
        Predict a safe action using the policy and safety critic.
        If no safe action is found after max_trials, fallback to the last sampled action.
        """
        obs_tensor, _ = self.policy.obs_to_tensor(observation)
    
        # Flatten dict observation
        if isinstance(obs_tensor, dict):
            obs_tensor = torch.cat(
                [v.float().flatten(start_dim=1) if v.ndim > 1 else v.float().unsqueeze(0)
                 for v in obs_tensor.values()],
                dim=-1
            )
    
        # Ensure obs_tensor is 2D
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
    
        last_action = None
    
        for _ in range(max_trials):
            # Sample action
            action, _ = self.policy.predict(observation, deterministic=deterministic)
            last_action = action
    
            # Handle Discrete or MultiDiscrete action space
            action_tensor = torch.tensor(action, device=self.device)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)  # [1, action_dim] for MultiDiscrete
            elif action_tensor.ndim == 0:
                action_tensor = action_tensor.view(1, 1)     # [1, 1] for Discrete
    
            # Evaluate safety
            with torch.no_grad():
                score = torch.sigmoid(self.safety_critic(obs_tensor, action_tensor)).item()
    
            if score >= self.safety_threshold:
                return action, None
    
        print("[SafeSAC] WARNING: No safe action found. Returning fallback.")
        return last_action, None



# +
env = CyberBattleIoT(
    maximum_node_count=12,
    maximum_total_credentials=10,
    observation_padding=True,
    throws_on_invalid_actions=False,
)

flatten_action_env = DiscreteToBoxActionWrapper(FlattenActionWrapper(env))

flatten_obs_env = FlattenObservationWrapper(flatten_action_env, ignore_fields=[
    # DummySpace
    "_credential_cache",
    "_discovered_nodes",
    "_explored_network",
])

print(DictToFlatBoxWrapper(flatten_obs_env))

env_as_gym = cast(GymEnv, flatten_obs_env)

o, _ = env_as_gym.reset()
print(o)
# -


check_env(flatten_obs_env)


# +
model_safe = SafeSAC(
    "MultiInputPolicy",
    env_as_gym,
    verbose=1,buffer_size=100000)

model_safe.learn(total_timesteps=100)
model_safe.save("safe_sac_model")
# -

model_sac = SAC("MultiInputPolicy", env_as_gym, verbose=1, buffer_size=100000)
model_sac.learn(total_timesteps=100, log_interval=4)
model_sac.save("sac_pendulum")

# +
envs = cast(CyberBattleIoT, gym.make('CyberBattleIoT-v0').unwrapped)
ep = w.EnvironmentBounds.of_identifiers(maximum_node_count=30, maximum_total_credentials=50, identifiers=envs.identifiers)

# %%
# Evaluate the Deep Q-learning agent for each env using transfer learning
_l = dqla.DeepQLearnerPolicy(
    ep=ep,
    gamma=0.015,
    replay_memory_size=10000,
    target_update=5,
    batch_size=512,
    learning_rate=0.01,  # torch default learning rate is 1e-2
)
dqn_learning_run = learner.epsilon_greedy_search(
        cyberbattle_gym_env=env,
        environment_properties=ep,
        learner=_l,
        episode_count=10,
        iteration_count=1000,
        epsilon=1e-3,
        epsilon_exponential_decay=50000,
        epsilon_minimum=0.1,
        verbosity=Verbosity.Quiet,
        render=False,
        plot_episodes_length=False,
        title=f"DQL",
    )
_l = dqn_learning_run["learner"]
print(_l)

tiny = cast(CyberBattleEnv, gym.make(f"ActiveDirectory-v{ngyms}"))
current_o, _ = tiny.reset()
tiny.reset(seed=1)
wrapped_env = AgentWrapper(tiny, ActionTrackingStateAugmentation(ep, current_o))
# Use the trained agent to run the steps one by one
max_steps = 1000
# next action suggested by DQL agent
# h = []
for i in range(max_steps):
    # run the suggested action
    _, next_action, _ = _l.exploit(wrapped_env, current_o)
    # h.append((tiny.get_explored_network_node_properties_bitmap_as_numpy(current_o), next_action))
    if next_action is None:
        print("No more learned moves")
        break
    current_o, _, is_done, _, _ = wrapped_env.step(next_action)
    if is_done:
        print("Finished simulation")
        break
tiny.render()
# -

if "a2c" in retrain:
    model_a2c = A2C("MultiInputPolicy", env_as_gym).learn(10)
    model_a2c.save("a2c_trained_iot")


if "ppo" in retrain:
    model_ppo = PPO("MultiInputPolicy", env_as_gym).learn(1)
    model_ppo.save("ppo_trained_iot")


# +
obs , _= env_as_gym.reset() 
print(obs)

for i in range(1000):
    assert isinstance(obs, dict)
    action, _states = model_dqn.predict(obs)
    print("action", action)
    print("states", _states)
    obs, reward, done, truncated, info = dqn_env.step(action)
    print("obs", obs)
# -


flatten_obs_env.render()
flatten_obs_env.close()



