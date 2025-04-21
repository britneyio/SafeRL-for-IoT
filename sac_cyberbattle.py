#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ipykernel')


# # SAC Agent for CyberBattleIoT Environment
#
# This notebook demonstrates how to use a Soft Actor-Critic (SAC) agent with the CyberBattleIoT environment.

# In[2]:


# Import necessary libraries
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


# In[3]:


# Import CyberBattleIoT environment and other modules
from final_project.iot_env import CyberBattleIoT
from cyberbattle.agents.baseline import agent_dql
from cyberbattle.agents.baseline.agent_dql import DeepQLearnerPolicy
from final_project.iot_agents import DiscreteSACAgent, SafeSACAgent, ReplayBuffer
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.learner as learner
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper, Verbosity
from typing import cast
from cyberbattle._env.cyberbattle_env import CyberBattleEnv
from cyberbattle._env.flatten_wrapper import (
    FlattenObservationWrapper,
    FlattenActionWrapper,
)
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.env_checker import check_env


# ## Helper Functions
#
# Define helper functions for flattening state observations and training the SAC agent.

# In[4]:


# Function to flatten dictionary state observations
def flatten_state(obs_dict):
    return np.concatenate([
        np.atleast_1d(value).astype(np.float32)
        for key, value in obs_dict.items()
    ])


# In[5]:


def sac_training(env, episodes=1000, replay_size=10000, batch_size=64, **kwargs):
    # Get the correct state and action dimensions from the environment
    state, _ = env.reset()
    if isinstance(state, dict):
        state_dim = len(flatten_state(state))
    else:
        state_dim = state.shape[0]

    # Calculate total number of possible actions for the CyberBattle environment using env bounds
    max_nodes = env.bounds.maximum_node_count
    n_local_vulns = len(env.local_vulnerabilities)
    n_remote_vulns = len(env.remote_vulnerabilities)

    # Calculate action space size for each action type
    connect_actions = max_nodes * max_nodes * len(env.ports)  # source * target * ports
    local_exploit_actions = max_nodes * n_local_vulns  # nodes * local vulnerabilities
    remote_exploit_actions = max_nodes * max_nodes * n_remote_vulns  # source * target * remote vulnerabilities
    total_actions = connect_actions + local_exploit_actions + remote_exploit_actions

    agent = DiscreteSACAgent(state_dim, total_actions, **kwargs)
    memory = ReplayBuffer(replay_size)

    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Convert state to tensor if it's a dictionary
            if isinstance(state, dict):
                state = flatten_state(state)

            # Pass the environment to the act method
            action = agent.act(state, env=env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Convert next_state to tensor if it's a dictionary
            if isinstance(next_state, dict):
                next_state = flatten_state(next_state)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                agent.update(memory, batch_size)

        rewards_history.append(total_reward)
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

    return agent, rewards_history


# ## Set Up the Environment
#
# Create and configure the CyberBattleIoT environment.

# In[6]:


# Create the environment
iot_env = CyberBattleIoT(
    maximum_node_count=12,
    maximum_total_credentials=10,
    observation_padding=True,
    throws_on_invalid_actions=False,
)


# In[7]:


# Wrap the environment
flatten_action_env = FlattenActionWrapper(iot_env)
flatten_obs_env = FlattenObservationWrapper(flatten_action_env, ignore_fields=[
    "_credential_cache",
    "_discovered_nodes",
    "_explored_network",
    "action_mask"
])


# In[8]:


# Cast to GymEnv
env_as_gym = cast(GymEnv, flatten_obs_env)


# In[9]:


# Check the environment
check_env(flatten_obs_env)


# ## Set Hyperparameters
#
# Define the hyperparameters for training the SAC agent.

# In[10]:


# Training hyperparameters
EPISODES = 10
MAX_STEPS = 1000
GAMMA = 0.99  # discount factor
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
ALPHA = 0.2  # entropy coefficient


# ## Train the SAC Agent
#
# Train the SAC agent on the CyberBattleIoT environment.

# In[11]:


# Train the agent
agent, rewards = sac_training(
    env_as_gym,
    episodes=EPISODES,
    replay_size=REPLAY_MEMORY_SIZE,
    gamma=GAMMA,
    alpha=ALPHA,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE
)


# In[ ]:


print("Training completed successfully!")


# ## Plot Training Results
#
# Visualize the training results.

# In[ ]:


# Plot the rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('SAC Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()


# ## Evaluate the Trained Agent
#
# Evaluate the trained agent on the environment.

# In[ ]:


# Evaluate the agent
def evaluate_agent(env, agent, num_episodes=5):
    eval_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, eval=True)  # Use deterministic actions for evaluation
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        eval_rewards.append(total_reward)
        print(f"Evaluation Episode {ep}, Reward: {total_reward:.2f}")

    return eval_rewards


# In[ ]:


# Run evaluation
eval_rewards = evaluate_agent(env_as_gym, agent, num_episodes=3)


# In[ ]:


# Print average evaluation reward
print(f"Average evaluation reward: {np.mean(eval_rewards):.2f}")

