
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from iot_env import CyberBattleIoT
from cyberbattle.agents.baseline import agent_dql
from cyberbattle.agents.baseline.agent_dql import DeepQLearnerPolicy
#from cyberbattle.agents.baseline.agent_wrapper import train
from iot_agents import DiscreteSACAgent, SafeSACAgent, ReplayBuffer  # your custom SAC agent
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


# Hyperparameters
EPISODES = 50
MAX_STEPS = 1000
SAFE_THRESHOLD = 0.5  # For SafeSAC filtering



def is_unsafe_state(state):
    # Basic heuristic: if state reveals access to a sensitive node, consider it unsafe
    risky_keywords = ["BabyMonitor", "DoorLock", "Thermostat"]
    state_str = str(state)
    return any(keyword in state_str for keyword in risky_keywords)

def run_dql(env, ep, title, episodes, steps, epsilon, gamma, learning_rate, batch_size, epsilon_decay, target_update,replay_memory_size):
    print("Training DQL...")
    dqn = DeepQLearnerPolicy(
            ep=ep,
            gamma=gamma,
            replay_memory_size=replay_memory_size,
            target_update=target_update,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    result = learner.epsilon_greedy_search(
        cyberbattle_gym_env=env,
        environment_properties=ep,
        learner=dqn,
        title=title,
        episode_count=episodes,
        iteration_count=steps, 
        epsilon=epsilon,
        epsilon_exponential_decay=epsilon_decay,
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        render=True,
        plot_episodes_length=True
    )

    return result

def sac_training(env, episodes=1000, replay_size=10000, batch_size=64, **kwargs):
    obs_space = 2 * env.bounds.maximum_node_count
    act_space = env.bounds.maximum_node_count * 1 + env.bounds.local_attacks_count + env.bounds.remote_attacks_count

    n_actions = act_space
    agent = DiscreteSACAgent(obs_space, act_space, **kwargs)
    memory = ReplayBuffer(replay_size)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                agent.update(memory, batch_size)

        print(f"Episode {ep}, Reward: {total_reward:.2f}")

    return agent



def run_safe_sac(env, episodes, steps):
    print("Pretraining SAC for SafeSAC...")
    env = CyberBattleIoT()
    agent = SACAgent(env)
    q_safe = {}  # simulate safety critic
    pretrain_rewards = []

    # Pretraining Phase
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            unsafe = is_unsafe_state(next_state)
            q_safe[(str(state), action)] = 0.0 if unsafe else 1.0
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update_parameters()
            state = next_state
            ep_reward += reward
            if done:
                break
        pretrain_rewards.append(ep_reward)

    print("Finetuning with SafeSAC constraints...")
    finetune_rewards = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(steps):
            unsafe_actions = [a for a in range(env.action_space.n)
                              if q_safe.get((str(state), a), 1.0) < SAFE_THRESHOLD]
            action = agent.select_action(state)
            while action in unsafe_actions:
                action = agent.select_action(state)  # reject unsafe
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update_parameters()
            state = next_state
            ep_reward += reward
            if done:
                break
        finetune_rewards.append(ep_reward)

    return pretrain_rewards, finetune_rewards

def plot_results(dql, sac, safe_pre, safe_fine):
    plt.figure()
    plt.plot(dql, label="DQL")
    plt.plot(sac, label="SAC")
    plt.plot(safe_pre, label="SafeSAC Pretrain")
    plt.plot(safe_fine, label="SafeSAC Finetune")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Agent Reward Comparison")
    plt.legend()
    plt.grid()
    plt.savefig("training_results.png")
    plt.show()

# +
iot_env = CyberBattleIoT(
    maximum_node_count=12,
    maximum_total_credentials=10,
    observation_padding=True,
    throws_on_invalid_actions=False,
)
# iot_env = cast(CyberBattleEnv, _gym_env)

# ep = w.EnvironmentBounds.of_identifiers(maximum_node_count=12, maximum_total_credentials=10, identifiers=iot_env.identifiers)

flatten_action_env = FlattenActionWrapper(iot_env)
flatten_obs_env = FlattenObservationWrapper(flatten_action_env, ignore_fields=[
    "_credential_cache",
    "_discovered_nodes",
    "_explored_network",
])

episodes = 10
steps = 100
gamma = 0.9 # discount factor
#safe_gamma = 0.99 # safe discount factor
learning_rate = 1e-3
batch_size = 100 # Deep Q-learning batch
epsilon_decay = 0.01
target_update = 5 #Deep Q-learning replay frequency (in number of episodes)
replay_memory_size = 1000 # replay_memory_size -- size of the replay memory
epsilon  =0.9
 # NOTE: Given the relatively low number of training episodes (50,
# a high learning rate of .99 gives better result
# than a lower learning rate of 0.25 (i.e. maximal rewards reached faster on average).
# Ideally we should decay the learning rate just like gamma and train over a
# much larger number of episodes



env_as_gym = cast(GymEnv, flatten_obs_env)
check_env(flatten_obs_env)

#dql_rewards = run_dql(env=iot_env, ep=ep, title="DQN in IoT Env", episodes=episodes, steps=steps, epsilon=epsilon, gamma=gamma, learning_rate=learning_rate, batch_size=batch_size, epsilon_decay=epsilon_decay, target_update=target_update,replay_memory_size=replay_memory_size)
#sac
EPISODES = 10
MAX_STEPS = 1000
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPSILON = 0.9
EPSILON_DECAY = 5000
TARGET_UPDATE = 5
REPLAY_MEMORY_SIZE = 10000
sac_rewards, failure_counts, avg_returns = sac_training(
    env=env_as_gym,
    episodes=EPISODES,
    replay_size=REPLAY_MEMORY_SIZE,
    gamma=GAMMA,
    alpha=0.2,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
)
# total_rewards, failure_counts, avg_returns = sac_training(env_as_gym, episodes=episodes, replay_size=replay_memory_size, seed=0, gamma=gamma, alpha=0.2, lr=learning_rate,
#                  hidden_size=256, batch_size=batch_size, updates_per_step=1, start_steps=1000)
#sac_rewards = run_sac(env_as_gym, episodes, steps)
#safe_pre, safe_fine = run_safe_sac(env_as_gym, episodes, steps)
#plot_results(dql_rewards, sac_rewards, safe_pre, safe_fine)
# -




