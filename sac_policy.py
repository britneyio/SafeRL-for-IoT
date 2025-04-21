
# In[1]:


import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import NamedTuple, List, Optional, Tuple, Union, Dict
from torch import Tensor
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.learner import Learner
from cyberbattle._env import cyberbattle_env
from cyberbattle.agents.baseline.agent_wrapper import EnvironmentBounds
import random
from cyberbattle.agents.baseline.agent_dql import CyberBattleStateActionModel, random_argmax, ChosenActionMetadata, Transition, ReplayMemory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


class CategoricalPolicy(nn.Module):
    def __init__(self, ep: EnvironmentBounds):
        super(CategoricalPolicy, self).__init__()
        model = CyberBattleStateActionModel(ep)
        linear_input_size = len(model.state_space.dim_sizes)
        output_size = model.action_space.flat_size()

        self.hidden_layer1 = nn.Linear(linear_input_size, 1024)
        # self.bn1 = nn.BatchNorm1d(256)
        self.hidden_layer2 = nn.Linear(1024, 512)
        self.hidden_layer3 = nn.Linear(512, 128)
        # self.hidden_layer4 = nn.Linear(128, 64)
        self.head = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.hidden_layer2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.hidden_layer3(x))
        # x = F.relu(self.hidden_layer4(x))
        return self.head(x.view(x.size(0), -1))
    def act(self, states):
        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)
                # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


# In[3]:


class Critic(nn.Module):
    def __init__(self, ep: EnvironmentBounds):
        super(Critic, self).__init__()
        model = CyberBattleStateActionModel(ep)
        linear_input_size = len(model.state_space.dim_sizes)
        output_size = model.action_space.flat_size()

        self.hidden_layer1 = nn.Linear(linear_input_size, 1024)
        # self.bn1 = nn.BatchNorm1d(256)
        self.hidden_layer2 = nn.Linear(1024, 512)
        self.hidden_layer3 = nn.Linear(512, 128)
        # self.hidden_layer4 = nn.Linear(128, 64)
        self.head = nn.Linear(128, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.hidden_layer2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.hidden_layer3(x))
        # x = F.relu(self.hidden_layer4(x))
        return self.head(x.view(x.size(0), -1))


# In[7]:


from cyberbattle.agents.baseline.agent_dql import CyberBattleStateActionModel
from gymnasium.spaces.utils import flatten_space
import numpy
from cyberbattle._env.cyberbattle_env import Action
class SACAgent(Learner):
    def __init__(self, ep: EnvironmentBounds, gamma=0.99, tau=0.005, alpha=0.2,
                 learning_rate=3e-4, replay_memory_size=1000000, batch_size=256):
        self.stateaction_model = CyberBattleStateActionModel(ep)
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.batch_size = batch_size
        self._last_actor_loss = 0.0
        self._last_critic_loss = 0.0

        self.actor = CategoricalPolicy(ep).to(device)
        self.critic1 = Critic(ep).to(device)
        self.critic2 = Critic(ep).to(device)
        self.critic1_target = Critic(ep).to(device)
        self.critic2_target = Critic(ep).to(device)
        #hard update
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.memory = ReplayMemory(replay_memory_size)



    def parameters_as_string(self):
        return f"γ={self.gamma}, τ={self.tau}, α={self.alpha}, batch={self.batch_size}, memory={self.memory.capacity}"
    def all_parameters_as_string(self) -> str:
        model = self.stateaction_model
        return (
            f"{self.parameters_as_string()}\n"
            f"dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, "
            f"Q={[f.name() for f in model.state_space.feature_selection]} "
            f"-> 'abstract_action'"
        )
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def optimize_model(self, norm_clipping=False):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map((lambda s: s is not None), batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        with torch.no_grad():
            next_probs = self.actor(next_state_batch)
            min_next_q = torch.min(self.critic1_target(next_state_batch), self.critic2_target(next_state_batch))
            next_value = (next_probs * (min_next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=1)
            target_q = reward_batch + self.gamma * next_value

        q1 = self.critic1(state_batch).gather(1, action_batch)
        q2 = self.critic2(state_batch).gather(1, action_batch)

        critic1_loss = F.mse_loss(q1, target_q.unsqueeze(1))
        critic2_loss = F.mse_loss(q2, target_q.unsqueeze(1))

        self.critic1_optimizer.zero_grad(); critic1_loss.backward(); self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(); critic2_loss.backward(); self.critic2_optimizer.step()

        probs = self.actor(state_batch)
        min_q = torch.min(self.critic1(state_batch), self.critic2(state_batch))
        actor_loss = (probs * (self.alpha * torch.log(probs) - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        return self._last_actor_loss, critic1_loss.item(), critic2_loss.item()

    def update_q_function(
        self,
        reward: float,
        actor_state: ndarray,
        abstract_action: np.int32,
        next_actor_state: Optional[ndarray],
    ):
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
        action_tensor = torch.tensor([[np.int_(abstract_action)]], device=device, dtype=torch.long)
        current_state_tensor = torch.as_tensor(actor_state, dtype=torch.float, device=device).unsqueeze(0)
        if next_actor_state is None:
            next_state_tensor = None
        else:
            next_state_tensor = torch.as_tensor(next_actor_state, dtype=torch.float, device=device).unsqueeze(0)
        self.memory.push(current_state_tensor, action_tensor, next_state_tensor, reward_tensor)

        self.optimize_model()

    def on_step(
        self,
        wrapped_env: w.AgentWrapper,
        observation,
        reward: float,
        done: bool,
        truncated: bool,
        info,
        action_metadata,
    ):
        agent_state = wrapped_env.state
        if done:
            self.update_q_function(
                reward,
                actor_state=action_metadata.actor_state,
                abstract_action=action_metadata.abstract_action,
                next_actor_state=None,
            )
        else:
            next_global_state = self.stateaction_model.global_features.get(agent_state, node=None)
            next_actor_features = self.stateaction_model.node_specific_features.get(agent_state, action_metadata.actor_node)
            next_actor_state = self.get_actor_state_vector(next_global_state, next_actor_features)

            self.update_q_function(
                reward,
                actor_state=action_metadata.actor_state,
                abstract_action=action_metadata.abstract_action,
                next_actor_state=next_actor_state,
            )


    def metadata_from_gymaction(self, wrapped_env, gym_action):
        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.stateaction_model.node_specific_features.get(wrapped_env.state, actor_node)
        abstract_action = self.stateaction_model.action_space.abstract_from_gymaction(gym_action)
        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features),
        )
    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray) -> ndarray:
        return np.concatenate(
            (
                np.array(global_state, dtype=np.float32),
                np.array(actor_features, dtype=np.float32),
            )
        )
    def end_of_episode(self, i_episode, t):
        # Update the target network, copying all weights and biases in DQN
        if i_episode % t == 0:
            self.critic1_target.load_state_dict(self.actor.state_dict())
            self.critic2_target.load_state_dict(self.actor.state_dict())

    def select_discrete_action(self, wrapped_env, device="cpu", eval_mode=False):
        """
        Flattens dict-based observation and samples action using a discrete policy.

        Args:
            policy_net (nn.Module): Your policy network (output is logits over discrete actions)
            observation_space (gym.Space): The Dict observation space of the environment
            observation_dict (dict): Actual observation returned from env
            device (str): Device for torch tensor
            eval_mode (bool): If True, selects greedy action; else samples stochastically

        Returns:
            action (int): The selected action index
            probs (Tensor): Action probabilities
            log_prob (Tensor): Log prob of selected action (None in eval mode)
        """
        # Flatten dict observation into vector
        flat_obs = flatten_space(wrapped_env.observation)
        obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Get logits and sample
        logits = self.actor(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if eval_mode:
            action = torch.argmax(probs, dim=-1)
            log_prob = None
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), probs, log_prob
    def map_index_to_action(self, action_index: int,
                        from_node: int,
                        bounds,
                        discovered_nodes: list[int],
                        credential_cache_len: int) -> Action:
        """
        Maps a discrete action index into a CyberBattleSim Action TypedDict.

        Args:
            action_index (int): The flat action index from the policy
            from_node (int): The node initiating the action (must be owned)
            bounds (EnvironmentBounds): The environment bounds
            discovered_nodes (list[node_identifier]): External indices of discovered nodes
            credential_cache_len (int): Number of discovered credentials

        Returns:
            Action (TypedDict): One of local_vulnerability / remote_vulnerability / connect
        """
        local_total = bounds.local_attacks_count
        remote_total = len(discovered_nodes) * bounds.remote_attacks_count
        connect_total = len(discovered_nodes) * bounds.port_count * credential_cache_len

        if action_index < local_total:
            # Local vulnerability
            vuln_index = action_index
            return Action(local_vulnerability=numpy.array([from_node, vuln_index], dtype=numpy.int32))

        elif action_index < local_total + remote_total:
            # Remote vulnerability
            offset = action_index - local_total
            target_index = offset // bounds.remote_attacks_count
            vuln_index = offset % bounds.remote_attacks_count
            return Action(remote_vulnerability=numpy.array(
                [from_node, target_index, vuln_index], dtype=numpy.int32))

        else:
            # Connect
            offset = action_index - (local_total + remote_total)
            product = bounds.port_count * credential_cache_len
            target_index = offset // product
            port_index = (offset % product) // credential_cache_len
            cred_index = (offset % product) % credential_cache_len

            return Action(connect=numpy.array(
                [from_node, target_index, port_index, cred_index], dtype=numpy.int32))

    def explore(self, wrapped_env: w.AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        """Random exploration that avoids repeating actions previously taken in the same state"""
        # sample local and remote actions only (excludes connect action)
        gym_action, _, _ = self.select_discrete_action(wrapped_env=wrapped_env)
        gym_action = self.map_index_to_action(gym_action, wrapped_env.state.node, wrapped_env.bounds, wrapped_env.observation["_discovered_nodes"], len(w.owned_nodes(wrapped_env.observation)))
        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
        return "explore", gym_action, metadata


# In[5]:


from final_project.iot_env import CyberBattleIoT
import gymnasium as gym
from typing import cast
from cyberbattle.agents.baseline.agent_wrapper import AgentWrapper
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation
env = cast(CyberBattleIoT, gym.make("CyberBattleIoT-v0").unwrapped)

ep = w.EnvironmentBounds.of_identifiers(maximum_node_count=30, maximum_total_credentials=50, identifiers=env.identifiers)
o, _ = env.reset()
wrapped_env = AgentWrapper(env,ActionTrackingStateAugmentation(ep, o))
batch_size = 64

agent = SACAgent(
    ep=ep,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    learning_rate=3e-4,
    replay_memory_size=10000,
    batch_size=batch_size)


# In[6]:


total_numsteps = 0
updates = 0
total_reward = 0
for i_episode in range(100):
    episode_reward = 0
    episode_steps = 0
    done = False

    observation, _ = wrapped_env.reset()

    while not done:
        # Always sample action from current stochastic policy (SAC exploration)
        action_style, gym_action, metadata = agent.explore(wrapped_env)

        if gym_action is None:
            break  # Skip if invalid action was returned

        # Take the action in the environment
        next_observation, reward, done, truncated, info = wrapped_env.step(gym_action)
        print(f"Action: {gym_action}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
        # Store the transition and train agent if ready
        agent.on_step(wrapped_env, next_observation, reward, done, truncated, info, metadata)

        total_numsteps += 1
        episode_steps += 1
        episode_reward += reward
        observation = next_observation

    print(f"Episode: {i_episode}, Total Steps: {total_numsteps}, Steps: {episode_steps}, Reward: {round(episode_reward, 2)}")

    # Evaluation every 10 episodes
    if i_episode % 10 == 0:
        avg_reward = 0.0
        for _ in range(10):
            observation, _ = wrapped_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action_style, gym_action, metadata = agent.exploit(wrapped_env, observation)
                if not gym_action:
                    #stats["exploit_deflected_to_explore"] += 1
                    _, gym_action, action_metadata = learner.explore(wrapped_env)
                if gym_action is None:
                    break
                observation, reward, done, truncated, info = env.step(gym_action)
                episode_reward += reward
            avg_reward += episode_reward
        avg_reward /= 10
        print(f"[Eval] Episode {i_episode}, Avg Reward: {avg_reward:.2f}")
env.render()
env.close()


# In[48]:


class SafetyCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        # One-hot encode the action
        action_onehot = F.one_hot(action.squeeze(-1), num_classes=self.fc3.out_features).float()
        x = torch.cat([state, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # returns value in [0, 1]


# In[44]:


class SafeSACPolicy(SACAgent):
    def __init__(self, *args, safety_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.stateaction_model.state_space.flat_size()
        action_dim = self.stateaction_model.action_space.flat_size()

        self.safety_critic = SafetyCritic(state_dim, action_dim, hidden_dim=256).to(device)
        self.safety_critic_optimizer = optim.Adam(self.safety_critic.parameters(), lr=3e-4)
        self.safety_threshold = safety_threshold
        self.in_finetune_phase = False

    def enable_finetuning(self):
        self.in_finetune_phase = True

    def disable_finetuning(self):
        self.in_finetune_phase = False

    def update_safety_critic(self, batch: Transition, unsafe_labels: List[int]) -> float:
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        label_tensor = torch.tensor(unsafe_labels, dtype=torch.float, device=device).unsqueeze(1)

        pred = self.safety_critic(state_batch, action_batch)
        loss = F.binary_cross_entropy(pred, label_tensor)

        self.safety_critic_optimizer.zero_grad()
        loss.backward()
        self.safety_critic_optimizer.step()
        return loss.item()

    def select_action(self, state: Tensor, evaluate: bool = False) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if not self.in_finetune_phase:
            return super().select_action(state, evaluate)

        with torch.no_grad():
            action_probs = self.actor(state)
            all_actions = torch.arange(action_probs.shape[-1], device=device).unsqueeze(0).repeat(state.shape[0], 1)

            safety_scores = self.safety_critic(state.repeat_interleave(action_probs.shape[-1], dim=0),
                                               all_actions.view(-1, 1))
            safety_scores = safety_scores.view(state.shape[0], -1)

            mask = (safety_scores > self.safety_threshold).float()
            filtered_probs = action_probs * mask
            filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-8)

            dist = Categorical(filtered_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, filtered_probs, log_prob

    def new_episode(self):
        """Called at the start of each new episode"""
        pass


# In[45]:


def safe_sac_training(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    learner: 'SafeSACPolicy',
    title: str,
    pretrain_episodes: int,
    finetune_episodes: int,
    iteration_count: int,
    initial_alpha: float = 0.2,
    alpha_decay: float = 0.995,
    min_alpha: float = 0.01,
    render: bool = True,
    verbosity: w.Verbosity = w.Verbosity.Normal,
    label_unsafe_fn = None  # Custom unsafe labeling function
) -> dict:
    """
    SafeSAC training loop with pretraining + finetuning phases.

    Parameters
    ==========
    learner -- the SafeSACPolicy learner
    label_unsafe_fn -- function that returns a list of 0 (unsafe) or 1 (safe) labels
                      given (state, action, reward) transitions
    """
    stats = {
        'train_rewards': [],
        'episode_lengths': [],
        'actor_losses': [],
        'critic_losses': [],
        'alphas': []
    }
    initial_observation, info = cyberbattle_gym_env.reset()

    # Initialize environment with wrapper
    wrapped_env = w.AgentWrapper(
        cyberbattle_gym_env,
        w.ActionTrackingStateAugmentation(environment_properties,initial_observation)
    )

    def run_phase(phase_name, num_episodes, finetune=False):
        if finetune:
            learner.enable_finetuning()
        else:
            learner.disable_finetuning()

        for episode in range(num_episodes):
            learner.new_episode()
            current_alpha = max(initial_alpha * (alpha_decay ** episode), min_alpha)
            learner.alpha = current_alpha
            total_reward = 0
            steps = 0
            transition_data = []

            # Reset environment and get initial observation
            observation, info = wrapped_env.reset()
            if observation is None:
                print(f"Warning: Received None observation on reset in {phase_name} episode {episode}")
                continue

            done = False
            truncated = False

            while not done and not truncated and steps < iteration_count:
                action_name, action, metadata = learner.explore(wrapped_env)
                if action is None:
                    break

                next_observation, reward, terminated, truncated, info = wrapped_env.step(action)
                if next_observation is None:
                    print(f"Warning: Received None observation on step in {phase_name} episode {episode}")
                    break

                done = terminated or truncated

                # Store transition for safety critic
                if not finetune and metadata is not None:
                    transition_data.append((
                        metadata.actor_state,
                        metadata.abstract_action,
                        reward
                    ))

                total_reward += reward
                steps += 1
                observation = next_observation

            if not finetune and label_unsafe_fn is not None and transition_data:
                unsafe_labels = label_unsafe_fn(transition_data)
                states = [torch.tensor([s], dtype=torch.float32, device=device) for s, _, _ in transition_data]
                actions = [torch.tensor([[a]], dtype=torch.long, device=device) for _, a, _ in transition_data]
                rewards = [torch.tensor([r], dtype=torch.float32, device=device) for _, _, r in transition_data]

                # Create valid next_states list with empty tensors for terminal states
                next_states = [torch.zeros_like(states[0]) for _ in range(len(states))]

                batch = Transition(states, actions, next_states, rewards)
                safety_loss = learner.update_safety_critic(batch, unsafe_labels)
                if verbosity != w.Verbosity.Quiet:
                    print(f"Safety critic loss: {safety_loss:.4f}")

            stats['train_rewards'].append(total_reward)
            stats['episode_lengths'].append(steps)
            stats['alphas'].append(current_alpha)
            if hasattr(learner, 'last_actor_loss'):
                stats['actor_losses'].append(learner.last_actor_loss)
            if hasattr(learner, 'last_critic_loss'):
                stats['critic_losses'].append(learner.last_critic_loss)

            if verbosity != w.Verbosity.Quiet:
                print(f"{phase_name} Episode {episode + 1}/{num_episodes} - Steps: {steps}, Reward: {total_reward:.2f}, Alpha: {current_alpha:.3f}")

            if render:
                wrapped_env.render()

    print("\n[Phase 1] Pretraining on base environment...")
    run_phase("Pretraining", pretrain_episodes, finetune=False)

    print("\n[Phase 2] Finetuning on target task with safety constraints...")
    run_phase("Finetuning", finetune_episodes, finetune=True)

    return {
        'learner': learner,
        'stats': stats,
        'title': title,
        'trained_on': str(cyberbattle_gym_env),
        'final_policy': learner.actor.state_dict()
    }


# In[49]:


agent = SafeSACPolicy(
    ep=ep,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    learning_rate=3e-4,
    replay_memory_size=10000,
    batch_size=64)
def dummy_label_unsafe_fn(transitions):
    # Example: mark as unsafe if reward < 0
    return [0 if r >= 0 else 1 for _, _, r in transitions]


# In[50]:


results = safe_sac_training(
    cyberbattle_gym_env=wrapped_env,
    environment_properties=ep,
    learner=agent,
    title="SmartHomeSafeSAC",
    pretrain_episodes=5,
    finetune_episodes=5,
    iteration_count=30,
    render=False,
    label_unsafe_fn=dummy_label_unsafe_fn
)

