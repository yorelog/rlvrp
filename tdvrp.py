import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
import data

class GATActor(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, num_classes):
        super(GATActor, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)

class GATCritic(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads):
        super(GATCritic, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, 1, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)

def train(actor, critic, positions, time_periods, densities, speeds, travel_times, num_episodes, learning_rate, gamma, lam):
    optimizer_actor = Adam(actor.parameters(), lr=learning_rate)
    optimizer_critic = Adam(critic.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        episode_travel_times = []
        log_probs = []
        values = []
        rewards = []
        masks = []

        for t in range(len(time_periods) - 1):
            data_t = data.create_data(positions, time_periods, travel_times, t)
            x, edge_index = data_t.x, data_t.edge_index
            x = x.to(device)
            edge_index = edge_index.to(device)

            prob = actor(x, edge_index)
            value = critic(x, edge_index)
            dist = Categorical(prob)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            reward, mask = get_reward_and_mask(action, positions, densities, speeds, travel_times, time_periods, t)
            travel_time = get_travel_time(action, positions, densities, speeds, travel_times, time_periods, t)

            episode_travel_times.append(travel_time)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(mask)

        # Calculate the returns and advantages
        returns = []
        advantages = []
        prev_return = 0
        prev_advantage = 0
        for reward, mask, value in reversed(zip(rewards, masks, values)):
            return_ = reward + gamma * prev_return * mask
            advantage = reward + gamma * prev_value * mask - value
            returns.insert(0, return_)
            advantages.insert(0, advantage)
            prev_return = return_
            prev_advantage = advantage

        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        # Update the actor and critic networks using PPO
        for _ in range(ppo_epochs):
            indices = torch.randperm(len(returns))
            for start in range(0, len(returns), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                sampled_log_probs = log_probs[batch_indices]
                sampled_values = values[batch_indices]

                new_prob = actor(x, edge_index)
                new_dist = Categorical(new_prob)
                new_action = new_dist.sample()
                new_log_prob = new_dist.log_prob(new_action)

                ratio = torch.exp(new_log_prob - sampled_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                policy_loss = -torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages).mean()

                value_loss = F.mse_loss(sampled_values, sampled_returns)

                optimizer_actor.zero_grad()
                policy_loss.backward()
                optimizer_actor.step()

                optimizer_critic.zero_grad()
                value_loss.backward()
                optimizer_critic.step()

        # Compute the mean travel time for the current episode
        mean_travel_time = np.mean(episode_travel_times)
        print(f"Episode {episode + 1}/{num_episodes}: Mean travel time: {mean_travel_time}")

if __name__ == "__main__":
    num_nodes = 10
    num_time_periods = 5
    num_features = 3
    hidden_channels = 32
    num_heads = 4
    num_classes = num_nodes
    num_episodes = 100
    learning_rate = 0.001
    gamma = 0.99
    lam = 0.95
    ppo_epochs = 4
    batch_size = 64
    clip_epsilon = 0.2
    positions, time_periods, densities, speeds, travel_times = data.generate_data(num_nodes, num_time_periods)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = GATActor(num_features, hidden_channels, num_heads, num_classes).to(device)
    critic = GATCritic(num_features, hidden_channels, num_heads).to(device)

    train(actor, critic, positions, time_periods, densities, speeds, travel_times, num_episodes, learning_rate, gamma, lam)
