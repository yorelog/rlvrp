import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical
import pytorch_lightning as pl

class TemporalGraphAttentionNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr, clip_epsilon, num_time_periods):
        super(TemporalGraphAttentionNetwork, self).__init__()
        self.save_hyperparameters()

        self.gat1 = GATConv(input_size, hidden_size)
        self.gat2 = GATConv(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.time_embedding = nn.Embedding(num_time_periods, hidden_size)
        self.time_window_size = 2

    def forward(self, data):
        x, edge_index, num_time_periods, time_windows = data.x, data.edge_index, data.num_time_periods, data.time_windows

        time_periods = torch.arange(num_time_periods, device=x.device).float()
        time_embeddings = self.time_embedding(time_periods)  # shape: (num_time_periods, hidden_size)

        # Add time embedding to node features
        time_window_starts = time_windows[:, 0].unsqueeze(-1)
        in_time_window = (time_periods >= time_window_starts) & (time_periods < time_window_starts + self.time_window_size)
        in_time_window = in_time_window.unsqueeze(-1).float()
        x = torch.cat([x, time_embeddings, in_time_window], dim=1)  # shape: (num_nodes, input_size)

        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        return Categorical(logits=self.policy_head(x)), self.value_head(x)


    def training_step(self, batch, batch_idx):
        policy, value = self(batch)
        action = policy.sample()
        log_prob = policy.log_prob(action)

        # Calculate rewards based on travel time and time window penalty
        rewards = -batch.travel_time.gather(1, action.unsqueeze(1)).squeeze()
        time_windows = batch.time_windows
        time_penalty = torch.where((action < time_windows[:, 0]) | (action >= time_windows[:, 1]), -1.0, 0.0)
        rewards += time_penalty

        # Calculate advantage for each action
        advantages = rewards - value.squeeze()

        if not hasattr(batch, 'old_log_probs'):
            batch.old_log_probs = log_prob.detach()
        else:
            batch.old_log_probs = batch.old_log_probs.detach()

        # PPO objective function
        ratio = torch.exp(log_prob - batch.old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.hparams.clip_epsilon, 1 + self.hparams.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + value_loss

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
    
# def create_data(positions, travel_times, time_periods):
#     num_nodes = len(positions)
#     num_time_periods = len(time_periods)
#     edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     travel_times_tensor = torch.tensor(travel_times, dtype=torch.float)
#     time_periods_tensor = torch.tensor(time_periods, dtype=torch.long)
#     x = torch.tensor(positions, dtype=torch.float)

#     return Data(x=x, edge_index=edge_index, travel_time=travel_times_tensor, time_periods=time_periods_tensor)
