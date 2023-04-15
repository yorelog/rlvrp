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
    def __init__(self, input_size, hidden_size, output_size, lr, clip_epsilon):
        super(TemporalGraphAttentionNetwork, self).__init__()
        self.save_hyperparameters()

        self.gat1 = GATConv(input_size, hidden_size)
        self.gat2 = GATConv(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        return Categorical(logits=self.policy_head(x)), self.value_head(x)

    def training_step(self, batch, batch_idx):
        policy, value = self(batch)
        action = policy.sample()
        log_prob = policy.log_prob(action)

        # Calculate rewards based on travel time
        rewards = -batch.travel_time.gather(1, action.unsqueeze(1)).squeeze()

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
