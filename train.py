import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer, loggers
import data
import model

def main():
    # Load data
    num_nodes_list = [10, 20, 30]
    num_time_periods = 5
    time_window_size = 2
    seed = 1235
    dataset = []

    for num_nodes in num_nodes_list:
        positions, time_periods, densities, speeds, travel_times = data.generate_data(num_nodes, num_time_periods, seed)
        graph_data = data.create_data(positions, time_periods, travel_times, time_window_size) # Add time_window_size as an argument
        dataset.append(graph_data)

    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, follow_batch=[])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4, follow_batch=[])

    # Create a PyTorch Lightning Trainer
    tb_logger = loggers.TensorBoardLogger("logs/")
    trainer = Trainer(
        max_epochs=100,
        logger=tb_logger,
        log_every_n_steps=5,
    )

    # Define hyperparameters
    hparams = {
        "lr": 1e-3,
        "clip_epsilon": 0.2,
        "input_size": 2 + 1 + 64,  # Including time windows
        "hidden_size": 64,
        "output_size": 1,
        "num_time_periods": 5,
    }

    # Create and train the model
    gat_model = model.TemporalGraphAttentionNetwork(**hparams)
    trainer.fit(gat_model, train_dataloader, test_dataloader)
    # Save the training loss log
    torch.save(model.loss_log, "train_loss_log.pt")

if __name__ == "__main__":
    main()