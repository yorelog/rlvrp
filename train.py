import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer, loggers
import data
import model

def main():
    # Load data
    num_nodes_list = [10, 20, 30]
    dataset = []

    for num_nodes in num_nodes_list:
        positions, speeds, travel_times = data.generate_data(num_nodes)
        graph_data = data.create_data(positions, travel_times)
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
        "input_size": 2,
        "hidden_size": 64,
        "output_size": 1,
    }

    # Create and train the model
    gat_model = model.TemporalGraphAttentionNetwork(**hparams)
    trainer.fit(gat_model, train_dataloader, test_dataloader)
    # Save the training loss log
    torch.save(model.loss_log, "train_loss_log.pt")

if __name__ == "__main__":
    main()