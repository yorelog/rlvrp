import matplotlib.pyplot as plt
import torch

def plot_training_loss(train_loss_log):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_log, label="Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

def main():
    # Load training loss log
    train_loss_log = torch.load("train_loss_log.pt")

    # Plot training loss
    plot_training_loss(train_loss_log)

if __name__ == "__main__":
    main()
