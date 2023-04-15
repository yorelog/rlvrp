import numpy as np
import torch
from torch_geometric.data import Data

def generate_data(num_nodes, seed=1234):
    np.random.seed(seed)
    positions = np.random.rand(num_nodes, 2)
    speeds, travel_times = generate_speeds_travel_times(positions)
    return positions, speeds, travel_times

def generate_speeds_travel_times(positions):
    num_nodes = len(positions)
    travel_times = np.zeros((num_nodes, num_nodes))
    speeds = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                speed = np.random.normal(60, 10)  # 速度符合正态分布，均值为60，标准差为10
                travel_time = distance / speed
                speeds[i, j] = speed
                travel_times[i, j] = travel_time

    return speeds, travel_times

def create_data(positions, travel_times):
    num_nodes = len(positions)
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    travel_times_tensor = torch.tensor(travel_times, dtype=torch.float)
    x = torch.tensor(positions, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, travel_time=travel_times_tensor)

if __name__ == "__main__":
    positions, speeds, travel_times = generate_data(10)
    print("Positions:")
    print(positions)
    print("\nSpeeds:")
    print(speeds)
    print("\nTravel Times:")
    print(travel_times)
