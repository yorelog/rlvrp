import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Generate data
def generate_data(num_nodes, num_time_periods, seed=7584):
    np.random.seed(seed)
    positions = np.random.rand(num_nodes, 2)
    time_periods = np.linspace(0, 1, num_time_periods)
    density_params = np.array([0.2, -2, 12])  # Quadratic function parameters for city-wide vehicle density
    densities, speeds, travel_times = generate_densities_speeds_travel_times(positions, time_periods, density_params)
    return positions, time_periods, densities, speeds, travel_times

def generate_time_windows_and_service_times(num_nodes, max_time_window_length=1):
    time_windows = np.zeros((num_nodes, 2))
    
    for i in range(num_nodes):
        start_time = np.random.rand() * (1 - max_time_window_length)
        end_time = start_time + max_time_window_length
        time_windows[i] = [start_time, end_time]
    
    return time_windows

def generate_densities_speeds_travel_times(positions, time_periods, density_params):
    num_nodes = len(positions)
    num_time_periods = len(time_periods)
    travel_times = np.zeros((num_nodes, num_nodes, num_time_periods))
    speeds = np.zeros((num_nodes, num_nodes, num_time_periods))
    densities = np.zeros((num_nodes, num_nodes, num_time_periods))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            for t in range(num_time_periods):

                if i == j:
                    continue
                
                distance = np.linalg.norm(positions[i] - positions[j])

                # 初始密度为当前时间段的密度 
                jam_density = np.polyval(density_params, time_periods[t])
                # assert jam_density > 0 
                remaining_sum = jam_density - densities[:i, :j].sum()
                remaining_count = (num_nodes - i) * (num_nodes - j)
                mean = remaining_sum / remaining_count
                # 不拥堵，密度为0
                if mean < 0:
                    mean = 0
                #使得平均密度为当前jam_density
                densities[i, j, t] = np.random.normal(mean, mean / 2)

                speeds[i, j, t] = lwr_speed(densities[i, j, t],jam_density)
                travel_times[i, j, t] = distance / speeds[i, j, t]
    return densities, speeds, travel_times

# jam_density 当前时间段的拥堵密度
# free_flow_speed 自由流速度
def lwr_speed(density,jam_density,free_flow_speed=60):
    return free_flow_speed * (1 - density / jam_density)

def create_data(positions, time_periods, travel_times, time_window_size):
    num_nodes = len(positions)
    num_time_periods = len(time_periods)
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    travel_times_tensor = torch.tensor(travel_times, dtype=torch.float)
    time_windows = np.random.randint(0, num_time_periods - time_window_size + 1, (num_nodes, 1))
    x = torch.tensor(np.hstack([positions, time_windows]), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, travel_time=travel_times_tensor, time_periods=torch.tensor(time_periods, dtype=torch.float))

def plot_density_speed_travel_time(positions, densities, speeds, travel_times, time_periods):
    num_nodes = len(positions)
    num_time_periods = len(time_periods)

    # Plot densities
    plt.figure()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                plt.plot(time_periods, densities[i, j], label=f'({i}, {j})')
    plt.xlabel('Time Periods')
    plt.ylabel('Densities')
    plt.legend()
    plt.title('Densities between nodes over time')
    plt.show()

    # Plot speeds
    plt.figure()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                plt.plot(time_periods, speeds[i, j], label=f'({i}, {j})')
    plt.xlabel('Time Periods')
    plt.ylabel('Speeds')
    plt.legend()
    plt.title('Speeds between nodes over time')
    plt.show()

    # Plot travel times
    plt.figure()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                plt.plot(time_periods, travel_times[i, j], label=f'({i}, {j})')
    plt.xlabel('Time Periods')
    plt.ylabel('Travel Times')
    plt.legend()
    plt.title('Travel times between nodes over time')
    plt.show()

if __name__ == "__main__":
    positions, time_periods, densities, speeds, travel_times = generate_data(10, 5)

    print("Positions:")
    print(positions)
    print("\nTime Periods:")
    print(time_periods)
    print("\nDensities:")
    print(densities)
    print("\nSpeeds:")
    print(speeds)
    print("\nTravel Times:")
    print(travel_times)
    plot_density_speed_travel_time(positions, densities, speeds, travel_times, time_periods)
