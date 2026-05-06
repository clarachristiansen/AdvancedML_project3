import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from torch_geometric.utils import to_dense_adj
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from model import GNNEncoder, InnerProductDecoder, GaussianPrior, GraphVAE
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import random_split
from main import ErdosRenyi

# convert adjacency matrix to NX Graph
def adj_to_nx(A):
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    else:
        A = np.asarray(A)
    A = np.squeeze(A)

    A = (A > 0.5).astype(int)
    np.fill_diagonal(A, 0)
    return nx.from_numpy_array(A)

# draws 3 NX graphs from sampled graphs
def plot_graphs(graphs, title):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i in range(3):
        nx.draw(graphs[i], ax=axes[i], node_size=20)
    plt.suptitle(title)
    plt.savefig(f"results/{title}.png", dpi=150)
    plt.show()

# calculate the average number of nodes (adjacendy representation vs NX graphs) - used to ensure the convertion works and does not modify structures
def mean_num_nodes(adj_samples, graph_samples, name_adj="Adj samples", name_graph="Graph samples"):
    # --- adjacency matrices ---
    adj_node_counts = []
    for A in adj_samples:
        A = np.asarray(A)

        # handle possible squeezing / batching
        A = np.squeeze(A)

        # assume square adjacency matrix
        n_nodes = A.shape[0]
        adj_node_counts.append(n_nodes)

    # --- networkx graphs ---
    graph_node_counts = [G.number_of_nodes() for G in graph_samples]

    # --- statistics ---
    adj_mean = np.mean(adj_node_counts)
    graph_mean = np.mean(graph_node_counts)

    print("\n=== Mean Number of Nodes ===")
    print(f"{name_adj:<20}: {adj_mean:.2f}")
    print(f"{name_graph:<20}: {graph_mean:.2f}")

    print("\n--- Details ---")
    print(f"{name_adj}: min={np.min(adj_node_counts)}, max={np.max(adj_node_counts)}")
    print(f"{name_graph}: min={np.min(graph_node_counts)}, max={np.max(graph_node_counts)}")

    return adj_node_counts, graph_node_counts


# convert continuous to binary
def binarize_adj(A, threshold=0.5):
    A = (A > threshold).astype(int)
    np.fill_diagonal(A, 0)
    return A

# convert training graph to adjacncy matrix
def graph_to_adj(data):
    A = to_dense_adj(data.edge_index)[0]
    return A.cpu().numpy()

# input: NX graphs
def compute_wl_hash(graphs):
    hashes = []
    for G in graphs:
        h = weisfeiler_lehman_graph_hash(G, iterations=3)
        hashes.append((G.number_of_nodes(), h))
    return hashes

# input: both arguments should be NX graphs
def compute_metrics(train_graphs, generated_graphs):
    print("Computing Sample Evaluation Metrics")

    # Computing graph hashes for training and generated graphs
    train_hashes = set(compute_wl_hash(train_graphs))
    generated_hashes = compute_wl_hash(generated_graphs)
    
    # Computing Novelty
    novelty_mask = [G not in train_hashes for G in generated_hashes]
    novelty = np.mean(novelty_mask)

    # Computing Uniqueness
    counts = Counter(generated_hashes)
    uniqueness_mask = [counts[G] == 1 for G in generated_hashes]
    uniqueness = np.mean(uniqueness_mask)

    # Computing Novelty + Uniqueness
    novelty_uniqueness_mask = [G not in train_hashes and counts[G] == 1 for G in generated_hashes]
    novelty_uniqueness = np.mean(novelty_uniqueness_mask)

    return novelty, uniqueness, novelty_uniqueness

# Compute graph statistics
def compute_statistics(graphs):
    # Node degrees
    node_degrees = []

    # Clustering Coefficients
    clustering_coeffs = []

    # Eigenvector Centrality
    eigenvector_cent = []

    for G in graphs:
        # Skip if empty
        if len(G) == 0:
            continue

        # Node Degree
        node_degrees.extend([degree for _, degree in G.degree()])

        # Clustering Coefficient
        clustering_coeffs.extend(list(nx.clustering(G).values()))

        # Eigenvector Centrality
        try:
            eigenvector_cent.extend(list(nx.eigenvector_centrality_numpy(G).values()))
        except:
            pass

    return node_degrees, clustering_coeffs, eigenvector_cent

# plot histograms - this expects input to be NX graphs (including for the baseline and deep generative model)
def plot_statistics(train_graphs, baseline_graphs, generated_graphs):
    print("Plotting Graph Statistics")
    train_stats = compute_statistics(train_graphs)
    baseline_stats = compute_statistics(baseline_graphs)
    generated_stats = compute_statistics(generated_graphs)

    titles = ["Degree", "Clustering Coefficient", "Eigenvector Centrality"]
    statistics = [train_stats, baseline_stats, generated_stats]
    origin = ["Training", "Baseline", "Deep Model"]

    _, axes = plt.subplots(3, 3, figsize=(12, 10))

    for row in range(3):
        for col in range(3):
            axes[row, col].hist(statistics[row][col], bins=20)
            axes[row, col].set_title(f"{origin[row]} - {titles[col]}")

    plt.tight_layout()
    plt.savefig("results/graph_statistics.png", dpi=150)
    print("Graph statistics plot saved to results/graph_statistics.png")
    plt.show()

# Load training graphs and samples
def load_graphs(adj_gen, adj_baseline):
    print("Loading Data")

    baseline_graphs = [adj_to_nx(x) for x in adj_baseline]

    generated_graphs = [adj_to_nx(x) for x in adj_gen]

    return baseline_graphs, generated_graphs

# print results in a table format
def print_results(baseline_metrics, generated_metrics):
    print("\n=== Sample Evaluation Metrics ===\n")
    print(f"{'Model':<25} {'Novel (%)':<15} {'Unique (%)':<15} {'Novel+Unique (%)':<20}")

    print(f"{'Baseline Model':<25} "
          f"{baseline_metrics[0]*100:.2f} "
          f"{baseline_metrics[1]*100:.2f} "
          f"{baseline_metrics[2]*100:.2f}")

    print(f"{'Deep Generative Model':<25} "
          f"{generated_metrics[0]*100:.2f} "
          f"{generated_metrics[1]*100:.2f} "
          f"{generated_metrics[2]*100:.2f}")

def main():
    samples = 1000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Training data
    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(42)
    train_ds, _, _ = random_split(dataset, [100, 44, 44], generator=rng)

    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True) 
    training_graphs = [data for data in train_loader.dataset]
    training_nx_graphs = [adj_to_nx(graph_to_adj(d)) for d in training_graphs]

    # Load models
    node_feature_dim = 7  # 7 for MUTAG
    STATE_DIM   = 128   
    LATENT_DIM  = 32    
    NUM_ROUNDS  = 5 
    encoder = GNNEncoder(node_feature_dim, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    decoder = InnerProductDecoder()
    prior   = GaussianPrior(LATENT_DIM)
    graph_VAE   = GraphVAE(encoder, decoder, prior).to(device)
    graph_VAE.load_state_dict(torch.load("models/graph_vae.pt", map_location=device))

    # select distribution of number of nodes for generated samples
    train_sizes = [g.number_of_nodes() for g in training_nx_graphs]

    adj_graph_VAE = [
        graph_VAE.sample(np.random.choice(train_sizes), device)
        .detach().cpu().numpy()
        for _ in range(samples)
    ]

    # Baseline model
    erdos_renyi = ErdosRenyi(dataset)
    adj_erdos_renyi = [erdos_renyi.sample().cpu().numpy() for _ in range(samples)]

    # Load graphs
    baseline_graphs, generated_graphs = load_graphs(adj_graph_VAE, adj_erdos_renyi)

    # Compute metrics
    baseline_metrics = compute_metrics(training_nx_graphs, baseline_graphs)
    generated_metrics = compute_metrics(training_nx_graphs, generated_graphs)

    # Print results
    print_results(baseline_metrics, generated_metrics)

    # Compute and plot graph statistics
    plot_statistics(training_nx_graphs, baseline_graphs, generated_graphs)

    # minor statistics computed to verify e.g. density
    for i in range(5):
        A = adj_graph_VAE[i]
        print("mean:", A.mean(), "std:", A.std())

    print("Train densities:")
    for i in range(5):
        print(nx.density(training_nx_graphs[i]))

    print("Generated densities:")
    for i in range(5):
        print(nx.density(generated_graphs[i]))

    print("Baseline densities:")
    for i in range(5):
        print(nx.density(baseline_graphs[i]))
    
    # draw graphs
    plot_graphs(training_nx_graphs, "Training Graphs")
    plot_graphs(baseline_graphs, "Baseline Graphs")
    plot_graphs(generated_graphs, "Generated Graphs")

    # just comparing average number of nodes for adjacency vs NX representations
    mean_num_nodes(adj_erdos_renyi,baseline_graphs)

if __name__ == "__main__":
    main()


    


    

