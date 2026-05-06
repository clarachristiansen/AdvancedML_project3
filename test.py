import os
import random
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from torch_geometric.utils import to_dense_adj
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from model import GNNEncoder, InnerProductDecoder, GaussianPrior, GraphVAE, MLPDecoder
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import random_split
from main import ErdosRenyi
from model_graph import GraphVAE as GraphVAE_Graph, sample_graphs
from sample_evaluation_metrics import plot_graphs


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_size_distribution(dataset):
    sizes = [data.num_nodes for data in dataset]
    probs = torch.bincount(torch.tensor(sizes)).float()
    probs /= probs.sum()
    return probs

def adj_to_nx(A):
    """Convert an adjacency matrix (tensor or ndarray) to a NetworkX graph."""
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    else:
        A = np.asarray(A)
    A = np.squeeze(A)
    return nx.from_numpy_array(A)


def graph_to_adj(data):
    """Convert a PyG Data object to a dense numpy adjacency matrix."""
    A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    return A.cpu().numpy()


# ---------------------------------------------------------------------------
# Weisfeiler-Lehman graph hashing
# ---------------------------------------------------------------------------

def compute_wl_hash(adjacency_matrices):
    """Return a list of WL hashes, one per adjacency matrix."""
    return [weisfeiler_lehman_graph_hash(adj_to_nx(A)) for A in adjacency_matrices]


# ---------------------------------------------------------------------------
# Novelty / Uniqueness metrics
# ---------------------------------------------------------------------------

def compute_metrics(train_graphs, generated_adj):
    """
    Compute novelty, uniqueness, and novel+unique for a set of generated
    adjacency matrices, relative to a set of training PyG Data objects.

    Parameters
    ----------
    train_graphs   : list of PyG Data objects  (the training set)
    generated_adj  : list of adjacency matrices (numpy or tensor)

    Returns
    -------
    novelty, uniqueness, novelty_uniqueness : float  (fractions in [0, 1])
    """
    print("Computing Sample Evaluation Metrics")

    # Hash the training graphs
    train_adj    = [graph_to_adj(G) for G in train_graphs]
    train_hashes = set(compute_wl_hash(train_adj))

    # Hash the generated graphs
    generated_hashes = compute_wl_hash(generated_adj)

    # Novel: not isomorphic to any training graph
    novelty_mask = [h not in train_hashes for h in generated_hashes]
    novelty = float(np.mean(novelty_mask))

    # Unique: appears exactly once in the generated set
    counts = Counter(generated_hashes)
    uniqueness_mask = [counts[h] == 1 for h in generated_hashes]
    uniqueness = float(np.mean(uniqueness_mask))

    # Novel AND unique
    nu_mask = [
        h not in train_hashes and counts[h] == 1
        for h in generated_hashes
    ]
    novelty_uniqueness = float(np.mean(nu_mask))

    return novelty, uniqueness, novelty_uniqueness


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def compute_statistics(nx_graphs):
    """
    Compute per-node degree, clustering coefficient, and eigenvector centrality
    for a list of NetworkX graphs.

    Returns three flat lists of floats.
    """
    node_degrees      = []
    clustering_coeffs = []
    eigenvector_cent  = []

    for G in nx_graphs:
        if len(G) == 0:
            continue

        node_degrees.extend([d for _, d in G.degree()])
        clustering_coeffs.extend(list(nx.clustering(G).values()))

        try:
            eigenvector_cent.extend(
                list(nx.eigenvector_centrality_numpy(G).values())
            )
        except Exception:
            pass

    return node_degrees, clustering_coeffs, eigenvector_cent


def plot_statistics(train_nx, baseline_nx, generated_nx, save_path="results/graph_statistics.png"):
    """
    Plot a 3×3 grid of histograms comparing train / baseline / deep model.

    Each column corresponds to one statistic; each row to one model.
    Within each column the bins are shared so the histograms are directly
    comparable visually.
    """
    print("Plotting Graph Statistics")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_stats     = compute_statistics(train_nx)
    baseline_stats  = compute_statistics(baseline_nx)
    generated_stats = compute_statistics(generated_nx)

    stat_names = ["Node Degree", "Clustering Coefficient", "Eigenvector Centrality"]
    row_names  = ["Training", "Baseline (Erdős–Rényi)", "Deep Generative Model"]
    all_stats  = [train_stats, baseline_stats, generated_stats]

    n_bins = 20
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.suptitle("Graph Statistics Comparison", fontsize=14, fontweight="bold", y=1.01)

    for col, stat_name in enumerate(stat_names):
        # Gather all values for this statistic across all three distributions
        # so we can define shared bin edges.
        all_values = []
        for row in range(3):
            all_values.extend(all_stats[row][col])

        if len(all_values) == 0:
            continue

        v_min = min(all_values)
        v_max = max(all_values)

        # Avoid zero-width range (e.g. all degrees identical)
        if v_min == v_max:
            v_min -= 0.5
            v_max += 0.5

        shared_bins = np.linspace(v_min, v_max, n_bins + 1)

        for row, row_name in enumerate(row_names):
            ax = axes[row, col]
            data = all_stats[row][col]

            color = ["steelblue", "darkorange", "seagreen"][row]
            # Draw histogram with the shared bin edges and capture returned bin edges
            counts, bins, patches = ax.hist(data, bins=shared_bins, color=color,
                                alpha=0.75, edgecolor="white")

            # Place xticks at histogram bar centers (not edges)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            if len(bin_centers) > 0:
                # For degree column, show integer labels; otherwise show floats
                if stat_name == "Node Degree":
                    labels = [str(int(round(x))) for x in bin_centers]
                
                    if len(bin_centers) > 10:
                        sel = np.linspace(0, len(bin_centers) - 1, 10).astype(int)
                        tick_positions = bin_centers[sel]
                        tick_labels = [labels[i] for i in sel]
                    else:
                        tick_positions = bin_centers
                        tick_labels = labels

                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=0)
            ax.grid(axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(stat_name, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_name, fontsize=10)

            ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Graph statistics plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results(baseline_metrics, generated_metrics):
    print("\n=== Sample Evaluation Metrics ===\n")
    header = f"{'Model':<30} {'Novel (%)':<15} {'Unique (%)':<15} {'Novel+Unique (%)':<20}"
    print(header)
    print("-" * len(header))
    print(
        f"{'Baseline (Erdős–Rényi)':<30} "
        f"{baseline_metrics[0]*100:>8.2f}       "
        f"{baseline_metrics[1]*100:>8.2f}       "
        f"{baseline_metrics[2]*100:>8.2f}"
    )
    print(
        f"{'Deep Generative Model':<30} "
        f"{generated_metrics[0]*100:>8.2f}       "
        f"{generated_metrics[1]*100:>8.2f}       "
        f"{generated_metrics[2]*100:>8.2f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_SAMPLES = 1000
    device = (
        torch.device("cuda")  if torch.cuda.is_available()          else
        torch.device("mps")   if torch.backends.mps.is_available()  else
        torch.device("cpu")
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(42)
    train_ds, _, _ = random_split(dataset, [100, 44, 44], generator=rng)
    train_loader = DataLoader(train_ds, batch_size=100, shuffle=False)
    training_graphs = list(train_loader.dataset)   # PyG Data objects

    train_sizes = [data.num_nodes for data in training_graphs]

    # ------------------------------------------------------------------
    # Deep Generative Model (GraphVAE)
    # ------------------------------------------------------------------
    NODE_FEATURE_DIM = 7    # MUTAG
    STATE_DIM  = 16
    LATENT_DIM = 32
    NUM_ROUNDS = 3

    encoder   = GNNEncoder(NODE_FEATURE_DIM, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    #decoder   = InnerProductDecoder()
    decoder   = MLPDecoder(latent_dim=LATENT_DIM)
    prior     = GaussianPrior(LATENT_DIM)
    graph_vae = GraphVAE(encoder, decoder, prior).to(device)
    graph_vae.load_state_dict(torch.load("models/graph_vae.pt", map_location=device))
    graph_vae.eval()

    # ## TEST START ##
    # max_nodes = max([data.num_nodes for data in dataset])
    # in_dim = dataset.num_node_features
    # graph_vae = GraphVAE_Graph(in_dim, hidden_dim=64, latent_dim=32, max_nodes=max_nodes).to(device)
    # graph_vae.load_state_dict(torch.load("models/graph_graph_vae.pt", map_location=device))
    # graph_vae.eval()
    #size_dist = get_size_distribution(dataset)
    #adj_graph_vae = sample_graphs(graph_vae, N_SAMPLES, size_dist, max_nodes)

    # ## TEST END ##

    #print(f"Decoder bias: {graph_vae.decoder.bias.item():.4f}")

    # Sample N_SAMPLES adjacency matrices from the VAE
    adj_graph_vae = [
        graph_vae.sample(random.choice(train_sizes), device).detach().cpu().numpy()
        for _ in range(N_SAMPLES)
    ]
    
    # ------------------------------------------------------------------
    # Baseline: Erdős–Rényi
    # ------------------------------------------------------------------
    erdos_renyi = ErdosRenyi(dataset)
    adj_erdos_renyi = [
        erdos_renyi.sample().cpu().numpy()
        for _ in range(N_SAMPLES)
    ]

    # ------------------------------------------------------------------
    # Convert adjacency matrices to NetworkX for statistics
    # ------------------------------------------------------------------
    training_nx   = [adj_to_nx(graph_to_adj(d)) for d in training_graphs]
    baseline_nx   = [adj_to_nx(A) for A in adj_erdos_renyi]
    generated_nx  = [adj_to_nx(A) for A in adj_graph_vae]

    # ------------------------------------------------------------------
    # Novelty / Uniqueness metrics
    # ------------------------------------------------------------------
    # Note: baseline_metrics compares Erdős–Rényi samples to training set
    #       generated_metrics compares VAE samples to training set
    baseline_metrics  = compute_metrics(training_graphs, adj_erdos_renyi)
    generated_metrics = compute_metrics(training_graphs, adj_graph_vae)

    print_results(baseline_metrics, generated_metrics)

    # ------------------------------------------------------------------
    # Graph statistics (3×3 histogram grid)
    # ------------------------------------------------------------------
    plot_statistics(training_nx, baseline_nx, generated_nx)

        # draw graphs
    plot_graphs(training_nx, "Training Graphs")
    plot_graphs(baseline_nx, "Baseline Graphs")
    plot_graphs(generated_nx, "Generated Graphs")



if __name__ == "__main__":
    main()