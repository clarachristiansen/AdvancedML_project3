import torch
import matplotlib.pyplot as plt
from model import GNNEncoder, InnerProductDecoder, GaussianPrior, GraphVAE
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import time


def plot_reconstruction(model, test_loader, device):
    model.eval()
    sample_data = next(iter(test_loader))
    sample_data = sample_data.to(device)

    g = 1  # first graph in batch
    mask       = sample_data.batch == g
    x_g        = sample_data.x[mask]
    global_idx = mask.nonzero(as_tuple=True)[0]
    e_mask     = mask[sample_data.edge_index[0]] & mask[sample_data.edge_index[1]]
    ei_g       = sample_data.edge_index[:, e_mask]
    remap      = torch.full((mask.size(0),), -1, dtype=torch.long, device=device)
    remap[global_idx] = torch.arange(global_idx.size(0), device=device)
    ei_g_local = remap[ei_g]

    probs = model.reconstruct_adj(x_g, ei_g_local).cpu()
    N     = x_g.size(0)

    # True adjacency
    true_adj = torch.zeros(N, N)
    true_adj[ei_g_local[0], ei_g_local[1]] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(true_adj.numpy(),   cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("True adjacency")
    axes[1].imshow(probs.numpy(),      cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Reconstructed P(A_uv=1)")
    for ax in axes:
        ax.set_xlabel("node v")
        ax.set_ylabel("node u")
    plt.suptitle(f"Graph VAE – reconstruction (graph 0, {N} nodes)")
    plt.tight_layout()
    plt.savefig("results/graph_vae_reconstruction.png", dpi=150)
    plt.show()
    print("Reconstruction plot saved to results/graph_vae_reconstruction.png")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(dataset, [100, 44, 44], generator=rng)
    test_loader  = DataLoader(test_ds,  batch_size=16)

    node_feature_dim = dataset.num_node_features  # 7 for MUTAG
    # OBS HAS TO MATCH TRAINING CONFIGURATION EXACTLY TO LOAD MODEL WEIGHTS
    STATE_DIM   = 128   
    LATENT_DIM  = 32    
    NUM_ROUNDS  = 5

    #gnn_enc = GNNEncoder(node_feature_dim, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    #encoder = GraphEncoder(gnn_enc)
    encoder = GNNEncoder(node_feature_dim, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    decoder = InnerProductDecoder()
    prior   = GaussianPrior(LATENT_DIM)
    model   = GraphVAE(encoder, decoder, prior).to(device)
    model.load_state_dict(torch.load("models/graph_vae.pt", map_location=device))

    plot_reconstruction(model, test_loader, device)


    start_time = time.time()
    graphs = [model.sample(20, device) for _ in range(1000)]
    end_time = time.time()

    print(f"Sampling took {end_time - start_time:.4f} seconds")
