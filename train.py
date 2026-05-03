import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import GNNEncoder, InnerProductDecoder, GaussianPrior, GraphVAE

def train_epoch(model, loader, optimizer, device, kl_beta: float = 1.0):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        # Process each graph in the batch independently
        loss = torch.tensor(0.0, device=device)
        num_graphs = int(data.batch.max().item()) + 1
        for g in range(num_graphs):
            mask       = data.batch == g
            x_g        = data.x[mask]
            # Remap edge indices to local node indices
            global_idx = mask.nonzero(as_tuple=True)[0]
            # Keep only edges within this graph
            e_mask     = mask[data.edge_index[0]] & mask[data.edge_index[1]]
            ei_g       = data.edge_index[:, e_mask]
            # Remap to local
            remap      = torch.full((mask.size(0),), -1, dtype=torch.long, device=device)
            remap[global_idx] = torch.arange(global_idx.size(0), device=device)
            ei_g_local = remap[ei_g]
            loss += model(x_g, ei_g_local, x_g.size(0), kl_beta=kl_beta)

        loss = loss / num_graphs
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        loss = torch.tensor(0.0, device=device)
        num_graphs = int(data.batch.max().item()) + 1
        for g in range(num_graphs):
            mask       = data.batch == g
            x_g        = data.x[mask]
            global_idx = mask.nonzero(as_tuple=True)[0]
            e_mask     = mask[data.edge_index[0]] & mask[data.edge_index[1]]
            ei_g       = data.edge_index[:, e_mask]
            remap      = torch.full((mask.size(0),), -1, dtype=torch.long, device=device)
            remap[global_idx] = torch.arange(global_idx.size(0), device=device)
            ei_g_local = remap[ei_g]
            loss += model(x_g, ei_g_local, x_g.size(0))
        total_loss += (loss / num_graphs).item()
    return total_loss / len(loader)

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [100, 44, 44], generator=rng)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)
    test_loader  = DataLoader(test_ds,  batch_size=16)

    node_feature_dim = dataset.num_node_features  # 7 for MUTAG

    STATE_DIM   = 128   # wider hidden state (was 64)
    LATENT_DIM  = 32    # larger per-node latent (was 16)
    NUM_ROUNDS  = 5     # more message passing (was 3)
    EPOCHS      = 200   # longer training (was 100)
    LR          = 1e-3
    # KL annealing: beta ramps linearly from 0 -> 1 over the first ANNEAL_EPOCHS
    # This lets the decoder learn a useful signal before the KL penalty is enforced
    ANNEAL_EPOCHS = int(EPOCHS * 0.5)  # first 50% of training

    #gnn_enc = GNNEncoder(node_feature_dim, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    #encoder = GraphEncoder(gnn_enc)
    encoder = GNNEncoder(node_feature_dim, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    decoder = InnerProductDecoder()
    prior   = GaussianPrior(LATENT_DIM)
    model   = GraphVAE(encoder, decoder, prior).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-5)

    # ---- Training loop ----------------------------------------------------
    train_losses = []
    val_losses   = []
    betas        = []

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
        # Linear KL annealing: 0 -> 1 over first ANNEAL_EPOCHS epochs
        kl_beta  = min(1.0, epoch / ANNEAL_EPOCHS)
        tr_loss  = train_epoch(model, train_loader, optimizer, device, kl_beta=kl_beta)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        betas.append(kl_beta)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}  |  beta: {kl_beta:.2f}  |  train loss: {tr_loss:.4f}  |  val loss: {val_loss:.4f}")

    # Test eval
    test_loss = eval_epoch(model, test_loader, device)
    print(f"\nTest loss (negative ELBO): {test_loss:.4f}")

    torch.save(model.state_dict(), "models/graph_vae.pt")
    print("Model saved to models/graph_vae.pt")

    # ---- Plot learning curves + beta schedule ----------------------------
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(train_losses, label="Train loss", color="steelblue")
    ax1.plot(val_losses,   label="Val loss",   color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Negative ELBO (per graph, avg)")
    ax1.set_title("Graph VAE – Learning Curves")
    ax2 = ax1.twinx()
    ax2.plot(betas, label="KL β", color="grey", linestyle="--", alpha=0.6)
    ax2.set_ylabel("KL annealing β", color="grey")
    ax2.set_ylim(0, 1.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig("results/graph_vae_loss.png", dpi=150)
    plt.show()
    print("Loss curve saved to results/graph_vae_loss.png")    