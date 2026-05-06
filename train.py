import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import GNNEncoder, InnerProductDecoder, GaussianPrior, GraphVAE, MLPDecoder


# ---------------------------------------------------------------------------
# Epoch helpers — free_bits must be identical in train and eval so the two
# losses are on the same scale and can be meaningfully compared.
# ---------------------------------------------------------------------------

FREE_BITS = 0.0   # nats per latent dimension (graph-size invariant)


def train_epoch(model, loader, optimizer, device, kl_beta: float = 1.0):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_mu = 0.0
    total_sigma = 0.0
    #total_bias = 0.0
    count = 0

    for data in loader:
        data       = data.to(device)
        loss       = torch.tensor(0.0, device=device)
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
            out = model(x_g, ei_g_local, x_g.size(0),
                          kl_beta=kl_beta, free_bits=FREE_BITS, return_parts=True)
            loss += out["loss"]
            total_recon += out["recon"].item()
            total_kl += out["kl"].item()
            total_mu += out["mu_abs"].item()
            total_sigma += out["sigma"].item()
            #total_bias += out["bias"].item()
            count += 1

        loss = loss / num_graphs
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    stats = {
        "loss": total_loss / len(loader),
        "recon": total_recon / count,
        "kl": total_kl / count,
        "mu_abs": total_mu / count,
        "sigma": total_sigma / count,
        #"bias": total_bias / count,
    }
    #return total_loss / len(loader), 
    return stats


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for data in loader:
        data       = data.to(device)
        loss       = torch.tensor(0.0, device=device)
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
            # kl_beta=1 for eval — we always want the true ELBO on val/test
            loss += model(x_g, ei_g_local, x_g.size(0),
                          kl_beta=1.0, free_bits=FREE_BITS)

        total_loss += (loss / num_graphs).item()
    return total_loss / len(loader)


def estimate_bias_init(dataset):
    """Log-odds of the average edge density — used to initialise decoder bias."""
    densities = []
    for data in dataset:
        n, e = data.num_nodes, data.num_edges
        if n > 1:
            densities.append(e / (n * n))
    p = float(np.clip(np.mean(densities), 1e-4, 1 - 1e-4))
    log_odds = float(np.log(p / (1 - p)))
    print(f"Training edge density: {p:.4f}  →  decoder bias init: {log_odds:.4f}")
    return log_odds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = (
        torch.device("cuda")  if torch.cuda.is_available()         else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    dataset = TUDataset(root="./data/", name="MUTAG")
    rng     = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [100, 44, 44], generator=rng)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)
    test_loader  = DataLoader(test_ds,  batch_size=16)

    node_feature_dim = dataset.num_node_features   # 7 for MUTAG

    STATE_DIM    = 16
    LATENT_DIM   = 32
    NUM_ROUNDS   = 3
    EPOCHS       = 100
    LR           = 5e-4    # lower LR for smoother convergence
    WEIGHT_DECAY = 1e-4
    # Ramp β over 80% of training — slow enough that reconstruction has time
    # to learn structure before the KL penalty compresses the posterior.
    ANNEAL_EPOCHS = int(EPOCHS * 0.8)
    MAX_BETA = 1e-1


    bias_init = estimate_bias_init(train_ds)

    encoder = GNNEncoder(node_feature_dim, STATE_DIM, LATENT_DIM, NUM_ROUNDS)
    decoder = InnerProductDecoder(init_bias=bias_init)
    decoder = MLPDecoder(latent_dim=LATENT_DIM)
    prior   = GaussianPrior(LATENT_DIM)
    model   = GraphVAE(encoder, decoder, prior).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_losses, val_losses, betas = [], [], []


    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
        kl_beta = MAX_BETA * min(1.0, epoch / ANNEAL_EPOCHS)
        tr_stats  = train_epoch(model, train_loader, optimizer, device, kl_beta=kl_beta)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(tr_stats["loss"])
        val_losses.append(val_loss)
        betas.append(kl_beta)

        if epoch % 10 == 0:
            #print(f"Epoch {epoch:3d}  |  β={kl_beta:.2f}  "
            #      f"|  train={tr_stats['loss']:.4f}  |  val={val_loss:.4f}")
            print(
                f"Epoch {epoch:3d} | "
                f"β={kl_beta:.3f} | "
                f"loss={tr_stats['loss']:.4f} | "
                f"recon={tr_stats['recon']:.4f} | "
                f"kl={tr_stats['kl']:.4f} | "
                f"|μ|={tr_stats['mu_abs']:.4f} | "
                f"σ={tr_stats['sigma']:.4f} | "
                #f"bias={tr_stats['bias']:.4f}"
            )

    test_loss = eval_epoch(model, test_loader, device)
    print(f"\nTest loss (neg ELBO): {test_loss:.4f}")

    torch.save(model.state_dict(), "models/graph_vae.pt")
    print("Saved → models/graph_vae.pt")

    # ---- Learning curve --------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(train_losses, label="Train loss", color="steelblue")
    ax1.plot(val_losses,   label="Val loss",   color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Neg ELBO (per graph, avg)")
    ax1.set_title("Graph VAE – Learning Curves")
    ax2 = ax1.twinx()
    ax2.plot(betas, label="KL β", color="grey", linestyle="--", alpha=0.6)
    ax2.set_ylabel("KL annealing β", color="grey")
    ax2.set_ylim(0, 1.2)
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig("results/graph_vae_loss.png", dpi=150)
    plt.show()
    print("Loss curve → results/graph_vae_loss.png")