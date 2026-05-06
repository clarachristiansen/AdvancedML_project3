import torch
import torch.nn as nn
import torch.distributions as td
from torch_geometric.utils import negative_sampling
import networkx as nx

class GNNEncoder(nn.Module):
    """Message-passing GNN: (X, A) -> q(Z | X, A) per node."""

    def __init__(self, node_feature_dim: int, state_dim: int,
                 latent_dim: int, num_rounds: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_rounds = num_rounds

        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.message_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(num_rounds)
        ])
        self.update_cell  = nn.GRUCell(state_dim, state_dim)
        self.mu_net       = nn.Linear(state_dim, latent_dim)
        self.log_sig_net  = nn.Linear(state_dim, latent_dim)

    def forward(self, x, edge_index):
        src, dst  = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        state     = self.input_net(x)

        for r in range(self.num_rounds):
            msg   = self.message_nets[r](state)
            agg   = x.new_zeros(num_nodes, state.size(1))
            agg   = agg.index_add(0, dst, msg[src])
            state = self.update_cell(agg, state)

        mu        = self.mu_net(state)
        log_sigma = self.log_sig_net(state)
        sigma     = torch.exp(log_sigma.clamp(-4, 4))
        return td.Independent(td.Normal(mu, sigma), 1)


class InnerProductDecoder(nn.Module):
    """P(A_uv = 1) = σ( z_u · z_v + b ), with L2-normalised embeddings.

    L2 normalisation keeps the dot product in [-1, 1] regardless of
    embedding magnitude, so the bias alone controls the baseline edge
    probability at sample time (when z ~ N(0,I)).
    """

    def __init__(self, init_bias: float = -2.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([init_bias]))

    def forward(self, z, edge_pairs):
        #z = nn.functional.normalize(z, p=2, dim=-1)
        u, v = edge_pairs[0], edge_pairs[1]
        return (z[u] * z[v]).sum(dim=-1) + self.bias


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z, edge_pairs):
        u, v = edge_pairs
        x = torch.cat([z[u], z[v]], dim=-1)
        return self.net(x).squeeze(-1)

class GaussianPrior(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std",  torch.ones(latent_dim))

    def forward(self, num_nodes: int):
        mean = self.mean.unsqueeze(0).expand(num_nodes, -1)
        std  = self.std.unsqueeze(0).expand(num_nodes, -1)
        return td.Independent(td.Normal(mean, std), 1)


class GraphVAE(nn.Module):
    """
    Graph VAE.  ELBO (normalised by N²):

        L = E_q[log p(A|Z)] / N²  -  β · KL_fb(q || p) / N²

    Free-bits is applied per latent *dimension* after averaging the KL
    over nodes.  This makes the threshold graph-size invariant: each of
    the `latent_dim` dimensions is allowed to keep `free_bits` nats
    regardless of how many nodes the graph has.
    """

    def __init__(self, encoder, decoder, prior):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior   = prior

    def elbo(self, x, edge_index, num_nodes: int, n_samples: int = 3,
             kl_beta: float = 1.0, free_bits: float = 0.5, return_parts: bool = False):
        """
        Parameters
        ----------
        kl_beta   : KL annealing weight, ramped 0 → 1 during training.
        free_bits : KL floor in nats *per latent dimension* (after averaging
                    over nodes).  0.5 nats/dim is a good default — large
                    enough to prevent full collapse but small enough that
                    the encoder must still encode the graph to beat it.
        """
        device = x.device
        N2     = num_nodes * num_nodes

        q = self.encoder(x, edge_index)
        p = self.prior(num_nodes)

        # KL per node per dim: (N, latent_dim)
        kl_nd = td.kl_divergence(
            td.Normal(q.base_dist.loc, q.base_dist.scale),
            td.Normal(p.base_dist.loc, p.base_dist.scale),
        )
        # Average over nodes → (latent_dim,).
        # Apply free-bits floor per dimension, then sum and normalise by N².
        # Averaging (not summing) over nodes makes the floor graph-size
        # invariant: a 10-node and a 20-node graph both contribute the same
        # per-dimension pressure.
        kl_per_dim = kl_nd.mean(dim=0)                     # (latent_dim,)
        kl = kl_per_dim.clamp(min=free_bits).sum() # / N2    # scalar, O(1/N²)

        # ---- Full N×N adjacency supervision ------------------------------
        idx       = torch.arange(num_nodes, device=device)
        u, v      = torch.meshgrid(idx, idx, indexing="ij")
        all_pairs = torch.stack([u.reshape(-1), v.reshape(-1)])  # (2, N²)

        adj_dense = torch.zeros(num_nodes, num_nodes, device=device)
        adj_dense[edge_index[0], edge_index[1]] = 1.0
        labels    = adj_dense.reshape(-1)                        # (N²,)

        triangle_w = 5.0  # extra weight for non-edges that would close triangles
        A2 = adj_dense @ adj_dense                      # (N, N)
        has_common  = (A2 > 0).float().reshape(-1)               # (N²,)
        # Only penalise non-edges that would close triangles
        tri_penalty = has_common * (1.0 - labels)                # 1 iff non-edge & triangle
        pair_weight = 1.0 + triangle_w * tri_penalty             # (N²,)

        # ---- Monte-Carlo reconstruction (no pos_weight) ------------------
        z_samples = q.rsample((n_samples,))                      # (S, N, M)
        recon = -torch.stack([
            nn.functional.binary_cross_entropy_with_logits(
                self.decoder(z_samples[s], all_pairs),
                labels,
                weight=pair_weight,
                reduction="sum",
            ) / N2
            for s in range(n_samples)
        ]).mean()
        # print(f"Recon: {recon.item():.4f}, KL: {kl.item():.4f}")

        elbo = recon - kl * kl_beta

        if return_parts:
            with torch.no_grad():
                mu_mean = q.base_dist.loc.abs().mean()
                sigma_mean = q.base_dist.scale.mean()
                #bias = self.decoder.bias.mean()

            return {
                "elbo": elbo,
                "recon": recon.detach(),
                "kl": kl.detach(),
                "mu_abs": mu_mean.detach(),
                "sigma": sigma_mean.detach(),
                #"bias": bias.detach(),
            }
        return elbo


    def forward(self, x, edge_index, num_nodes: int,
                kl_beta: float = 1.0, free_bits: float = 0.5, return_parts: bool = False):
        out = self.elbo(x, edge_index, num_nodes,
                          kl_beta=kl_beta, free_bits=free_bits, return_parts=return_parts)
        if return_parts:
            out["loss"] = -out["elbo"]
            return out
        return -out

    @torch.no_grad()
    def reconstruct_adj(self, x, edge_index):
        q     = self.encoder(x, edge_index)
        z     = q.mean
        N     = z.size(0)
        idx   = torch.arange(N, device=z.device)
        u, v  = torch.meshgrid(idx, idx, indexing="ij")
        pairs = torch.stack([u.reshape(-1), v.reshape(-1)])
        return torch.sigmoid(self.decoder(z, pairs)).reshape(N, N)

    # @torch.no_grad()
    # def sample(self, num_nodes: int, device):
    #     z     = torch.randn(num_nodes, self.prior.latent_dim, device=device)
    #     idx   = torch.arange(num_nodes, device=device)
    #     u, v  = torch.meshgrid(idx, idx, indexing="ij")
    #     pairs = torch.stack([u.reshape(-1), v.reshape(-1)])
    #     probs = torch.sigmoid(self.decoder(z, pairs)).reshape(num_nodes, num_nodes)
    #     return torch.bernoulli(probs)
    @torch.no_grad()
    def sample(self, num_nodes: int, device):
        num_nodes = torch.randint(10, num_nodes, (1,)).item()  # Sample a random number of nodes up to num_nodes
        z     = torch.randn(num_nodes, self.prior.latent_dim, device=device)
        idx   = torch.arange(num_nodes, device=device)
        u, v  = torch.meshgrid(idx, idx, indexing="ij")
        pairs = torch.stack([u.reshape(-1), v.reshape(-1)])

        probs = torch.sigmoid(self.decoder(z, pairs)).reshape(num_nodes, num_nodes)

        # ---- REMOVE SELF-LOOPS ----
        probs.fill_diagonal_(0)

        # ---- SYMMETRIZE ----
        probs = (probs + probs.T) / 2

        # Sample
        A = torch.bernoulli(probs)

        # # Convert to CPU numpy
        # A_np = A.cpu().float().numpy()

        # # Build graph
        # G = nx.from_numpy_array(A_np)

        # # Keep largest connected component
        # largest_cc = max(nx.connected_components(G), key=len)

        # G = G.subgraph(largest_cc).copy()

        # # Convert back to adjacency matrix
        # A_lcc = nx.to_numpy_array(G)

        # A = torch.tensor(A_lcc, device=device, dtype=torch.float32)

        return A