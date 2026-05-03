import torch
import torch.nn as nn
import torch.distributions as td

class GNNEncoder(nn.Module):
    """Message-passing GNN that maps (X, A) -> (mu_Z, log_sigma_Z) per node.

    Parameters
    ----------
    node_feature_dim : int
    state_dim        : int  – hidden dimension
    latent_dim       : int  – per-node latent dimension M
    num_rounds       : int  – message passing rounds
    """

    def __init__(self, node_feature_dim: int, state_dim: int,
                 latent_dim: int, num_rounds: int = 3):
        super().__init__()
        self.num_rounds = num_rounds

        # Map raw features -> initial hidden state
        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU(),
        )

        # One message network per round
        self.message_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.ReLU(),
            ) for _ in range(num_rounds)
        ])

        # GRU-based node-state update 
        self.update_cell = nn.GRUCell(state_dim, state_dim)

        # Project final state -> (mu, log_sigma)
        self.mu_net      = nn.Linear(state_dim, latent_dim)
        self.log_sig_net = nn.Linear(state_dim, latent_dim)

    def forward(self, x, edge_index):
        """
        Parameters
        ----------
        x          : (N, node_feature_dim)
        edge_index : (2, E)  – [source, target]

        Returns
        -------
        mu       : (N, M)
        log_sigma: (N, M)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        state = self.input_net(x)                          # (N, state_dim)

        for r in range(self.num_rounds):
            # Compute messages from each source node
            msg = self.message_nets[r](state)              # (N, state_dim)

            # Aggregate messages at each destination node (sum)
            agg = x.new_zeros(num_nodes, state.size(1))
            agg = agg.index_add(0, dst, msg[src])          # (N, state_dim)

            # GRU update
            state = self.update_cell(agg, state)           # (N, state_dim)

        mu        = self.mu_net(state)                     # (N, M)
        log_sigma = self.log_sig_net(state)                # (N, M)

        sigma = torch.exp(log_sigma.clamp(-4, 4))  # numerical safety
        return td.Independent(td.Normal(mu, sigma), 1)

class InnerProductDecoder(nn.Module):
    """P(A_uv = 1) = sigma( z_u^T z_v + b )

    Only evaluates the pairs supplied via `edge_pairs`.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z, edge_pairs):
        """
        Parameters
        ----------
        z          : (N, M)  – sampled node embeddings
        edge_pairs : (2, K)  – pairs of nodes to score (both positive & negative)

        Returns
        -------
        logits : (K,)
        """
        u, v = edge_pairs[0], edge_pairs[1]
        logits = (z[u] * z[v]).sum(dim=-1) + self.bias
        return logits


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std",  torch.ones(latent_dim))

    def forward(self, num_nodes: int):
        """Return a batched prior of shape (num_nodes, M)."""
        mean = self.mean.unsqueeze(0).expand(num_nodes, -1)
        std  = self.std.unsqueeze(0).expand(num_nodes, -1)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)

class GraphVAE(nn.Module):
    """VAE with node-level latents for graph generation.

    The ELBO objective is:

        L = E_q[ log p(A | Z) ] - KL( q(Z|X,A) || p(Z) )

    where:
      - q(Z|X,A) = prod_i  N(z_i | mu_i, diag(sigma_i^2))   (GNN encoder)
      - p(A|Z)   = prod_{u,v} Bernoulli(sigma(z_u^T z_v + b)) (inner product decoder)
      - p(Z)     = prod_i  N(0, I)                             (standard Gaussian)
    """

    def __init__(self, encoder, decoder, prior):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior   = prior

    def elbo(self, x, edge_index, num_nodes: int, n_samples: int = 3,
             kl_beta: float = 1.0):
        """Compute a single-graph ELBO estimate.

        Uses the *full* N×N adjacency so every pair is supervised — no random
        subsampling noise.  A pos_weight corrects for the class imbalance
        (sparse graphs have far more non-edges than edges).

        Parameters
        ----------
        kl_beta : float in [0,1]
            Annealing coefficient for the KL term (beta-VAE style).
            Pass values ramping from 0 → 1 during training to prevent
            posterior collapse in early epochs.
        """
        device = x.device

        q = self.encoder(x, edge_index)               # dist over (N, M)
        p = self.prior(num_nodes)                      # dist over (N, M)

        kl = td.kl_divergence(q, p).sum()             # scalar

        # ---- Build full adjacency label matrix ----------------------------
        # All N*N pairs
        idx        = torch.arange(num_nodes, device=device)
        u, v       = torch.meshgrid(idx, idx, indexing="ij")
        all_pairs  = torch.stack([u.reshape(-1), v.reshape(-1)])  # (2, N^2)

        # Binary labels: 1 if edge exists
        adj_dense  = torch.zeros(num_nodes, num_nodes, device=device)
        adj_dense[edge_index[0], edge_index[1]] = 1.0
        labels     = adj_dense.reshape(-1)                         # (N^2,)

        # pos_weight = #negatives / #positives  (handles class imbalance)
        num_pos    = labels.sum().clamp(min=1)
        num_neg    = (num_nodes * num_nodes) - num_pos
        pos_weight = (num_neg / num_pos).detach()

        # ---- Monte-Carlo reconstruction -----------------------------------
        z_samples = q.rsample((n_samples,))                   # (S, N, M)
        logits = torch.stack([
            self.decoder(z_samples[s], all_pairs) for s in range(n_samples)
        ])
        # recon_sum = 0.0
        # for _ in range(n_samples):
        #     z      = q.rsample()                               # (N, M)
        #     logits = self.decoder(z, all_pairs)                # (N^2,)
        #     bce    = nn.functional.binary_cross_entropy_with_logits(
        #         logits, labels,
        #         pos_weight=pos_weight,
        #         reduction="sum",
        #     )
        #     recon_sum += -bce

        # recon = recon_sum / n_samples
        recon = -torch.stack([
            nn.functional.binary_cross_entropy_with_logits(
                logits[s], labels, pos_weight=pos_weight, reduction="sum"
            ) for s in range(n_samples)
        ]).mean()

        return recon - kl_beta * kl

    def forward(self, x, edge_index, num_nodes: int, kl_beta: float = 1.0):
        return -self.elbo(x, edge_index, num_nodes, kl_beta=kl_beta)

    @torch.no_grad()
    def reconstruct_adj(self, x, edge_index, threshold: float = 0.5):
        """Return the reconstructed adjacency matrix for a single graph."""
        q      = self.encoder(x, edge_index)
        z      = q.mean                               # use mean for reconstruction
        N      = z.size(0)
        # all pairs
        idx    = torch.arange(N, device=z.device)
        u, v   = torch.meshgrid(idx, idx, indexing="ij")
        pairs  = torch.stack([u.reshape(-1), v.reshape(-1)])
        logits = self.decoder(z, pairs).reshape(N, N)
        probs  = torch.sigmoid(logits)
        return probs

    @torch.no_grad()
    def sample(self, num_nodes: int, device):
        """Generate a brand new graph by sampling from the prior."""
        z = torch.randn(num_nodes, self.prior.latent_dim, device=device)  # p(Z)
        idx = torch.arange(num_nodes, device=device)
        u, v = torch.meshgrid(idx, idx, indexing="ij") # ALL node pairs (N^2)
        pairs = torch.stack([u.reshape(-1), v.reshape(-1)]) # Same but edge list format (2, N^2)
        probs = torch.sigmoid(self.decoder(z, pairs)).reshape(num_nodes, num_nodes)
        adj = torch.bernoulli(probs)  # actually sample edges
        return adj
