import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class ErdosRenyi:
    def __init__(self, dataset):
        self.NDistribution = []

        #Maybe ugly as shit but since the dataset is so small i dont care if i loop multiple times
        for data in dataset:
            N = data.num_nodes
            self.NDistribution.append(N)
        maxN = max(self.NDistribution)+1
        maxEdges = [0]*(maxN)
        numEdges = [0]*(maxN)
        for data in dataset:
            N = data.num_nodes
            maxEdges[N] += N * (N - 1) / 2
            numEdges[N] += data.num_edges / 2
        self.r = [0]*(maxN)
        for i in range(maxN):
            if maxEdges[i] > 0:
                self.r[i] = numEdges[i]/maxEdges[i]
        
        self.NDistribution = torch.tensor(self.NDistribution)
        self.r = torch.tensor(self.r)
        
    def sample(self, K=1):
        #This implementation is also slow but takes like 13 sec for 1000 graphs so it is fine
        samples = []
        for _ in range(K):
            idx = torch.randint(0,self.NDistribution.size(dim=0), (1,))
            N = self.NDistribution[idx]
            r = self.r[N]

            #My hack is to generate a random matrix, (only care about upper triangular part) and then for each number smaller than r it becomes a connection, and then mirror it down to become symmetrical
            randMatrix = torch.rand((N,N))
            adjMatrix = (randMatrix < r).int()
            adjMatrix = torch.triu(adjMatrix, diagonal=1)
            adjMatrix = adjMatrix + adjMatrix.T

            samples.append(adjMatrix)

        return samples

device = 'cpu'
samples = 1000

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = dataset.num_node_features

# Create model
erdosRenyi = ErdosRenyi(dataset)
eRSamples = erdosRenyi.sample(samples)