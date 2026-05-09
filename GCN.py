
from torch.autograd import Variable
import torch
import torch.nn as nn

import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GlobalAttention, GATConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from tqdm import tqdm
# create a Graph Convolutional Neural Network
# using PyTorch

GRAPH_SIZE = 15
ATT_HEADS = 6

class GraphConv(torch.nn.Module):
    def __init__(self, input_size, learning, have_first=True, att_heads=6, optimizer=True, gat=True, atom_amount=15):
        super(GraphConv, self).__init__()

        self.have_first = have_first
        ATT_HEADS = att_heads
        GRAPH_SIZE = atom_amount

        if gat:
            if have_first:
                self.GCN1 = GATConv(input_size, input_size, heads=ATT_HEADS, concat=True)
                self.GCN2 = GATConv(input_size * ATT_HEADS, input_size, heads=int(ATT_HEADS/2), concat=True)
                self.linear = torch.nn.Linear(input_size * GRAPH_SIZE * int(ATT_HEADS/2), input_size * int(GRAPH_SIZE/2))
            else:
                self.GCN = GATConv(input_size, input_size, heads=int(ATT_HEADS), concat=True)
                self.linear = torch.nn.Linear(input_size * GRAPH_SIZE * int(ATT_HEADS), input_size * int(GRAPH_SIZE/2))
            # After GCN layers, each graph is flattened to a vector of size input_size * GRAPH_SIZE * int(ATT_HEADS/2)
        else:
            if have_first:
                self.GCN1 = GCNConv(input_size, input_size)
                self.GCN2 = GCNConv(input_size, input_size)
                self.linear = torch.nn.Linear(input_size * GRAPH_SIZE, input_size * int(GRAPH_SIZE/2))
            else:
                self.GCN = GCNConv(input_size, input_size)
                self.linear = torch.nn.Linear(input_size * GRAPH_SIZE, input_size * int(GRAPH_SIZE/2))

        self.linear2 = torch.nn.Linear(input_size * int(GRAPH_SIZE/2), input_size)
        self.linear3 = torch.nn.Linear(input_size, 1)

        self.loss = nn.MSELoss()
        if optimizer:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning, weight_decay=0)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning, weight_decay=0)

        self.test = torch.nn.Linear(1,1)

        self.lossHistory = []
        self.valLossHistory = []

    def forward(self, inputFeatures, edge_index, edge_weight, batch):
        
        # #print('a', inputFeatures[0])
        if self.have_first:

            x = self.GCN1(inputFeatures, edge_index, edge_weight)
            x = F.relu(x)

            x = self.GCN2(x, edge_index, edge_weight)
            x = F.relu(x)

        else:
            x = self.GCN(inputFeatures, edge_index, edge_weight)
            x = F.relu(x)


        # Flatten each graph in the batch into a single vector
        # This will create a tensor of shape (num_graphs, num_nodes*num_features_per_graph)
        graph_vecs = []
        for i in batch.unique():
            mask = (batch == i)
            graph_x = x[mask].reshape(-1)
            graph_vecs.append(graph_x)
        
        x = torch.stack(graph_vecs, dim=0)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
 
        return x

    def trainModel(self, data_train, data_val, epochs, verbose=True):
        torch.manual_seed(10)
        for epoch in range(epochs):
            epoch_loss = 0
            self.train()  # set to train mode
            with tqdm(enumerate(data_train), total=len(data_train), desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose) as pbar:
                
                for i, data in pbar:
                    inputFeatures = data.x
                    edge_index = data.edge_index
                    edge_weight = data.edge_weight
                    outputFeature = data.y
                    batch = data.batch if hasattr(data, 'batch') else None
                    output = self.forward(inputFeatures, edge_index, edge_weight, batch)
                    loss = self.loss(output.squeeze(), outputFeature)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                epoch_loss /= len(data_train)
                self.lossHistory.append(epoch_loss)
            val_loss = 0
            self.eval()  # set to eval mode
            with torch.no_grad():
                for data in data_val:
                    inputFeatures = data.x
                    edge_index = data.edge_index
                    edge_weight = data.edge_weight
                    outputFeature = data.y
                    batch = data.batch if hasattr(data, 'batch') else None
                    output = self.forward(inputFeatures, edge_index, edge_weight, batch)
                    loss = self.loss(output.squeeze(), outputFeature)
                    val_loss += loss.item()
            val_loss /= len(data_val)
            self.valLossHistory.append(val_loss)
            if verbose:
                print(f'Epoch {epoch+1} Loss --> {epoch_loss:.4f} | Val Loss --> {val_loss:.4f}')


class InterpreterDataset(Dataset):
    def __init__(self, inputFeaturesList, adjacencyMatrixList, outputFeatureList, atom_amount=GRAPH_SIZE, transform=None, ):
        assert len(inputFeaturesList) == len(outputFeatureList), "Input and output feature lists must be of the same length"
        assert len(inputFeaturesList) == len(adjacencyMatrixList), "Input feature and adjacency matrix lists must be of the same length"
        self.inputFeaturesList = inputFeaturesList
        self.atom_amount = atom_amount
        if transform:
            self.adjacencyMatrixList = [self.transform(adjacencyMatrix) for adjacencyMatrix in adjacencyMatrixList]
        else:
            self.adjacencyMatrixList = adjacencyMatrixList
        self.outputFeatureList = outputFeatureList

    def __len__(self):
        return len(self.inputFeaturesList)

    def __getitem__(self, idx):
        # Convert dense adjacency matrix to edge index and edge_weight (distance)
        edge_index, edge_weight = dense_to_sparse(self.adjacencyMatrixList[idx, :self.atom_amount, :self.atom_amount])
        data = Data(x=self.inputFeaturesList[idx, :self.atom_amount], edge_index=edge_index, edge_weight=edge_weight.t().contiguous(), y=self.outputFeatureList[idx])
        return data
