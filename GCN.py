
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


class GraphConv(torch.nn.Module):
    def __init__(self, input_size, lr):
        super(GraphConv, self).__init__()

        ATOM_AMOUNT = 118
        self.embedding = nn.Embedding(ATOM_AMOUNT, input_size)

        self.GCN = GATConv(input_size, input_size)

        # After pooling, the input size is 500 (output of last GCN layer)
        self.linear = torch.nn.Linear(input_size, 10)
        self.linear2 = torch.nn.Linear(10, 1)

        self.test = torch.nn.Linear(1, 1)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0)

        gate_nn = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()  # attention scores in [0,1]
        )
        self.att_pool = GlobalAttention(gate_nn)

        self.lossHistory = []
        self.valLossHistory = []

    def forward(self, inputFeatures, edge_index, edge_weight, batch):
        
        #print('a', inputFeatures[0])
        x = self.GCN(inputFeatures, edge_index)
        #x = F.leaky_relu(x)

        #print('b', x[0])
        
        for n in range(2):
            x = self.GCN(x, edge_index)
           # x = F.leaky_relu(x)

        #print('c', x[0])

        x = global_mean_pool(x, batch)

        #print('d', x[0])

        x = self.linear(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
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
                    #print(output, outputFeature)
                    loss = self.loss(output, outputFeature)
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
                    loss = self.loss(output, outputFeature)
                    val_loss += loss.item()
            val_loss /= len(data_val)
            self.valLossHistory.append(val_loss)
            if verbose:
                print(f'Epoch {epoch+1} Loss --> {epoch_loss:.4f} | Val Loss --> {val_loss:.4f}')


class InterpreterDataset(Dataset):
    def __init__(self, inputFeaturesList, adjacencyMatrixList, outputFeatureList, transform=None):
        assert len(inputFeaturesList) == len(outputFeatureList), "Input and output feature lists must be of the same length"
        assert len(inputFeaturesList) == len(adjacencyMatrixList), "Input feature and adjacency matrix lists must be of the same length"
        self.inputFeaturesList = inputFeaturesList
        if transform:
            self.adjacencyMatrixList = [self.transform(adjacencyMatrix) for adjacencyMatrix in adjacencyMatrixList]
        else:
            self.adjacencyMatrixList = adjacencyMatrixList
        self.outputFeatureList = outputFeatureList

    def __len__(self):
        return len(self.inputFeaturesList)

    def __getitem__(self, idx):
        # Convert dense adjacency matrix to edge index and edge_weight (distance)
        edge_index, edge_weight = dense_to_sparse(self.adjacencyMatrixList[idx])
        # Optionally limit to 1000 edges
        # edge_index = edge_index[:, :50]
        # edge_weight = edge_weight[:50]                    self.optimizer.zero_grad()
        data = Data(x=self.inputFeaturesList[idx], edge_index=edge_index, edge_weight=edge_weight.t().contiguous(), y=self.outputFeatureList[idx])
        #print(self.inputFeaturesList[idx, :, 0], self.inputFeaturesList[idx, 0, -1],  self.outputFeatureList[idx])
        return data
