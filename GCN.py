
from torch.autograd import Variable
import torch

import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from tqdm import tqdm
# create a Graph Convolutional Neural Network
# using PyTorch


class GraphConv(torch.nn.Module):
    def __init__(self, input_size, lr):
        super(GraphConv, self).__init__()
        
        self.GCN1 = GCNConv(input_size, 100)
        self.GCN2 = GCNConv(100, 50)
        self.GCN3 = GCNConv(50, 25)


        self.linear = torch.nn.Linear(25, 25)
        self.linear2 = torch.nn.Linear(25, 10)
        self.linear3 = torch.nn.Linear(10, 1)
    


        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=.001)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputFeatures, edge_index):
        x = self.GCN1(inputFeatures, edge_index)
        x = F.relu(x)
        x = self.GCN2(x, edge_index)
        x = F.relu(x)
        x = self.GCN3(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)

        return x

    def trainModel(self, data_train, data_val, epochs, verbose=True):
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.train()  # set to train mode
            with tqdm(enumerate(data_train), total=len(data_train), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for i, data in pbar:
                    inputFeatures = data.x
                    edge_index = data.edge_index
                    outputFeature = data.y
                    output = self.forward(inputFeatures, edge_index)
                    loss = self.loss(output, outputFeature)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            val_loss = 0
            self.eval()  # set to eval mode
            with torch.no_grad():
                for data in data_val:
                    inputFeatures = data.x
                    edge_index = data.edge_index
                    outputFeature = data.y
                    output = self.forward(inputFeatures, edge_index)
                    loss = self.loss(output, outputFeature)
                    val_loss += loss.item()
            val_loss /= len(data_val)
            if verbose:
                print(f'Epoch {epoch+1} Loss --> {epoch_loss/len(data_train):.4f} | Val Loss --> {val_loss:.4f}')


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

        # Convert dense adjacency matrix to edge index
        edge_index, _ = dense_to_sparse(self.adjacencyMatrixList[idx])
        edge_index = edge_index[:, :1000]  # Limit to 1000 edges

        data = Data(x=self.inputFeaturesList[idx], edge_index=edge_index, y=self.outputFeatureList[idx])

        return data
