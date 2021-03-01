from dgl.nn.pytorch import DenseGraphConv
from dgl.nn.pytorch import DenseSAGEConv
import dgl
import torch.nn.functional as F
import torch.nn as nn
from torch import randn_like
from torch import transpose
from torch import matmul
from torch import exp
from torch import mean


class BinaryGraphClassifier(nn.Module):
    """Torch NN module class performing graph classification

    Uses DGL's GraphConv GNN model to perform binary classification on input graphs

    """

    def __init__(self, input_dim, hidden_dim):
        """Initializes the model

        Initializes the GNN layers using the input and hidden dimensions.

        Args:
            input_dim:
                The dimension of the input data
            hidden_dim:
                The dimension of the hidden layer
        """
        super(BinaryGraphClassifier, self).__init__()

        # Define the graph convolutional layers
        self.conv_1 = DenseGraphConv(in_feats=input_dim, out_feats=hidden_dim)
        self.conv_2 = DenseGraphConv(in_feats=hidden_dim, out_feats=hidden_dim)

        # Define the fully connected layers
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 1)

        # The output activation function
        self.output_func = nn.Sigmoid()

    def forward(self, adj, features):
        """Performs the classification's forward path

        Performs the forward path of the binary graph classification

        Args:
            adj:
                Input graph adjacency matrix
            features:
                The input node features

        Returns:
            The output of the sigmoid function indicating the classification for the input graph g
        """

        # Perform convolutional layers with Relu as the activation function
        h = F.relu(self.conv_1(adj, features))
        h = F.relu(self.conv_2(adj, h))

        # Find the mean of node embeddings to use as the graph embedding
        hg = mean(h, dim=0)

        # Perform the linear layers
        h = F.relu(self.fc_1(hg))
        out = F.relu(self.fc_2(h))

        # Perform the output activation function
        out = self.output_func(out)

        return out


class VariationalGraphAutoEncoder(nn.Module):
    """Torch NN Module performing graph generation using VGAE

    Performs graph generation using VGAE as described in: https://arxiv.org/abs/1611.07308

    """

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, num_nodes):
        """Initializes the model

        Initializes the GNN layers using the input and hidden dimensions.

        Args:
            input_dim:
                The dimension of the input data (e.g. number of PCA components)
            hidden_dim_1:
                The dimension of the first hidden layer
            hidden_dim_2:
                The dimension of the second hidden layer
        """
        super(VariationalGraphAutoEncoder, self).__init__()

        # Define the graph convolutional layers
        self.conv_shared = DenseGraphConv(in_feats=input_dim, out_feats=hidden_dim_1)
        self.conv_mean = DenseGraphConv(in_feats=hidden_dim_1, out_feats=hidden_dim_2)
        self.conv_log_std = DenseGraphConv(in_feats=hidden_dim_1, out_feats=hidden_dim_2)

        # The output activation function
        self.output_func = nn.Sigmoid()

        # Other attributes
        self.num_nodes = num_nodes
        self.hidden_dim_2 = hidden_dim_2
        self.h_mean = None
        self.h_log_std = None

        self.z = None

    def forward(self, adj, features):
        """Performs the graph generation's forward path

        Performs the encoder and decoder for the graph generation process.

        Args:
            adj:
                Input graph adjacency matrix
            features:
                The input node features

        Returns:
            The reconstructed graph adjacency matrix
        """

        # Perform the GNN layer that is shared for both the mean and the std layers
        h = F.relu(self.conv_shared(adj, features))

        # Perform the GNN layer to obtain embedding means
        self.h_mean = self.conv_mean(adj, h)

        # Perform the GNN layer to obtain embeddings std
        self.h_log_std = self.conv_log_std(adj, h)

        # Generate the posterior embeddings
        self.z = self.h_mean + randn_like(self.h_mean) * exp(self.h_log_std)

        # Reconstruct the graph
        reconstruction = matmul(self.z, transpose(self.z, 0, 1))

        return self.output_func(reconstruction)
