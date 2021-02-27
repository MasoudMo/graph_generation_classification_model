from dgl.nn.pytorch import SGConv
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
import dgl
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import normal
from torch import exp
from torch import transpose
from torch import matmul

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
        self.conv_1 = SGConv(in_feats=input_dim, out_feats=hidden_dim, k=3)
        self.conv_2 = SGConv(in_feats=hidden_dim, out_feats=hidden_dim, k=3)

        # Define the fully connected layers
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 1)

        # The output activation function
        self.output_func = nn.Sigmoid()

    def forward(self, g):
        """Performs the classification's forward path

        Performs the forward path of the binary graph classification

        Args:
            g:
                The input graph to perform classification on

        Returns:
            The output of the sigmoid function indicating the classification for the input graph g
        """

        # Extract the graph's node features
        h = g.ndata['x']

        # Perform convolutional layers with Relu as the activation function
        h = F.relu(self.conv_1(g, h))
        h = F.relu(self.conv_2(g, h))

        # Store the generated node embeddings
        g.ndata['h'] = h

        # Find the mean of node embeddings to use as the graph embedding
        hg = dgl.mean_nodes(g, 'h')

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
        self.conv_shared = SAGEConv(in_feats=input_dim, out_feats=hidden_dim_1, aggregator_type='mean')
        self.conv_mean = SAGEConv(in_feats=hidden_dim_1, out_feats=hidden_dim_2, aggregator_type='mean')
        self.conv_log_std = SAGEConv(in_feats=hidden_dim_1, out_feats=hidden_dim_2, aggregator_type='mean')

        # The output activation function
        self.output_func = nn.Sigmoid()

        # Other attributes
        self.num_nodes = num_nodes
        self.hidden_dim_2 = hidden_dim_2

    def forward(self, g):
        """Performs the graph generation's forward path

        Performs the encoder and decoder for the graph generation process.

        Args:
            g:
                The input graph to use for the generation process

        Returns:
            The reconstructed graph adjacency matrix
        """

        # Extract the graph's node features
        h = g.ndata['x']

        # Perform the GNN layer that is shared for both the mean and the std layers
        h = F.relu(self.conv_shared(g, h))

        # Perform the GNN layer to obtain embedding means
        h_mean = self.conv_mean(g, h)

        # Perform the GNN layer to obtain embeddings std
        h_log_std = self.conv_log_std(g, h)

        # Generate the posterior embeddings
        distribution = normal.Normal(loc=h_mean, scale=exp(h_log_std))
        z = distribution.sample()
        z.requires_grad = True

        # Reconstruct the graph
        reconstruction = matmul(z, transpose(z, 0, 1))

        return self.output_func(reconstruction)
