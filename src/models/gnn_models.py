from dgl.nn.pytorch import SGConv
import dgl
import torch.nn.functional as F
import torch.nn as nn


class BinaryGraphClassifier(nn.Module):
    """Torch NN module class performing graph classification

    Uses DGL's GraphConv GNN model to perform binary classification on input graphs

    """

    def __init__(self, input_dim, hidden_dim):
        """Initializes the model

        Initializes the GNN layers using the input and hidden dimensions.

        Args:
            input_dim:
                The dimension of the input data (e.g. number of PCA components)
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

