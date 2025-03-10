import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, heads=2):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)  # Graph-level pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
