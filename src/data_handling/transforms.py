# In src/data_handling/transforms.py

import torch
from torch_geometric.data import Data

class SubsampleNodes(object):
    """
    A PyG transform that randomly subsamples a fixed number of nodes
    from a graph data object.
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def __call__(self, data: Data) -> Data:
        """
        Args:
            data (Data): The graph data object.

        Returns:
            Data: The subsampled graph data object.
        """
        num_nodes_original = data.num_nodes

        if num_nodes_original <= self.num_nodes:
            return data

        perm = torch.randperm(num_nodes_original)
        node_mask = perm[:self.num_nodes]

        return data.subgraph(node_mask)