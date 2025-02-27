import torch
import random
import numpy as np
from torch_geometric.transforms import BaseTransform
import dejavu_gi
import torch, torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import * 



class NodeAndEdgeAttrStandardizer(BaseTransform):
    def __init__(self, node_value=1.0, edge_value=1.0, node_attr_dim=1, edge_attr_dim=1):
        self.node_value = float(node_value)  # Ensure the constant value is float
        self.edge_value = float(edge_value)
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim

    def __call__(self, data):
        # Check and handle node features (x)
        if data.x is None:
            # Add constant node attributes if not present, ensure they are floats
            num_nodes = data.num_nodes
            data.x = torch.full((num_nodes, self.node_attr_dim), self.node_value).float()
        else:
            # Reshape node attributes if present, and cast to float
            if data.x.dim() == 1:
                data.x = data.x.unsqueeze(1)
            data.x = data.x.float()  # Ensure node attributes are floats
        
        # Check and handle edge attributes (edge_attr)
        if data.edge_attr is None:
            # Add constant edge attributes if not present, ensure they are floats
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.full((num_edges, self.edge_attr_dim), self.edge_value).float()
        else:
            # Reshape edge attributes if present, and cast to float
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.unsqueeze(1)
            data.edge_attr = data.edge_attr.float()  # Ensure edge attributes are floats
        
        return data
