import networkx as nx
import itertools
import torch
import pickle
from torch_geometric.utils import to_networkx
from networkx.algorithms import isomorphism
from tqdm import tqdm
import math



def read_from_fsg(name, pattern_index):

    """
    Creates networkx graph from fsg output file
    
    Parameters:
    p (nx.graph): pattern
    dataset (list of torch_geometric.data): target graphs

    Returns:
    list of torch_geometric.data: input target graphs with pattern count as class label
    """
  
    # extract relevant lines from file
    with open(f'../fsg/{name}_fsg.fp', 'r') as file:
        skip_lines = itertools.dropwhile(lambda line: not line.startswith(f't # {pattern_index}'), file)
        next(skip_lines, None)
        extract_lines = list(itertools.takewhile(lambda line: line.startswith('v') or line.startswith('u'), skip_lines))

    # read files for converting hashes back to original node and edge attributes
    file = open(f'../fsg/{name}_hash2attr.pkl', 'rb')
    node_hash2label, edge_hash2label = pickle.load(file)

    # setup node and edge label dicts
    node_labels = {}
    edge_labels = {}
    for line in extract_lines:
        splits = line.split()
        if splits[0] == "v":
            v = int(splits[1])
            l = node_hash2label[int(splits[2])]
            node_labels[v] = l
        if splits[0] == "u":
            u = int(splits[1])
            v = int(splits[2])
            l = edge_hash2label[int(splits[3])]
            edge_labels[(u,v)] = l

    # create pattern
    nodes = list(node_labels.keys())
    edges = list(edge_labels.keys())
    p = nx.Graph()
    p.add_nodes_from(nodes)
    p.add_edges_from(edges)
    nx.set_node_attributes(p, node_labels, 'x')
    nx.set_edge_attributes(p, edge_labels, 'edge_attr')

    return p



# aux method for count
def node_match(n1, n2):
    return n1['x'] == n2['x']

# aux method for count
def edge_match(e1, e2):
    return e1['edge_attr'] == e2['edge_attr']


def count_induced(p, dataset):

    """
    Counts the number of occurences of pattern p in each graph of dataset
    
    Parameters:
    p (nx.graph): pattern
    dataset (list of torch_geometric.data): target graphs

    Returns:
    list of torch_geometric.data: input target graphs with pattern count as class label
    """

    for data in tqdm(dataset, ncols=64):
        g = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
        matcher = isomorphism.GraphMatcher(g, p, node_match=node_match, edge_match=edge_match)
        isomorphism_count = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
        c = isomorphism_count
        data.y = torch.tensor(c)



def count_noninduced(p, dataset):

    """
    Counts the number of occurences of pattern p in each graph of dataset
    
    Parameters:
    p (nx.graph): pattern
    dataset (list of torch_geometric.data): target graphs

    Returns:
    list of torch_geometric.data: input target graphs with pattern count as class label
    """

    for data in tqdm(dataset, ncols=64):
        g = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
        matcher = isomorphism.GraphMatcher(g, p, node_match=node_match, edge_match=edge_match)
        isomorphism_count = sum(1 for _ in matcher.subgraph_monomorphisms_iter())
        c = isomorphism_count
        data.y = torch.tensor(c)