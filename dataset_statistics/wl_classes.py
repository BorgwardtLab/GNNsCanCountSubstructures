import argparse
import random
import torch, torchvision
import torch_geometric
from torch_geometric.datasets import TUDataset, LRGBDataset, ZINC
from ogb.graphproppred import PygGraphPropPredDataset
import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from weisfeiler_lehman import WL
import utils.transforms as transforms

def node_match(n1, n2):
    return (n1['x'] == n2['x']) and (n1['is_root'] == n2['is_root'])

def edge_match(e1, e2):
    return e1['edge_attr'] == e2['edge_attr']

def is_duplicate(subg, wl2subgs):
    for h in wl2subgs:
        if nx.is_isomorphic(subg, h, node_match=node_match, edge_match=edge_match):
            return True
    return False


# Return the subgraph induced by the k-hop neighborhood of the input node n of graph g
def k_hop_induced_subgraph(g, n, k):
    nodes_within_k_hops = nx.single_source_shortest_path_length(g, n, cutoff=k).keys()
    subgraph = g.subgraph(nodes_within_k_hops).copy()
    return subgraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='None')
    parser.add_argument("--noedge", action='store_true')
    parser.add_argument("--fullfeatures", action='store_true')
    args = parser.parse_args()

    trs = [transforms.NodeAndEdgeAttrStandardizer()]
    name = args.dataset
    if args.dataset in ['Mutagenicity', 'MCF-7']:
        dataset = TUDataset(root='data', name=name, transform=torchvision.transforms.Compose(trs))
    elif args.dataset in ['ogbg-molhiv', 'ogbg-molpcba']:
        dataset = PygGraphPropPredDataset(root='data', name=name, transform=torchvision.transforms.Compose(trs))
    elif args.dataset in ['Peptides-func', 'PCQM-Contact']:
        dataset = []
        data = LRGBDataset(root='data', name=name, transform=torchvision.transforms.Compose(trs), split='train')
        dataset += [x for x in data]
        data = LRGBDataset(root='data', name=name, transform=torchvision.transforms.Compose(trs), split='val')
        dataset += [x for x in data]
        data = LRGBDataset(root='data', name=name, transform=torchvision.transforms.Compose(trs), split='test')
        dataset += [x for x in data]
    elif args.dataset in ['ZINC']: 
        dataset = []
        data = ZINC(root='data', subset=True, transform=torchvision.transforms.Compose(trs), split='train')
        dataset += [x for x in data]
        data = ZINC(root='data', subset=True, transform=torchvision.transforms.Compose(trs), split='val')
        dataset += [x for x in data]
        data = ZINC(root='data', subset=True, transform=torchvision.transforms.Compose(trs), split='test')
        dataset += [x for x in data]
    else:
        raise Exception('Dataset not present')


    print(name)
    # prepare data
    dataset = [data for data in dataset]
    if (not args.fullfeatures) and (name in ['ogbg-molhiv', 'ogbg-molpcba', 'Peptides-func', 'PCQM-Contact']):
        # keep only first node attribute entry corresponding to the atom type
        for data in dataset:
            data.x = data.x[:,:1]

    wl = WL()
    k = 8

    graphs = []
    for data in tqdm(dataset, ncols=64):
        if args.noedge:
            data.edge_attr = torch.zeros_like(data.edge_attr)
        g = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
        wl.perform_k_steps(g, k)
        graphs.append(g)


    labels = set()
    for g in graphs:
        for n in g.nodes(data=True):
            lbl = str(n[1]['x'])
            labels.add(lbl)
    print('vlabels', len(labels))

    labels = set()
    for g in graphs:
        for n in g.edges(data=True):
            lbl = str(n[2]['edge_attr'])
            labels.add(lbl)
    print('elabels', len(labels))

    for i in range(1,k+1):
        wls = set()
        for g in graphs:
            for n in g.nodes(data=True):
                lbl = n[1][f'wl_{i}']
                wls.add(lbl)
        print('eta', i, len(wls))
    
    for i in range(1,k+1):
        wls = set()
        for g in graphs:
            inv = g.graph[f'invariant_{i}']
            wls.add(inv)
        print('wl classes', i, len(wls))

    wls = {}
    for g in graphs:
        inv = g.graph[f'invariant_{k}']
        if inv not in wls:
            wls[inv] = [g]
        else:
            insert = True
            for g2 in wls[inv]:
                if nx.is_isomorphic(g, g2):
                    insert = False 
                    break
            if insert:
                wls[inv].append(g)
    iso_classes = 0
    for k, v in wls.items():
        iso_classes += len(v)
    print('iso classes', len(wls))



if __name__ == "__main__":

    main()