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

    rs = [1,2,3]
    ks = [0,1,2,3]



    print(name)

    # prepare data
    dataset = [data for data in dataset]
    if (not args.fullfeatures) and (name in ['ogbg-molhiv', 'ogbg-molpcba', 'Peptides-func', 'PCQM-Contact']):
        # keep only first node attribute entry corresponding to the atom type
        for data in dataset:
            data.x = data.x[:,:1]
            
    
    wl = WL()

    graphs = []
    for data in tqdm(dataset, ncols=64):
        if args.noedge:
            data.edge_attr = torch.zeros_like(data.edge_attr)
        g = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
        wl.perform_k_steps(g, max(rs)+max(ks))
        graphs.append(g)

    for r in rs:
        for k in ks:
            k = r+k

            # datastrcuture
            wl2subgs = {}       # maps wl label to list of subgraphs

            # iterate over all graphs
            for g in graphs:
                
                # iterate over all nodes                                                            
                for n in g.nodes(data=True):

                    # generate k-egonet of node n in graph g and store information
                    subg = k_hop_induced_subgraph(g, n[0], r)
                    
                    # add root node attribute to egonet
                    isroot = {v: (True if v == n[0] else False) for v in subg.nodes()}          
                    nx.set_node_attributes(subg, isroot, 'is_root')

                    # compute depth-l wl label of node n
                    wl_lbl = n[1][f'wl_{k}']

                    # if wl label has not been seen before, add it to the wl2subg and wl2gn dictionaries
                    if wl_lbl not in wl2subgs:
                        wl2subgs[wl_lbl] = []
                    
                    # add subgraph to list if not contained
                    if not is_duplicate(subg, wl2subgs[wl_lbl]):
                        wl2subgs[wl_lbl].append(subg)

            # count number of deptj-l wl labels for which the k-egonets are not unique
            number_of_all_nodes = 0
            number_of_nodes_with_unique_wl2subg = 0
            number_of_all_graphs = 0
            number_of_graphs_with_unique_wl2subg = 0

            for g in graphs:
                number_of_all_graphs += 1
                all_nodes_of_g_are_unique = True
                for n in g.nodes(data=True):
                    number_of_all_nodes += 1
                    wl_lbl = n[1][f'wl_{k}']
                    if len(wl2subgs[wl_lbl]) == 1:
                        number_of_nodes_with_unique_wl2subg += 1
                    else:
                        all_nodes_of_g_are_unique = False
                if all_nodes_of_g_are_unique:
                    number_of_graphs_with_unique_wl2subg += 1

            print()
            print('k', r, 'l', k)
            print('quite colorful fraction (nodes): ', number_of_nodes_with_unique_wl2subg/number_of_all_nodes)
            print('quite colorful fraction (graphs):', number_of_graphs_with_unique_wl2subg/number_of_all_graphs)
            


if __name__ == "__main__":

    main()