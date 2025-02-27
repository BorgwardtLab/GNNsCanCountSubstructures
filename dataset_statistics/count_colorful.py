import argparse
import pickle
import random
from tqdm import tqdm
import torch, torchvision
import torch_geometric
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, LRGBDataset, ZINC
from ogb.graphproppred import PygGraphPropPredDataset
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_networkx
from networkx.algorithms import isomorphism

import utils.pattern as pattern
import utils.transforms as transforms
from weisfeiler_lehman import WL



# return directed graph g rooted at node r
def to_rooted(g, r):
    g_rooted = nx.DiGraph()
    for u in g.nodes():
        g_rooted.add_node(u)
    for u, v in g.edges():
        if nx.shortest_path_length(g, source=r, target=v) < nx.shortest_path_length(g, source=r, target=u):
            g_rooted.add_edge(v,u)
        else:
            g_rooted.add_edge(u,v)
    return g_rooted
    
# check condition (i) of quite colorfulness
def condition_i(g, subiso, rooted_p, k):
    for node in rooted_p.nodes():
        g_node = subiso[node]
        for child in rooted_p.successors(node):
            for grandchild in rooted_p.successors(child):
                g_grandchild = subiso[grandchild]
                if g.nodes[g_node][f'wl_{k}'] == g.nodes[g_grandchild][f'wl_{k}']:
                    return False
    return True

# check condition (ii) of quite colorfulness
def condition_ii(g, k):
    for u in g.nodes():
        for v in g.nodes():
            if nx.shortest_path_length(g, u, v) >= 3:
                if g.nodes[u][f'wl_{k}'] == g.nodes[v][f'wl_{k}']:
                    return False
    return True


# aux method for count
def node_match(n1, n2):
    return n1['x'] == n2['x']

# aux method for count
def edge_match(e1, e2):
    return e1['edge_attr'] == e2['edge_attr']
            



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='None')
    parser.add_argument("--noedge", action='store_true')
    parser.add_argument("--fullfeatures", action='store_true')
    parser.add_argument("--pattern-idx", type=str, nargs='+', default='None')
    parser.add_argument("--k", type=int, default=8)
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

    # output file
    output = {'args': args}

    g = dataset[0]

    # load pattern(s)
    for idx in args.pattern_idx:
        p = pattern.read_from_fsg(args.dataset, idx)
        print('Pattern index:', idx)
        print(p)
        print(p.edges())
        print(nx.get_node_attributes(p, 'x'))
        output[idx] = {}
        output[idx]['pattern'] = p


        # initialize wl and counts
        wl = WL()
        k = args.k
        subiso_count = 0
        quitecolorful_counts = [[0] * (k+1) for r in list(p.nodes())]

        # iterate over all dataset graphs
        for data in tqdm(dataset, ncols=64):
            if args.noedge:
                data.edge_attr = torch.zeros_like(data.edge_attr)
            # cast graph to networkx
            g = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)
            
            # perform wl and assign wl colors
            wl.perform_k_steps(g,k)
            node_attrs = nx.get_node_attributes(g,'x')
            nx.set_node_attributes(g, node_attrs, 'wl_0')  
            wl_attrs = nx.get_node_attributes(g, f'wl_{k}')
            nx.set_node_attributes(g, wl_attrs, 'wl') 
            
            # iterate over all subgraph isomorphisms
            matcher = isomorphism.GraphMatcher(g, p, node_match=node_match, edge_match=edge_match)
            for subiso in matcher.subgraph_monomorphisms_iter():
                subiso_count += 1

                # get target subgraph that is isomorphic to the pattern
                subiso = {v: k for k, v in subiso.items()}
                subgraph_edges = [(subiso[u],subiso[v]) for u,v in p.edges()]
                subgraph = nx.edge_subgraph(g, subgraph_edges)

                # check for each wl depth whether subgraph isomorphism is quite-colorful
                is_quitecolorful_withroot = {r:False for r in list(p.nodes())}
                for i in range(k+1):
                    cond_ii = condition_ii(subgraph, i)
                    
                    # check for every root node
                    for r in list(p.nodes()):
                        if is_quitecolorful_withroot[r]:
                            continue
                        rooted_p = to_rooted(p,r)
                        if condition_i(subgraph, subiso, rooted_p, i) and cond_ii:
                            is_quitecolorful_withroot[r] = True
                            for j in range(i,k+1):
                                quitecolorful_counts[r][j] += 1


        print('subiso_count', subiso_count)
        for r in list(p.nodes()):
            print('root', r)
            for i in range(k+1):
                print('k', i, quitecolorful_counts[r][i], round(quitecolorful_counts[r][i]/subiso_count,4))

        output[idx]['subiso_count'] = subiso_count
        output[idx]['quitecolorful_counts'] = quitecolorful_counts 

        
    outfile = open(f'quitecolorfulcounts/{args.dataset}_{args.pattern_idx}_quitecolorful_counts.pickle', 'wb')
    pickle.dump(output, outfile)

if __name__ == "__main__":

    main()