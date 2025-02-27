import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch, torchvision
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, LRGBDataset, ZINC
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
import networkx as nx
from collections import Counter

import utils.transforms as transforms
import utils.pattern as pattern
from utils.metrics import normalized_mae
import models



def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dataset", type=str, default='None')
    parser.add_argument("--pattern-idx", type=str, default='None')
    parser.add_argument("--fullfeatures", action='store_true')
    
    args = parser.parse_args()
    args.cuda = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print('device', args.device)

    torch.set_printoptions(precision=3, linewidth=300)

    return args



def main():
    args = cline()

    # load dataset
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

    print(args.dataset)
    print(args.pattern_idx)

    # print dataset graph
    print('sample target')
    print(dataset[0])
    print(dataset[0].x)
    print(dataset[0].edge_attr)

    # load pattern
    pattern_graph = pattern.read_from_fsg(args.dataset, args.pattern_idx)

    # count pattern
    pattern.count_noninduced(pattern_graph, dataset)

    # print class balance
    classes = [int(g.y) for g in dataset]
    nmb_classes = max(classes)+1
    class_counter = Counter(classes)
    normalized_counter = {k:v/sum(class_counter.values()) for k,v in class_counter.items()}
    print('\nclass balance')
    print(normalized_counter)

    # split training/test
    split = 0.8
    random.shuffle(dataset)
    train = dataset[0:int(split*len(dataset))]
    test = dataset[int(split*len(dataset)):]

    # initialize
    model = models.NetEdge(train[0].x.shape[1], train[0].edge_attr.shape[1], args.hidden, out_channels=nmb_classes, layers=args.layers).to(args.device)
    batch_size = args.batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)



    for epoch in tqdm(range(args.epochs), ncols=64):
        
        model.train()
        for data in train_loader:
            optimizer.zero_grad()

            data = data.to(args.device)
            
            lgits = model(data.x, data.edge_attr, data.edge_index, data.batch)
            loss = F.cross_entropy(lgits, data.y)

            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        labels = []
        preds = torch.empty(0, dtype=torch.int64)
        probs = torch.empty(0)
        with torch.no_grad():
            for data in train_loader:
                data = data.to(args.device)
                
                lgits = model(data.x, data.edge_attr, data.edge_index, data.batch)
                pred = torch.argmax(lgits, dim=1).cpu()
                prob = torch.softmax(lgits, dim=1).cpu()

                preds = torch.cat((preds, pred), dim=0)
                probs = torch.cat((probs, prob), dim=0)
                labels += data.y.cpu().tolist()
        
        train_acc = accuracy_score(labels, preds.tolist())
        train_mae = mean_absolute_error(labels, preds.tolist())
        train_nmae = normalized_mae(labels, preds.tolist())
        train_roc = roc_auc_score(labels, probs.tolist(), multi_class='ovo', labels=list(range(nmb_classes)))


        labels = []
        preds = torch.empty(0, dtype=torch.int64)
        probs = torch.empty(0)
        with torch.no_grad():
            for data in test_loader:
                data = data.to(args.device)
                
                lgits = model(data.x, data.edge_attr, data.edge_index, data.batch)
                pred = torch.argmax(lgits, dim=1).cpu()
                prob = torch.softmax(lgits, dim=1).cpu()
                
                preds = torch.cat((preds, pred), dim=0)
                probs = torch.cat((probs, prob), dim=0)
                labels += data.y.cpu().tolist()

        test_acc = accuracy_score(labels, preds.tolist())
        test_mae = mean_absolute_error(labels, preds.tolist())
        test_nmae = normalized_mae(labels, preds.tolist())
        test_roc = roc_auc_score(labels, probs.tolist(), multi_class='ovo', labels=list(range(nmb_classes)))


        print("Epoch", epoch)
        print('loss', loss.item())
        print('acc', train_acc, test_acc)
        print('mae', train_mae, test_mae)
        print('nmae', train_nmae, test_nmae)
        print('auc', train_roc, test_roc)


if __name__ == "__main__":
    main()