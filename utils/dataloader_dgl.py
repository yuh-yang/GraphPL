from parse_args import args
from os import path
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import dgl
import torch


class EdgeListData:
    def __init__(self, edgelist_path):
        self.edgelist_path = edgelist_path
        self.edgelist = []
        self.edge_weight = []
        self.node_mapping = {}
        self.node_id_mapping = {}
        self.num_nodes = 0
        self.num_edges = 0

        self._load_edge_partition()

    def _map_node(self, node):
        if node not in self.node_mapping:
            id = len(self.node_mapping)
            self.node_mapping[node] = id
            self.node_id_mapping[id] = node
        return self.node_mapping[node]

    def _load_edge_partition(self):
        for partition_idx in tqdm(range(1000)):
            file_path = path.join(self.edgelist_path, 'part-{}'.format(str(partition_idx).zfill(5)))
            if not path.exists(file_path):
                break
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip().split(',')
                    src, dst, w = line[0], line[1], float(line[2])
                    src = self._map_node(src)
                    dst = self._map_node(dst)
                    self.edgelist.append((src, dst))
                    self.edge_weight.append(w)
        self.edgelist = np.array(self.edgelist, dtype=np.int32)
        self.edge_weight = np.array(self.edge_weight, dtype=np.float32)
        self.num_edges = len(self.edgelist)
        self.num_nodes = len(self.node_mapping)
        print('Number of nodes: {}'.format(self.num_nodes))
        print('Number of edges: {}'.format(self.num_edges))
    
    def get_edges(self):
        test_size = int(len(self.edgelist) * 0.1)
        train_size = len(self.edgelist) - test_size

        sorted_eids = np.argsort(self.edge_weight)
        test_eid_candidate = sorted_eids[-int(0.5*self.num_edges):]
        test_eid = np.random.choice(test_eid_candidate, test_size, replace=False)
        test_edges = self.edgelist[test_eid]
        test_w = self.edge_weight[test_eid]
        train_edges = np.delete(self.edgelist, test_eid, axis=0)
        train_w = np.delete(self.edge_weight, test_eid, axis=0)

        train_edges = torch.from_numpy(train_edges).long()
        test_edges = torch.from_numpy(test_edges).long()

        train_edge_weights = torch.from_numpy(train_w).float()
        test_edge_weights = torch.from_numpy(test_w).float()
        return train_edges, test_edges, train_edge_weights, test_edge_weights
    
    def get_edges_full(self):
        edges = torch.from_numpy(self.edgelist)
        edge_weights = torch.from_numpy(self.edge_weight)
        return edges, edge_weights

    def load_data_sample(self):
        sample_size = args.neighbor_sample_num
        edges, edge_weights = self.get_edges_full()
        graph = dgl.graph((edges[:,0], edges[:,1]), num_nodes=self.num_nodes, device=args.device)
        graph.edata['weight'] = edge_weights.to(args.device)

        test_size = int(0.1 * graph.num_edges())
        test_eid = np.random.choice(np.arange(graph.num_edges()), test_size, replace=False)
        train_eid = np.delete(np.arange(graph.num_edges()), test_eid)

        train_negative_sampler = dgl.dataloading.negative_sampler.Uniform(args.neg_num)
        train_sampler = dgl.dataloading.NeighborSampler([sample_size, sample_size])
        train_sampler = dgl.dataloading.as_edge_prediction_sampler(train_sampler, negative_sampler=train_negative_sampler)
        train_dataloader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DataLoader.
            graph,                                  # The graph
            torch.from_numpy(train_eid).to(graph.device).to(torch.int32),  # The edges to iterate over
            train_sampler,                                # The neighbor sampler
            device=args.device,                          # Put the MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=args.batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=0,       # Number of sampler processes
        )


        # 99 negative samples
        test_negative_sampler = dgl.dataloading.negative_sampler.Uniform(99)
        test_sampler = dgl.dataloading.NeighborSampler([sample_size, sample_size])
        test_sampler = dgl.dataloading.as_edge_prediction_sampler(test_sampler, negative_sampler=test_negative_sampler)
        test_dataloader = dgl.dataloading.DataLoader(
            # The following arguments are specific to DataLoader.
            graph,                                  # The graph
            torch.from_numpy(test_eid).to(graph.device).to(torch.int32),  # The edges to iterate over
            test_sampler,                                # The neighbor sampler
            device=args.device,                          # Put the MFGs on CPU or GPU
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=args.batch_size,    # Batch size
            shuffle=False,       # Whether to shuffle the nodes for every epoch
            drop_last=False,    # Whether to drop the last incomplete batch
            num_workers=0,       # Number of sampler processes
        )

        self.graph = graph
        self.train_dataloader, self.test_dataloader = train_dataloader, test_dataloader
