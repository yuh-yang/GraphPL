from .parse_args import args
from os import path
import numpy as np
import scipy.sparse as sp
import torch
import logging
from copy import deepcopy
import time
# PyG imports
# from torch_geometric.data import Data
# from torch_geometric.sampler.base import NumNeighbors
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.sampler.utils import to_csc

import dgl

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)

class EdgeListData_sampling:
    def __init__(self, train_file, test_file, phase='pretrain', pre_dataset=None, has_time=True):
        self.phase = phase
        self.pre_dataset = pre_dataset

        self.edgelist = []
        self.edge_time = []
        self.num_users = 0
        self.num_items = 0
        self.num_edges = 0

        self.train_user_dict = {}
        self.test_user_dict = {}

        self._load_from_file(train_file, test_file, has_time)

        if phase == 'pretrain':
            self.user_hist_dict = self.train_user_dict
        elif phase == 'finetune':
            self.user_hist_dict = deepcopy(pre_dataset.user_hist_dict)
            for u in self.train_user_dict:
                self.user_hist_dict[u].extend(self.train_user_dict[u])

    def _load_from_file(self, train_file, test_file, has_time=True):
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if not has_time:
                    user, items = line
                    times = " ".join(["0"] * len(items.split(" ")))
                else:
                    user, items, times = line
                    
                for i in items.split(" "):
                    self.edgelist.append((int(user), int(i)))
                for i in times.split(" "):
                    self.edge_time.append(int(i))
                self.train_user_dict[int(user)] = [int(i) for i in items.split(" ")]
        self.edgelist = np.array(self.edgelist, dtype=np.int32)
        self.edge_time = np.array(self.edge_time, dtype=np.int32)
        self.num_edges = len(self.edgelist)
        if self.phase == 'finetune':
            self.num_users = self.pre_dataset.num_users
            self.num_items = self.pre_dataset.num_items
        else:
            self.num_users = np.max(self.edgelist[:, 0]) + 1
            self.num_items = np.max(self.edgelist[:, 1]) + 1

        logger.info('Number of users: {}'.format(self.num_users))
        logger.info('Number of items: {}'.format(self.num_items))
        logger.info('Number of edges: {}'.format(self.num_edges))

        # self.graph = sp.coo_matrix((np.ones(self.num_edges), (self.edgelist[:, 0], self.edgelist[:, 1])), shape=(self.num_users, self.num_items))

        # item starts with num_users
        self.graph_dgl = dgl.graph((self.edgelist[:, 0], self.edgelist[:, 1]+self.num_users), num_nodes=self.num_users+self.num_items).to(args.device)
        self.graph_dgl.edata['time'] = torch.tensor(self.edge_time).to(args.device)
        # self.graph_dgl.ndata[dgl.NID] = torch.arange(self.num_users+self.num_items).to(args.device)

        self.test_edge_num = 0
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                user, items = line
                self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
                self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))
    
    def sample_subgraph(self):
        pass

    def dgl_sample(self, graph, fanouts, seeds):
        sample_out = []
        for i in fanouts:
            i_hop_subgraph = dgl.sampling.sample_neighbors(graph, seeds, i)
            i_block = dgl.to_block(i_hop_subgraph, seeds)
            i_block.edata[dgl.EID] = i_hop_subgraph.edata[dgl.EID]
            seeds = i_block.srcdata[dgl.NID]
            sample_out.append(i_block)

        sample_out = dgl.merge(sample_out)
        return sample_out

    def get_train_batch(self, start, end):

        def negative_sampling(user_item, train_user_set):
            neg_items = []
            for user, _ in user_item:
                user = int(user)
                while True:
                    neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
                    if neg_item not in train_user_set[user]:
                        break
                neg_items.append(neg_item)
            return neg_items

        # time_1 = time.time()

        ui_pairs = self.edgelist[start:end]
        users = torch.LongTensor(ui_pairs[:, 0]).to(args.device)
        pos_items = torch.LongTensor(ui_pairs[:, 1]).to(args.device)
        neg_items = torch.LongTensor(negative_sampling(ui_pairs, self.train_user_dict)).to(args.device)

        # time_2 = time.time()

        central_nodes = torch.cat([users, pos_items+self.num_users, neg_items+self.num_users])
        sample_out = self.dgl_sample(self.graph_dgl, [5,5,5], central_nodes)
        sample_eids = sample_out.edata[dgl.EID]
        sample_nids = sample_out.ndata[dgl.NID]
        nodes = sample_out.nodes()
        edges = torch.stack(sample_out.edges(), dim=1)
        # 验证reid是否正确
        # edge_0 = edges[0]
        # edge_0_oid = sample_nids[edge_0]
        # edge_0_rid_0 = (sample_nids == edge_0_oid[0]).nonzero().squeeze()
        # edge_0_rid_1 = (sample_nids == edge_0_oid[1]).nonzero().squeeze()
        # print(sample_nids[(nodes == edge_0_rid_0).nonzero().squeeze()])

        nodes_list = nodes.cpu().tolist()
        nids_list = sample_nids.cpu().tolist()
        node_mapping = {nids_list[i]:nodes_list[i] for i in range(len(nodes))}
        node_mapping = [node_mapping[i] for i in central_nodes.cpu().tolist()]

        # time_3 = time.time()
        # print(f"batch time: {time_3-time_1}, sampling time: {time_2-time_1}, postprocessing time: {time_3-time_2}")

        # node_mapping = {sample_nids[i].cpu().item():nodes[i].cpu().item() for i in range(len(nodes))}
        # # for u in users.cpu().tolist():
        # #     assert u in node_mapping
        # # for i in pos_items.cpu().tolist():
        # #     assert i+self.num_users in node_mapping
        # # for i in neg_items.cpu().tolist():
        # #     assert i+self.num_users in node_mapping


        return users, pos_items, neg_items, [nodes, edges, sample_nids, node_mapping]

    def shuffle(self):
        random_idx = np.random.permutation(self.num_edges)
        self.edgelist = self.edgelist[random_idx]
        self.edge_time = self.edge_time[random_idx]
        
    def _generate_binorm_adj(self, edgelist):
        adj = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])),
                            shape=(self.num_users, self.num_items), dtype=np.float32)
        
        a = sp.csr_matrix((self.num_users, self.num_users))
        b = sp.csr_matrix((self.num_items, self.num_items))
        adj = sp.vstack([sp.hstack([a, adj]), sp.hstack([adj.transpose(), b])])
        adj = (adj != 0) * 1.0
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        adj = adj.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        ui_adj = adj.tocsr()[:self.num_users, self.num_users:].tocoo()
        return adj

if __name__ == '__main__':
    args.device = 'cpu'
    edgelist_dataset = EdgeListData_sampling("dataset/taobao/pretrain.txt", "dataset/yelp_small/pretrain_val.txt")
    edgelist_dataset.shuffle()
    edgelist_dataset.get_train_batch(0, 100)