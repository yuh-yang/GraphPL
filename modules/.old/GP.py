import torch
import torch.nn as nn
from modules.base_model import BaseModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
import numpy as np
import scipy.sparse as sp
import math
import networkx as nx
import random
from copy import deepcopy
# from torch_scatter import scatter_add, scatter_sum
from modules.utils import scatter_add, scatter_sum

init = nn.init.xavier_uniform_

# def edge_sample(edgelist):
#     sampled_edges = edgelist.T[torch.randperm(edgelist.shape[-1])][:500]
#     return sampled_edges.to(args.device)

def edge_sample(adj: sp.coo_matrix):
    sampled_edges = np.random.choice(adj.nnz, 500)
    sampled_edges = np.vstack((adj.row[sampled_edges], adj.col[sampled_edges])).T
    return torch.LongTensor(sampled_edges).to(args.device)

# def node_mask(graph: nx.graph):
#     new_graph = deepcopy(graph)
#     nodes = list(graph.nodes)
#     random.shuffle(nodes)
#     # dropout size: 500
#     nodes = nodes[:500]
#     neighbors = []
#     for node in nodes:
#         neighbors.append(list(graph.neighbors(node)))
#     new_graph.remove_nodes_from(nodes)
#     edges = new_graph.edges
#     edge_list = [[edge[0], edge[1]] for edge in edges]
#     edge_times = [graph.get_edge_data(edge[0], edge[1])['time'] for edge in edges]
#     return torch.LongTensor(edge_list).to(args.device), torch.LongTensor(edge_times).to(args.device), torch.LongTensor(node).to(args.device), neighbors

# def node_mask(edgelist, edge_times, num_users, num_items):
#     n_edgelist = deepcopy(edgelist)
#     # remap item_id to [num_users, num_users + num_items]
#     n_edgelist[:, 1] += num_users
#     sampled_nodes = np.random.randint(0, num_users+num_items, 500)
#     neighbors = []
#     for node in sampled_nodes:
#         i_edges_src = n_edgelist[:, 0] == node
#         i_edges_dst = n_edgelist[:, 1] == node
#         if i_edges_src.any() == False:
#             neighbors.append(n_edgelist[i_edges_dst, 0].tolist())
#             n_edgelist = n_edgelist[~i_edges_dst]
#             edge_times = edge_times[~i_edges_dst]
#         elif i_edges_dst.any() == False:
#             neighbors.append(n_edgelist[i_edges_src, 1].tolist())
#             n_edgelist = n_edgelist[~i_edges_src]
#             edge_times = edge_times[~i_edges_src]
#     n_edgelist[:, 1] -= num_users
#     return torch.LongTensor(n_edgelist).to(args.device), torch.LongTensor(edge_times).to(args.device), torch.LongTensor(sampled_nodes).to(args.device), neighbors

def node_mask_partial(edgelist, num_users, num_items):
    # remap item_id to [num_users, num_users + num_items]
    n_edgelist = torch.stack([edgelist[:, 0], edgelist[:, 1] + num_users], dim=1)
    nodes = np.random.randint(0, num_users+num_items, 500)
    mask = torch.ones(n_edgelist.shape[0], dtype=torch.bool).to(args.device)
    neighbors = []
    sampled_nodes = []
    for node in nodes:
        i_edges_src = n_edgelist[:, 0] == node
        i_edges_dst = n_edgelist[:, 1] == node
        # make sure the node has at least 2 neighbor, and the node is not dropped
        if i_edges_src.sum() < 2 and i_edges_dst.sum() < 2:
            continue
        sampled_nodes.append(node)
        if i_edges_src.any() == False:
            i_edges = i_edges_dst
            position = 0
        elif i_edges_dst.any() == False:
            i_edges = i_edges_src
            position = 1
        neigh_idx = i_edges.nonzero().squeeze()
        masked_negih_idx = neigh_idx[torch.randperm(neigh_idx.size(0))][:int(neigh_idx.size(0)/2)]
        mask[masked_negih_idx] = False
        # i_edges[masked_negih_idx] = False
        neighbors.append(n_edgelist[masked_negih_idx, position].tolist())
    n_edgelist[:, 1] -= num_users
    return torch.LongTensor(sampled_nodes).to(args.device), neighbors, mask

# def node_drop(graph: Data):
#     pass

class TemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 200, dropout = 0.2):
        super(TemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t)))


class GP(BaseModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.edges = torch.LongTensor(dataset.edgelist).to(args.device)
        self.edge_times = torch.LongTensor(dataset.edge_time).to(args.device)
        self.edge_norm = self._edge_binorm(self.edges)

        self.uu_adj = dataset.uu_adj
        self.ii_adj = dataset.ii_adj

        # self.uu_adj = self._sp_matrix_to_sp_tensor(dataset.uu_adj)
        # self.ii_adj = self._sp_matrix_to_sp_tensor(dataset.ii_adj)

        self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
        self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

        self.temporal_encoding = TemporalEncoding(self.emb_size)

        self.edge_dropout = EdgelistDrop()

    def _agg(self, user_emb, item_emb, edges, edge_times, edge_norm):
        user_emb = self.temporal_encoding(user_emb[edges[:, 0]], edge_times)
        item_emb = self.temporal_encoding(item_emb[edges[:, 1]], edge_times)
        # user_emb = user_emb[edges[:, 0]]
        # item_emb = item_emb[edges[:, 1]]

        # bi-norm
        user_emb = user_emb * edge_norm.unsqueeze(1)
        item_emb = item_emb * edge_norm.unsqueeze(1)

        # conv
        user_res_emb = scatter_sum(item_emb, edges[:, 0], dim=0, dim_size=self.num_users)
        item_res_emb = scatter_sum(user_emb, edges[:, 1], dim=0, dim_size=self.num_items)
        # user_res_emb = torch.scatter_add(torch.zeros([self.num_users, self.emb_size], dtype=user_emb.dtype, device=args.device), dim=0, index=edges[:, 0], src=item_emb)
        # item_res_emb = torch.scatter_add(torch.zeros([self.num_items, self.emb_size], dtype=item_emb.dtype, device=args.device), dim=0, index=edges[:, 1], src=user_emb)
        return user_res_emb, item_res_emb
    
    def _edge_binorm(self, edges):
        # user_degs = torch.scatter_add(torch.zeros([self.num_users,], dtype=edges.dtype, device=args.device), dim=0, index=edges[:, 0], src=torch.ones_like(edges[:, 0]))
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]
        # item_degs = torch.scatter_add(torch.zeros([self.num_items,], dtype=edges.dtype, device=args.device), dim=0, index=edges[:, 1], src=torch.ones_like(edges[:, 1]))
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm

    def forward(self, edges, edge_times, edge_norm=None):
        if edge_norm is None:
            edge_norm = self._edge_binorm(edges)

        user_emb = self.user_embedding
        item_emb = self.item_embedding
        user_res_emb = user_emb
        item_res_emb = item_emb
        for l in range(args.num_layers):
            user_emb, item_emb = self._agg(user_emb, item_emb, edges, edge_times, edge_norm)
            user_emb = F.normalize(user_emb)
            item_emb = F.normalize(item_emb)
            user_res_emb = user_res_emb + user_emb
            item_res_emb = item_res_emb + item_emb
        return user_res_emb, item_res_emb
    
    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 0.5, return_mask=True)
        edge_times = self.edge_times[dropout_mask]
        edge_norm = self.edge_norm[dropout_mask]

        # node dropout
        masked_nodes, neighbors, mask = node_mask_partial(edges, self.num_users, self.num_items)
        # mask = torch.ones([edges.shape[0],], dtype=torch.bool, device=args.device)
        # apply mask to edges
        edges = edges[mask]
        edge_times = edge_times[mask]
        edge_norm = edge_norm[mask]
        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_times, edge_norm)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        all_emb = torch.cat([user_emb, item_emb], dim=0)
        # heterogenous neighbor generation
        neigh_emb = []
        for ns in neighbors:
            ns = torch.LongTensor(ns).to(args.device)
            neigh_emb.append(all_emb[ns].mean(dim=0))
        neigh_emb = torch.stack(neigh_emb)
        masked_node_emb = all_emb[masked_nodes]
        recon_loss = args.het_lmd * -torch.log(1e-10 + torch.sigmoid(torch.mul(masked_node_emb, neigh_emb).sum(1))).mean()
        # recon_loss = torch.zeros(1).to(args.device)

        # homogenous neighbor generation
        h_ii_edges_sampled = edge_sample(self.ii_adj)
        h_uu_edges_sampled = edge_sample(self.uu_adj)
        pos_pairs = torch.cat([item_emb[h_ii_edges_sampled], user_emb[h_uu_edges_sampled]], dim=0)
        pos_score = torch.mul(pos_pairs[:, 0], pos_pairs[:, 1]).sum(1)
        random_negs = torch.randint(self.num_users+self.num_items, (pos_pairs.shape[0],))
        neg_score = torch.mul(pos_pairs[:, 0], all_emb[random_negs]).sum(1)
        homo_loss = args.hom_lmd * self._infonce_loss(pos_score, neg_score.unsqueeze(1), tau=1.0)
        # homo_loss = torch.zeros(1).to(args.device)

        loss = rec_loss + recon_loss + homo_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "recon_loss": recon_loss.item(),
            "homo_loss": homo_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict
    
    @torch.no_grad()
    def generate(self):
        return self.forward(self.edges, self.edge_times)
    
    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())
    
    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embedding[users]
        pos_i_emb = self.item_embedding[pos_items]
        neg_i_emb = self.item_embedding[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
                          pos_i_emb.norm(2).pow(2) +
                          neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss
