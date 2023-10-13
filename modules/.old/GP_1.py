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

def edge_sample(adj: sp.coo_matrix):
    sampled_edges = np.random.choice(adj.nnz, 500)
    sampled_edges = np.vstack((adj.row[sampled_edges], adj.col[sampled_edges])).T
    return torch.LongTensor(sampled_edges).to(args.device)

def node_mask_partial(edgelist, num_users, num_items):
    nodes = np.random.randint(0, num_users+num_items, 500)
    mask = torch.ones(edgelist.shape[0], dtype=torch.bool).to(args.device)
    neighbors = []
    sampled_nodes = []
    for node in nodes:
        i_edges_src = edgelist[:, 0] == node
        # make sure the node has at least 2 neighbor, and the node is not dropped
        if i_edges_src.sum() < 2:
            continue
        sampled_nodes.append(node)
        neighbor_axis = 1
        neigh_idx = i_edges_src.nonzero().squeeze()
        masked_negih_idx = neigh_idx[torch.randperm(neigh_idx.size(0))][:int(neigh_idx.size(0)/2)]
        mask[masked_negih_idx] = False
        # i_edges[masked_negih_idx] = False
        neighbors.append(edgelist[masked_negih_idx, neighbor_axis].tolist())
    return torch.LongTensor(sampled_nodes).to(args.device), neighbors, mask

def all_node_mask(adj, num_users, num_items):
    # random sample 10% nodes
    sampled_nodes = np.random.permutation(num_users + num_items)[:int(0.1*(num_users+num_items))]
    # delete the sampled nodes from sparse adj, and get their neighbors
    n_adj = deepcopy(adj).tocsr()
    n_adj[sampled_nodes] = 0
    n_adj[:, sampled_nodes] = 0
    # get the neighbors of sampled nodes
    adj = adj.tolil()
    neighbors = []
    for node in sampled_nodes:
        neigh_idx = adj.rows[node]
        neighbors.append(neigh_idx.tolist())
    return torch.LongTensor(sampled_nodes).to(args.device), neighbors

def all_node_mask_zero(adj, num_users, num_items):
    # random sample x% nodes
    rdn_nodes = np.random.permutation(num_users + num_items)[:int(0.5*(num_users+num_items))]
    # get the neighbors of sampled nodes
    # adj = adj.tolil()
    # neighbors = []
    # sampled_nodes = []
    # for node in rdn_nodes:
    #     neigh_idx = adj.rows[node]
    #     if len(neigh_idx) == 0:
    #         continue
    #     neighbors.append(neigh_idx)
    #     sampled_nodes.append(node)
    return torch.LongTensor(rdn_nodes).to(args.device), None
    # return torch.LongTensor(sampled_nodes).to(args.device), neighbors

class GP_1(BaseModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()
        # self.edges = torch.LongTensor(dataset.edgelist).to(args.device)
        # self.edge_norm = self._edge_binorm(self.edges)

        self.uu_adj = dataset.uu_adj
        self.ii_adj = dataset.ii_adj


        self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
        self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

        self.edge_dropout = EdgelistDrop()
        self.mse = nn.MSELoss()

    def _agg(self, all_emb, edges, edge_norm):
        # dst_emb = src_emb sum-> dst_idx
        src_emb = all_emb[edges[:, 0]]

        # bi-norm
        src_emb = src_emb * edge_norm.unsqueeze(1)

        # conv
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users+self.num_items)
        return dst_emb
    
    def _edge_binorm(self, edges):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm

    def forward(self, edges, edge_norm, set_zero_indices=None):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        if set_zero_indices is not None:
            all_emb[set_zero_indices] = torch.zeros(self.emb_size).to(args.device)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb
    
    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 0.5, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]

        # # neighbor partial mask
        # masked_nodes, neighbors, mask = node_mask_partial(edges, self.num_users, self.num_items)
        # # apply mask to edges
        # edges = edges[mask]
        # edge_norm = edge_norm[mask]

        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm)
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        if args.ab in ["homo", "full"]:
            # homogenous neighbor generation
            h_ii_edges_sampled = edge_sample(self.ii_adj)
            h_uu_edges_sampled = edge_sample(self.uu_adj)
            pos_pairs = torch.cat([item_emb[h_ii_edges_sampled], user_emb[h_uu_edges_sampled]], dim=0)
            random_negs_i = torch.randint(self.num_items, (h_ii_edges_sampled.shape[0], 100))
            random_negs_u = torch.randint(self.num_users, (h_uu_edges_sampled.shape[0], 100))
            # [num_pos, num_negs, emb_size]
            neg_samples = torch.cat([item_emb[random_negs_i], user_emb[random_negs_u]], dim=0)
            homo_loss = args.hom_lmd * self._infonce_loss(pos_pairs[:, 0], pos_pairs[:, 1], neg_samples, tau=0.3)
        else:
            homo_loss = torch.zeros((1,), device=args.device)

        if args.ab in ["heter", "full"]:
            # heterogenous neighbor generation
            # neighbor partial mask
            adj = sp.coo_matrix((edge_norm.cpu().numpy(), (edges[:,0].cpu().numpy(), edges[:,1].cpu().numpy())), shape=(self.num_users+self.num_items, self.num_users+self.num_items))
            masked_nodes, neighbors = all_node_mask_zero(adj, self.num_users, self.num_items)
            # masked_nodes, neighbors, m_edges, m_edge_norm = all_node_mask(adj, self.num_users, self.num_items)

            user_emb_m, item_emb_m = self.forward(edges, edge_norm, set_zero_indices=masked_nodes)
            all_emb_m = torch.cat([user_emb_m, item_emb_m], dim=0)

            # neigh_emb = []
            # for ns in neighbors:
            #     ns = torch.LongTensor(ns).to(args.device)
            #     neigh_emb.append(all_emb_m[ns].mean(dim=0))
            # neigh_emb = torch.stack(neigh_emb)
            # here use back the first forward node embedding as the reconstruction target
            masked_node_emb = all_emb[masked_nodes]
            masked_node_emb_recon = all_emb_m[masked_nodes]
            # recon_loss = args.het_lmd * self.mse(masked_node_emb, masked_node_emb_recon)
            recon_loss = args.het_lmd * -torch.log(1e-10 + torch.sigmoid(torch.mul(masked_node_emb, masked_node_emb_recon).sum(1))).mean()
        else:
            recon_loss = torch.zeros((1,), device=args.device)
        
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
        return self.forward(self.edges, self.edge_norm)
    
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
