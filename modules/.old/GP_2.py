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

def all_node_mask_zero(adj, num_users, num_items):
    # random sample x% nodes
    rdn_nodes = np.random.permutation(num_users + num_items)[:int(0.5*(num_users+num_items))]
    return torch.LongTensor(rdn_nodes).to(args.device), None

def edge_mask(edges, edge_norm, mask_rate):
    # random sample x% edges
    rdn_edges = np.random.permutation(edges.shape[0])[:int(mask_rate*edges.shape[0])]
    mask = torch.ones_like(edge_norm).bool()
    mask[rdn_edges] = False
    return edges[mask], edge_norm[mask], edges[~mask]


class GP_2(BaseModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()
        # self.edges = torch.LongTensor(dataset.edgelist).to(args.device)
        # self.edge_norm = self._edge_binorm(self.edges)

        # self.uu_adj = dataset.uu_adj
        # self.ii_adj = dataset.ii_adj

        self.MLP = nn.Sequential(
            nn.Linear(args.emb_size, args.emb_size, bias=True),
            nn.PReLU(),
            nn.Linear(args.emb_size, args.emb_size, bias=True),
            nn.Sigmoid(),
        )

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

    def forward(self, edges, edge_norm, num_layers=None):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        res_emb = [all_emb]
        if num_layers is None:
            num_layers = args.num_layers
        for l in range(num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb
    
    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 0.5, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]

        # mask before forward
        # edges, edge_norm, masked_edges = edge_mask(edges, edge_norm)

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
            n_edges, n_edge_norm, masked_edges = edge_mask(edges, edge_norm, args.mask_ratio)

            user_emb_m, item_emb_m = self.forward(n_edges, n_edge_norm, num_layers=1)
            all_emb_m = torch.cat([user_emb_m, item_emb_m], dim=0)

            # here use back the first forward node embedding as the reconstruction target
            pos_u = all_emb_m[masked_edges[:, 0]]
            pos_v = all_emb_m[masked_edges[:, 1]]
            neg_v = all_emb_m[torch.randint(all_emb_m.size(0), [masked_edges.size(0), 1])]
            recon_loss = args.het_lmd * self._infonce_loss(pos_u, pos_v, neg_v, tau=0.5)
            # pos_u = all_emb[masked_edges[:, 0]]
            # pos_v = all_emb[masked_edges[:, 1]]
            # recon_loss = args.het_lmd * -torch.log(1e-10 + torch.sigmoid(torch.mul(pos_u, pos_v).sum(1))).mean()
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
