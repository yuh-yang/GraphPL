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

def cal_infonce(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(cl_loss)

def cal_uniformity(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    cl_loss = -torch.log(1 / pos_score+10e-6)
    return torch.mean(cl_loss)

class FT_2(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.phase = phase

        if self.phase == 'pretrain' or self.phase == 'vanilla':
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

            self.emb_gate = lambda x: x

        elif self.phase == 'finetune':
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

            self.user_emb_new = nn.Parameter(init(torch.empty(self.num_users, 16)))
            self.item_emb_new = nn.Parameter(init(torch.empty(self.num_items, 16)))
            self.new_emb_mlp = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, args.emb_size),
            )
            self.new_emb_gating_weight = nn.Parameter(init(torch.empty(2*args.emb_size, 1)))

            self.gating_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.gating_bias = nn.Parameter(init(torch.empty(1, args.emb_size)))

            self.emb_gate = lambda x: torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias))

        self.edge_dropout = EdgelistDrop()

    def _channel_gate(self, emb_1, emb_2):
        emb = torch.cat([emb_1, emb_2], dim=1)
        gate = torch.sigmoid(torch.matmul(emb, self.new_emb_gating_weight))
        return emb_1 + (1-gate)/gate * emb_2

    def _agg(self, all_emb, edges, edge_norm):
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

    def forward(self, edges, edge_norm, all_emb = None):
        if all_emb is None:
            all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
            all_emb = self.emb_gate(all_emb)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb

    def _cal_variance_loss(self, emb, pre_emb):
        # emb: (n, d)
        # pre_emb: (n, d)
        # var_loss = -torch.mean(torch.norm(emb - pre_emb, dim=1))
        var_loss = cal_uniformity(emb, pre_emb, 0.2)
        return var_loss

    def _cal_align_loss(self, emb, pre_emb):
        # emb: (n, d)
        # pre_emb: (n, d)
        # align_loss = torch.mul(emb, pre_emb).sum(dim=1).mean()
        align_loss = cal_infonce(emb, pre_emb, 1.0)
        return align_loss
    
    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 0.5, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]

        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        # forward with new emb
        user_emb_new, item_emb_new = self.forward(edges, edge_norm, all_emb=self.new_emb_mlp(torch.cat([self.user_emb_new, self.item_emb_new], dim=0)))
        # batch_user_emb_new = self.new_emb_mlp(user_emb_new[users])
        # pos_item_emb_new = self.new_emb_mlp(item_emb_new[pos_items])
        # neg_item_emb_new = self.new_emb_mlp(item_emb_new[neg_items])
        batch_user_emb_new = user_emb_new[users]
        pos_item_emb_new = item_emb_new[pos_items]
        neg_item_emb_new = item_emb_new[neg_items]

        batch_user_emb = self._channel_gate(batch_user_emb, batch_user_emb_new)
        pos_item_emb = self._channel_gate(pos_item_emb, pos_item_emb_new)
        neg_item_emb = self._channel_gate(neg_item_emb, neg_item_emb_new)

        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
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
