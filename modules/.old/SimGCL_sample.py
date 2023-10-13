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
import dgl
from tqdm import tqdm
import time

init = nn.init.xavier_uniform_

class SimGCL_sample(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset)
        self.dataset = dataset


        self.phase = phase

        self.eps = args.eps
        self.lbd = args.lbd

        if self.phase == 'pretrain' or self.phase == 'vanilla':
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

            self.emb_gate = lambda x: x

        elif self.phase == 'finetune':
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

            self.gating_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.gating_bias = nn.Parameter(init(torch.empty(1, args.emb_size)))

            self.emb_gate = lambda x: torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias))
        
        elif self.phase == 'justtune':
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)
            self.emb_gate = lambda x: x

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]

        # bi-norm
        src_emb = src_emb * edge_norm.unsqueeze(1)

        # conv
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=all_emb.shape[0])
        return dst_emb
    
    def _edge_binorm(self, edges, nodes):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=nodes.shape[0])
        user_degs = user_degs[edges[:, 0]]
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=nodes.shape[0])
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm

    def forward(self, edges, sampled_nids, perturbed=False):
        edge_norm = self._edge_binorm(edges, sampled_nids)
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)[sampled_nids]
        res_emb = []
        for l in range(args.num_layers):
            all_emb = self._agg(all_emb, edges, edge_norm)
            if perturbed:
                random_noise = torch.rand_like(all_emb).to(all_emb.device)
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        return res_emb
    
    def forward_sampled_graph(self, edges, sample_nids):
        return self.forward(edges, sample_nids)
    
    def cal_loss(self, batch_data):
        # forward
        users, pos_items, neg_items, [nodes, edges, sample_nids, node_mapping] = batch_data
        res_emb = self.forward_sampled_graph(edges, sample_nids)
        batch_user_emb = res_emb[node_mapping[:len(users)]]
        pos_item_emb = res_emb[node_mapping[len(users):len(users)+len(pos_items)]]
        neg_item_emb = res_emb[node_mapping[len(users)+len(pos_items):]]
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
        all_nodes = torch.arange(self.num_users+self.num_items).to(args.device)
        # iterate all nodes with mini batch
        batch_size = args.eval_batch_size
        s = 0
        res = torch.zeros(self.num_users+self.num_items, self.emb_size).to(args.device)
        pbar = tqdm(total=self.num_users+self.num_items // batch_size + 1)
        while s <= self.num_users+self.num_items:
            e = s + batch_size
            batch_nodes = all_nodes[s:e]
            # time_1 = time.time()
            sample_out = self.dataset.dgl_sample(self.dataset.graph_dgl, [5,5,5], batch_nodes)
            # time_3 = time.time()
            sample_nids = sample_out.ndata[dgl.NID]
            edges = torch.stack(sample_out.edges(), dim=1)
            nodes_list = sample_out.nodes().cpu().tolist()
            nids_list = sample_nids.cpu().tolist()
            node_mapping = {nids_list[i]:nodes_list[i] for i in range(len(nodes_list))}
            node_mapping = [node_mapping[i] for i in batch_nodes.cpu().tolist()]
            # time_2 = time.time()
            # print(f"sampling time: {time_3-time_1}, postprocessing time: {time_2-time_3}")
            res_emb = self.forward_sampled_graph(edges, sample_nids)
            res[s:e] = res_emb[node_mapping]
            s += batch_size
            pbar.update(1)
        return res.split([self.num_users, self.num_items], dim=0)

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
