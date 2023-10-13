import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from utils.parse_args import args
from modules.base_model import BaseModel
from modules.utils import SpAdjEdgeDrop
import scipy.sparse as sp


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

def mask_adj(adj, mask):
    vals = adj._values()
    idxs = adj._indices()
    newVals = vals[mask]  # / keep_rate
    newIdxs = idxs[:, mask]
    return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

def edge_sample(adj: sp.coo_matrix):
    sampled_edges = np.random.choice(adj.nnz, 500)
    sampled_edges = np.vstack((adj.row[sampled_edges], adj.col[sampled_edges])).T
    return torch.LongTensor(sampled_edges).to(args.device)

def node_mask_symmetric(adj):
    # convert torch sparse tensor to scipy sparse matrix
    adj = adj.coalesce()
    adj = sp.coo_matrix((adj._values().cpu().numpy(), (adj._indices()[0].cpu().numpy(), adj._indices()[1].cpu().numpy())), shape=adj.shape).tolil()
    # random select 500 nodes from row
    selected_nodes = np.random.randint(0, adj.shape[0], 500)
    sampled_nodes = []
    masked_neighbors = []
    for node in selected_nodes:
        i_neighbors = adj.rows[node]
        if len(i_neighbors) < 2:
            continue
        i_masked = list(np.random.choice(list(i_neighbors), len(i_neighbors)//2, replace=False))
        masked_neighbors.append(i_masked)
        sampled_nodes.append(node)
        # mask the edges in adj
        adj[node, i_masked] = 0
        # this is the symmetric part
        adj[i_masked, node] = 0
    # convert back to torch sparse tensor
    adj = adj.tocoo()
    masked_adj = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((adj.row, adj.col))), torch.FloatTensor(adj.data), adj.shape).to(args.device)
    return torch.LongTensor(sampled_nodes).to(args.device), masked_neighbors, masked_adj

def node_mask_partial(adj, num_users, num_items):
    edgelist = adj._indices().t()
    # ALREADY remapped item_id to [num_users, num_users + num_items]
    n_edgelist = torch.stack([edgelist[:, 0], edgelist[:, 1]], dim=1)
    nodes = np.random.randint(0, num_users+num_items, 500)
    mask = torch.ones(n_edgelist.shape[0], dtype=torch.bool).to(args.device)
    neighbors = []
    sampled_nodes = []
    for node in nodes:
        i_edges_src = n_edgelist[:, 0] == node
        # make sure the node has at least 2 neighbor, and the node is not dropped
        if i_edges_src.sum() < 2:
            continue
        sampled_nodes.append(node)
        i_edges = i_edges_src
        position = 1
        neigh_idx = i_edges.nonzero().squeeze()
        masked_negih_idx = neigh_idx[torch.randperm(neigh_idx.size(0))][:int(neigh_idx.size(0)/2)]        
        neighbor_ids = n_edgelist[masked_negih_idx, position]

        masked_edges_dst = n_edgelist[:, 1] == node & n_edgelist[:, 0].isin(neighbor_ids)
        masked_edges_dst_idx = masked_edges_dst.nonzero().squeeze()
        mask[masked_negih_idx] = False
        mask[masked_edges_dst_idx] = False
        neighbors.append(neighbor_ids.tolist())
    n_edgelist[:, 1] -= num_users
    return torch.LongTensor(sampled_nodes).to(args.device), neighbors, mask


class LightGCN_GP(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_GP, self).__init__(data_handler)
        self.adj = self._make_binorm_adj(data_handler.graph)
        self.layer_num = 3
        self.reg_weight = 1e-4
        self.keep_rate = 0.5
        self.user_embeds = nn.Parameter(
            init(torch.empty(self.num_users, self.emb_size)))
        self.item_embeds = nn.Parameter(
            init(torch.empty(self.num_items, self.emb_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.is_training = True

        self.uu_adj = data_handler.uu_adj
        self.ii_adj = data_handler.ii_adj


    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def forward(self, adj, keep_rate):
        if not self.is_training:
            return self.final_embeds[:self.num_users], self.final_embeds[self.num_users:]
        embeds = torch.cat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            F.normalize(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list) / len(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.num_users], embeds[self.num_users:]

    def cal_loss(self, batch_data):
        self.is_training = True
        adj = self.edge_dropper(self.adj, self.keep_rate)
        masked_nodes, neighbors, adj = node_mask_symmetric(adj)
        # masked_nodes, neighbors, mask = node_mask_partial(adj, self.num_users, self.num_items)
        # adj = mask_adj(adj, mask)

        user_embeds, item_embeds = self.forward(adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = self._bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        reg_loss = self.reg_weight * self._reg_loss(ancs, poss, negs)

        all_emb = torch.cat([user_embeds, item_embeds], dim=0)
        # heterogenous neighbor generation
        neigh_emb = []
        for ns in neighbors:
            ns = torch.LongTensor(ns).to(args.device)
            neigh_emb.append(all_emb[ns].mean(dim=0))
        neigh_emb = torch.stack(neigh_emb)
        masked_node_emb = all_emb[masked_nodes]
        recon_loss = args.het_lmd * -torch.log(1e-10 + torch.sigmoid(torch.mul(masked_node_emb, neigh_emb).sum(1))).mean()

        # homogenous neighbor generation
        h_ii_edges_sampled = edge_sample(self.ii_adj)
        h_uu_edges_sampled = edge_sample(self.uu_adj)
        pos_pairs = torch.cat([item_embeds[h_ii_edges_sampled], user_embeds[h_uu_edges_sampled]], dim=0)
        pos_score = torch.mul(pos_pairs[:, 0], pos_pairs[:, 1]).sum(1)
        random_negs = torch.randint(self.num_users+self.num_items, (pos_pairs.shape[0],))
        neg_score = torch.mul(pos_pairs[:, 0], all_emb[random_negs]).sum(1)
        homo_loss = args.hom_lmd * self._infonce_loss(pos_score, neg_score.unsqueeze(1), tau=1.0)


        loss = bpr_loss + reg_loss + recon_loss + homo_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'recon_loss': recon_loss, 'homo_loss': homo_loss}
        return loss, losses

    @torch.no_grad()
    def generate(self):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        return user_embeds, item_embeds

    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())

    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embeds[users]
        pos_i_emb = self.item_embeds[pos_items]
        neg_i_emb = self.item_embeds[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
                          pos_i_emb.norm(2).pow(2) +
                          neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss
