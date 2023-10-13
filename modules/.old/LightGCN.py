import torch as t
from torch import nn
import torch.nn.functional as F
from utils.parse_args import args
from modules.base_model import BaseModel
from modules.utils import SpAdjEdgeDrop


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCN(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN, self).__init__(data_handler)
        self.adj = self._make_binorm_adj(data_handler.graph)
        self.layer_num = 3
        self.reg_weight = 1e-4
        self.keep_rate = 0.5
        self.user_embeds = nn.Parameter(
            init(t.empty(self.num_users, self.emb_size)))
        self.item_embeds = nn.Parameter(
            init(t.empty(self.num_items, self.emb_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.is_training = True

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def forward(self, adj, keep_rate):
        if not self.is_training:
            return self.final_embeds[:self.num_users], self.final_embeds[self.num_users:]
        embeds = t.cat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)  # / len(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.num_users], embeds[self.num_users:]

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = self._bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        reg_loss = self.reg_weight * self._reg_loss(ancs, poss, negs)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    @t.no_grad()
    def generate(self):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        return user_embeds, item_embeds

    @t.no_grad()
    def rating(self, user_emb, item_emb):
        return t.matmul(user_emb, item_emb.t())

    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embeds[users]
        pos_i_emb = self.item_embeds[pos_items]
        neg_i_emb = self.item_embeds[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
                          pos_i_emb.norm(2).pow(2) +
                          neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss
