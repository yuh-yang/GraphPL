import torch
import torch.nn as nn
from modules.base_model import BaseModel
import logging
from utils.parse_args import args
from torch_scatter import scatter_softmax

init = nn.init.xavier_uniform_
logger = logging.getLogger('train_logger')

class BasePluginModel(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()
        
        self.phase = phase

        if self.phase not in ['vanilla', 'incremental']:
            self.edge_times = [dataset.edge_time_dict[e[0]][e[1]] for e in self.edges.cpu().tolist()]
            self.edge_times = torch.LongTensor(self.edge_times).to(args.device)
            logger.info(f"Using Time Encoding. Max Time Step: {self.edge_times.max()}")
        else:
            self.edge_times = None

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

            self.emb_dropout = nn.Dropout(args.emb_dropout)

            self.emb_gate = lambda x: self.emb_dropout(torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias)))
        
        elif self.phase == 'continue_tune':
            # re-initialize for loading state dict
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

            self.gating_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.gating_bias = nn.Parameter(init(torch.empty(1, args.emb_size)))

            self.emb_gate = lambda x: torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias))
            
        elif self.phase == 'incremental':
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)
            self.emb_gate = lambda x: x
    
    def _relative_edge_time_encoding(self, edges, edge_times):
        # for each node, normalize edge_times according to its neighbors
        # edge_times: [E]
        # rescal to 0-1
        edge_times = edge_times.float()
        edge_times = (edge_times - edge_times.min()) / (edge_times.max() - edge_times.min())
        # edge_times = torch.sigmoid(edge_times)
        dst_nodes = edges[:, 1]
        time_norm = scatter_softmax(edge_times, dst_nodes, dim_size=self.num_users+self.num_items)
        return time_norm
