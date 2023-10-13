from utils.parse_args import args
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import Metric
from os import path
import numpy as np
from utils.logger import log_exceptions
import time

class Trainer(object):
    def __init__(self, dataset, logger, pre_dataset=None):
        self.dataloader = dataset
        self.pre_dataloader = pre_dataset
        self.metric = Metric()
        self.logger = logger

    def create_optimizer(self, model):
        self.optimizer = optim.Adam(model.parameters(
        ), lr=args.lr)

    def train_epoch(self, model, epoch_idx):
        self.dataloader.shuffle()
        # for recording loss
        s = 0
        loss_log_dict = {}
        ep_loss = 0
        time_1 = time.time()
        # start this epoch
        model.train()
        pbar = tqdm(total=self.dataloader.num_edges // args.batch_size + 1)
        while s + args.batch_size <= self.dataloader.num_edges:
            batch_data = self.dataloader.get_train_batch(s, s+args.batch_size)
            self.optimizer.zero_grad()
            loss, loss_dict = model.cal_loss(batch_data)
            loss.backward()
            self.optimizer.step()
            ep_loss += loss.item()
            # print(loss.item())

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(
                    loss_dict[loss_name]) / (self.dataloader.num_edges // args.batch_size + 1)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

            # print(loss.item())

            s += args.batch_size
            pbar.update(1)
            if self.stop_flag:
                break
        time_2 = time.time()
        loss_log_dict['train_time'] = round(time_2 - time_1, 2)
        # log
        self.logger.log_loss(epoch_idx, loss_log_dict)

    @log_exceptions
    def train(self, model):
        # self.output_emb(model)
        # self.evaluate(model, 0, self.dataloader)
        self.create_optimizer(model)
        self.best_perform = {'recall': [0.], 'ndcg': [0.]}
        self.stop_counter = 0
        self.stop_flag = False
        # self.evaluate(model, 0)
        for epoch_idx in range(args.num_epochs):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            self.evaluate(model, epoch_idx, self.dataloader)
            if self.stop_flag:
                break

    @log_exceptions
    def train_finetune(self, model, pre_model):
        # verify pretrain model performances
        pre_model.eval()
        # self.logger.log("Testing Pretrained Model on Pretrain Dataset...")
        # self.logger.log_eval(self.metric.eval(pre_model, self.pre_dataloader), self.metric.k)

        # self.logger.log("Testing Pretrained Model on Test Dataset...")
        # self.logger.log_eval(self.metric.eval(pre_model, self.dataloader), self.metric.k)

        self.create_optimizer(model)
        self.best_perform = {'recall': [0.], 'ndcg': [0.]}
        self.stop_counter = 0
        self.stop_flag = False
        # self.evaluate(model, 0)
        for epoch_idx in range(args.num_epochs):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            self.evaluate(model, epoch_idx, self.dataloader)
            if self.stop_flag:
                break
        return self.best_perform

    def evaluate(self, model, epoch_idx, dataloader):
        model.eval()
        eval_result = self.metric.eval(model, dataloader)
        self.logger.log_eval(eval_result, self.metric.k)
        perform = eval_result['recall'][0]
        if perform > self.best_perform['recall'][0]:
            self.best_perform = eval_result
            self.logger.log('Find better model at epoch: {}: recall={}'.format(
                epoch_idx, self.best_perform['recall'][0]))
            # self.output_emb(model)
            # self.logger.log('Embedding saved!')
            if args.log:
                self.save_model(model)
                self.logger.log('Model saved!')
            self.stop_counter = 0
        else:
            self.stop_counter += 1
            if self.stop_counter >= args.early_stop_patience:
                self.logger.log('Early stop!')
                self.logger.log(f"Best performance: recall={self.best_perform['recall'][0]}, ndcg={self.best_perform['ndcg'][0]}")
                self.stop_flag = True
        model.train()

    # @torch.no_grad()
    # def output_emb(self, model):
    #     emb_dict = {}
    #     output_path = path.join(args.save_dir, 'ouput_embeddings')

    #     sampler = dgl.dataloading.NeighborSampler([args.neighbor_sample_num, args.neighbor_sample_num])
    #     gen_dataloader = dgl.dataloading.DataLoader(
    #         self.graph, torch.arange(self.graph.number_of_nodes(), device=args.device,dtype=torch.int32), sampler,
    #         batch_size=128,
    #         shuffle=False,
    #         drop_last=False,
    #         num_workers=0,
    #         device=args.device)

    #     for _, (input_nodes, output_nodes, mfgs) in enumerate(gen_dataloader):
    #         # predict result
    #         batch_emb = model.gen_emb((input_nodes, output_nodes, mfgs))

    #         output_nodes = output_nodes.cpu().tolist()
    #         for i in range(len(output_nodes)):
    #             emb_dict[self.node_id_mapping[output_nodes[i]]] = batch_emb[i].cpu().numpy()

    #     with open(output_path, 'w') as f:
    #         for node in emb_dict:
    #             f.write(str(node) + '\t' + np.array2string(emb_dict[node], max_line_width=1e8, separator='#') + '\n')

    def save_model(self, model):
        self.save_path = path.join(args.save_dir, f'saved_model_{args.exp_time}.pt')
        torch.save(model.state_dict(), self.save_path)
        pass
