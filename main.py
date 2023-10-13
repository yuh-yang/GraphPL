import pandas as pd
import setproctitle
import importlib
from utils.trainer import Trainer
from utils.logger import Logger, log_exceptions
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from utils.dataloader import EdgeListData
from utils.parse_args import args
from os import path
import datetime
import sys
from copy import deepcopy
sys.path.append('./')

# import dgl

modules_class = 'modules.'
if args.plugin:
    modules_class = 'modules.plugins.'


def import_model():
    module = importlib.import_module(modules_class + args.model)
    return getattr(module, args.model)


def import_pretrained_model():
    module = importlib.import_module(modules_class + args.pre_model)
    return getattr(module, args.pre_model)


def import_finetune_model():
    module = importlib.import_module(modules_class + args.f_model)
    return getattr(module, args.f_model)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def merge_pd(pds):
    for i in range(len(pds)):
        if i == 0:
            merged = pds[i]
        else:
            merged = pd.merge(merged, pds[i], on=["user"], how="left")

            merged.loc[merged["item_y"].notna(), "item_x"] = merged.loc[merged["item_y"].notna(
            ), "item_x"] + " " + merged.loc[merged["item_y"].notna(), "item_y"]
            merged.drop(columns=["item_y"], inplace=True)

            merged.loc[merged["time_y"].notna(), "time_x"] = merged.loc[merged["time_y"].notna(
            ), "time_x"] + " " + merged.loc[merged["time_y"].notna(), "time_y"]
            merged.drop(columns=["time_y"], inplace=True)
            merged.rename(columns={"item_x": "item",
                          "time_x": "time"}, inplace=True)
    return merged


init_seed(args.seed)
logger = Logger(args)

pretrain_data = path.join(args.data_path, "pretrain.txt")
pretrain_val_data = path.join(args.data_path, "pretrain_val.txt")
finetune_data = path.join(args.data_path, "fine_tune.txt")
test_data = path.join(args.data_path, "test_1.txt")
test_2_data = path.join(args.data_path, "test_2.txt")
test_3_data = path.join(args.data_path, "test_3.txt")
test_4_data = path.join(args.data_path, "test_4.txt")
test_data_num = 8 if args.data_path.split('/')[-1] == 'amazon_w' else 7 if args.data_path.split('/')[-1] == 'wechat' else 4
logger.log(f"test_data_num: {test_data_num}")
test_datas = [
    path.join(args.data_path, f"test_{i}.txt") for i in range(1, test_data_num+1)]
all_data = [pretrain_data, finetune_data, *test_datas]

recalls, ndcgs = [], []

if args.phase == "finetune":
    @log_exceptions
    def run():
        state_dict = torch.load(args.pre_model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(s) for s in ["user_embedding", "item_embedding", "gating_weight", "gating_bias"]):
                new_state_dict[k] = v

        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        for num_stage in range(1, len(test_datas)+1):
            # starts with test_data, which equals 3
            test_data_idx = num_stage + 2
            ft_data_idx = test_data_idx - 1
            logger.info(
                f"Finetune Stage {num_stage}, test data: {all_data[test_data_idx]}, finetune data {all_data[ft_data_idx]}")

            # use all data to propogate as prompt
            logger.info(
                f"Gen Mode = 2: use all data {all_data[:ft_data_idx]} to propogate as prompt")

            all_data_pd = [pd.read_csv(pretrain_data, sep="\t", names=["user", "item", "time"]),
                           pd.read_csv(pretrain_val_data, sep="\t",
                                       names=["user", "item", "time"]),
                           pd.read_csv(finetune_data, sep="\t", names=[
                                       "user", "item", 'time']),
                           *[pd.read_csv(test_data, sep="\t", names=[
                                       "user", "item", "time"]) for test_data in test_datas],
                           ]
            data_to_merge = all_data_pd[:ft_data_idx]
            merged_pre_pd = merge_pd(data_to_merge)
            # test file here is useless
            pre_dataset = EdgeListData(
                train_file=merged_pre_pd, test_file=all_data_pd[ft_data_idx], has_time=True)

            # 没有gating参数
            pretrained_model = import_pretrained_model()(
                pre_dataset, phase='for_tune').to(args.device)

            pretrained_model.load_state_dict(new_state_dict, strict=True)
            logger.info(f"Successfully loaded: {new_state_dict.keys()}")
            pretrained_model.eval()

            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=path.join(
                args.data_path, f"test_{num_stage}.txt"), phase='finetune', pre_dataset=pre_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])
            model = import_finetune_model()(finetune_dataset, pretrained_model,
                                            "finetune").to(args.device)

            trainer = Trainer(finetune_dataset, logger,
                              pre_dataset=pretrain_dataset)
            best_perform = trainer.train_finetune(model, pretrained_model)

            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])

            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")

    run()

elif args.phase.startswith("vanilla_"):
    @log_exceptions
    def run():
        for num_stage in range(1, len(test_datas)+1):
            test_data_idx = num_stage + 2
            logger.info(
                f"Vanilla Stage {num_stage}, test data: {all_data[test_data_idx]}, train data: {all_data[:test_data_idx]}")

            all_data_pd = [
                pd.read_csv(pretrain_data, sep="\t", names=[
                            "user", "item", "time"]),
                pd.read_csv(pretrain_val_data, sep="\t",
                            names=["user", "item", "time"]),
                pd.read_csv(finetune_data, sep="\t", names=[
                            "user", "item", 'time']),
                *[pd.read_csv(test_data, sep="\t", names=[
                            "user", "item", "time"]) for test_data in test_datas],
            ]
            data_to_merge = all_data_pd[:test_data_idx]

            merged_train_pd = merge_pd(data_to_merge)
            print(merged_train_pd.iloc[1])

            edgelist_dataset = EdgeListData(
                merged_train_pd, all_data_pd[test_data_idx], has_time=False)

            model = import_model()(edgelist_dataset, phase="vanilla").to(args.device)

            trainer = Trainer(edgelist_dataset, logger)
            best_perform = trainer.train(model)

            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])

            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')
            
        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")

    run()

elif args.phase == "incr":
    @log_exceptions
    def run():
        state_dict = torch.load(args.pre_model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(s) for s in ["user_embedding", "item_embedding", "gating_weight", "gating_bias"]):
                new_state_dict[k] = v

        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)
        pre_model = import_pretrained_model()(
            pretrain_dataset, phase='vanilla').to(args.device)
        pre_model.load_state_dict(new_state_dict)
        pre_model.eval()

        for num_stage in range(1, len(test_datas)+1):
            test_data_idx = num_stage + 2
            ft_data_idx = test_data_idx - 1
            logger.info(
                f"Incremental Learning Stage {num_stage}, test data: {all_data[test_data_idx]}, incremental train data: {all_data[ft_data_idx]}")

            last_train_data_idx = ft_data_idx-1
            last_test_data_idx = ft_data_idx
            pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)
            if num_stage == 1:
                pre_dataset = pretrain_dataset
            else:
                pre_dataset = EdgeListData(train_file=all_data[last_train_data_idx], test_file=all_data[last_test_data_idx],
                                        phase='finetune', pre_dataset=pretrain_dataset, has_time=True, user_hist_files=all_data[:last_train_data_idx])

            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=path.join(
                args.data_path, f"test_{num_stage}.txt"), phase='finetune', pre_dataset=pre_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])

            model = import_finetune_model()(finetune_dataset, pre_model,
                                            phase='incremental').to(args.device)

            trainer = Trainer(finetune_dataset, logger,
                            pre_dataset=pretrain_dataset)
            
            best_perform = trainer.train_finetune(model, pre_model)

            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])

            pre_model = deepcopy(model)
            pre_model.eval()
            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")

    run()

if args.phase == "inter_finetune":
    @log_exceptions
    def run():
        saved_model_paths = []

        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        for num_stage in range(1, len(test_datas)+1):
            interval = args.updt_inter
            if len(saved_model_paths) >= interval:
                all_state_dict = [torch.load(args.pre_model_path, map_location=args.device)]
                for i in range(interval):
                    all_state_dict.append(torch.load(saved_model_paths[-i-1], map_location=args.device))
                
                # inter_weight = 0.5 * F.softmax(-1 * torch.arange(1, interval+1).float(), dim=0)
                pretrain_weight = 0.5
                inter_weight = (1-pretrain_weight) * F.normalize(torch.arange(1, interval+1).float(), dim=0, p=1).flip(dims=[0])
                inter_weight = torch.cat([torch.tensor([pretrain_weight]), inter_weight]).unsqueeze(1).unsqueeze(1).to(args.device)
                state_dict = {}
                for k in all_state_dict[0].keys():
                    if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
                        state_dict[k] = torch.sum(torch.stack([sd[k] for sd in all_state_dict]) * inter_weight, dim=0)
            else:
                state_dict = torch.load(args.pre_model_path)

            new_state_dict = {}
            for k, v in state_dict.items():
                if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
                    new_state_dict[k] = v

            # starts with test_data, which equals 2
            test_data_idx = num_stage + 1
            ft_data_idx = test_data_idx - 1
            logger.info(
                f"Finetune Stage {num_stage}, test data: {all_data[test_data_idx]}, finetune data {all_data[ft_data_idx]}")

            # use all data to propogate as prompt
            logger.info(
                f"Gen Mode = 2: use all data {all_data[:ft_data_idx]} to propogate as prompt")

            all_data_pd = [pd.read_csv(pretrain_data, sep="\t", names=["user", "item", "time"]),
                           pd.read_csv(finetune_data, sep="\t", names=[
                                       "user", "item", 'time']),
                           *[pd.read_csv(test_data, sep="\t", names=[
                                       "user", "item", "time"]) for test_data in test_datas],
                           ]
            data_to_merge = all_data_pd[:ft_data_idx]
            merged_pre_pd = merge_pd(data_to_merge)
            # test file here is useless
            pre_dataset = EdgeListData(
                train_file=merged_pre_pd, test_file=all_data_pd[ft_data_idx], has_time=True)

            # 没有gating参数
            pretrained_model = import_pretrained_model()(
                pre_dataset, phase='for_tune').to(args.device)

            pretrained_model.load_state_dict(new_state_dict, strict=True)
            logger.info(f"Successfully loaded: {new_state_dict.keys()}")
            pretrained_model.eval()

            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=path.join(
                args.data_path, f"test_{num_stage}.txt"), phase='finetune', pre_dataset=pre_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])
            model = import_finetune_model()(finetune_dataset, pretrained_model,
                                            "finetune").to(args.device)

            trainer = Trainer(finetune_dataset, logger,
                              pre_dataset=pretrain_dataset)
            best_perform = trainer.train_finetune(model, pretrained_model)

            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])

            saved_model_paths.append(trainer.save_path)
            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")

    run()


if args.phase == "inter_finetune_ab_no_prpt":
    @log_exceptions
    def run():
        saved_model_paths = []

        pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

        for num_stage in range(1, len(test_datas)+1):
            interval = args.updt_inter
            if len(saved_model_paths) >= interval:
                all_state_dict = [torch.load(args.pre_model_path, map_location=args.device), torch.load(saved_model_paths[-interval], map_location=args.device)]
                state_dict = {}
                for k in all_state_dict[0].keys():
                    if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
                        state_dict[k] = torch.mean(
                            torch.stack([sd[k] for sd in all_state_dict]), dim=0)
            else:
                state_dict = torch.load(args.pre_model_path)

            new_state_dict = {}
            for k, v in state_dict.items():
                if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
                    new_state_dict[k] = v

            # starts with test_data, which equals 3
            test_data_idx = num_stage + 2
            ft_data_idx = test_data_idx - 1
            logger.info(
                f"Finetune Stage {num_stage}, test data: {all_data[test_data_idx]}, finetune data {all_data[ft_data_idx]}")

            # logger.info(f"Gen Mode = 1: only propogate on pretrain data")
            # pre_dataset = pretrain_dataset

            logger.info(f"Gen Mode = 0: only propogate on last training data")
            last_train_data_idx = ft_data_idx-1
            last_test_data_idx = ft_data_idx
            pre_dataset = EdgeListData(train_file=all_data[last_train_data_idx], test_file=all_data[last_test_data_idx], phase='finetune', pre_dataset=pretrain_dataset, has_time=True, user_hist_files=all_data[:last_train_data_idx])
            
            # 没有gating参数
            pretrained_model = import_pretrained_model()(
                pre_dataset, phase='for_tune').to(args.device)

            pretrained_model.load_state_dict(new_state_dict, strict=True)
            logger.info(f"Successfully loaded: {new_state_dict.keys()}")
            pretrained_model.eval()

            finetune_dataset = EdgeListData(train_file=all_data[ft_data_idx], test_file=path.join(
                args.data_path, f"test_{num_stage}.txt"), phase='finetune', pre_dataset=pre_dataset, has_time=True, user_hist_files=all_data[:ft_data_idx])
            model = import_finetune_model()(finetune_dataset, pretrained_model,
                                            "finetune").to(args.device)

            trainer = Trainer(finetune_dataset, logger,
                              pre_dataset=pretrain_dataset)
            best_perform = trainer.train_finetune(model, pretrained_model)

            recalls.append(best_perform['recall'][0])
            ndcgs.append(best_perform['ndcg'][0])

            saved_model_paths.append(trainer.save_path)
            # update exp time for saving new model
            args.exp_time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

        logger.info(f"recalls: {recalls} \n ndcgs: {ndcgs} \n avg. recall: {np.round(np.mean(recalls), 4)}, avg. ndcg: {np.round(np.mean(ndcgs), 4)}")

    run()