import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='GNN Pretraining')
    parser.add_argument('--phase', type=str, default='pretrain')
    parser.add_argument('--plugin', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default="saved" ,help='where to save model and logs')
    parser.add_argument('--data_path', type=str, default="dataset/yelp",help='where to load data')
    parser.add_argument('--exp_name', type=str, default='1')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--ab', type=str, default='full')
    parser.add_argument('--log', type=int, default=1)

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model', type=str, default='GP')
    parser.add_argument('--pre_model', type=str, default='LightGCN')
    parser.add_argument('--f_model', type=str, default='GF_1')
    parser.add_argument('--pre_model_path', type=str, default='saved/lightgcn_edge/saved_model.pt')

    # parser.add_argument('--var_coef', type=float, default=0.01)
    # parser.add_argument('--align_coef', type=float, default=0.001)

    # parser.add_argument('--het_lmd', type=float, default=1.0)
    # parser.add_argument('--het_temp', type=float, default=1.0)
    # parser.add_argument('--hom_lmd', type=float, default=0.1)
    # parser.add_argument('--hom_temp', type=float, default=0.5)
    # parser.add_argument('--mask_ratio', type=float, default=0.5)
    # parser.add_argument('--sample_size', type=int, default=500)

    parser.add_argument('--hour_interval_pre', type=float, default=1)
    parser.add_argument('--hour_interval_f', type=int, default=1)
    parser.add_argument('--emb_dropout', type=float, default=0)
    # 0: 从上一个微调图生成; 1: 从原始pretrain图生成； 2: 从拼接图生成
    parser.add_argument('--gen_mode', type=int, default=0)
    parser.add_argument('--updt_inter', type=int, default=1)
    
    parser.add_argument('--edge_dropout', type=float, default=0.5)
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--neighbor_sample_num', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--metrics', type=str, default='recall;ndcg')
    parser.add_argument('--metrics_k', type=str, default='20')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=1)

    parser.add_argument('--num_layers', type=int, default=3)


    return parser

def parse_args_simgcl(parser):
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--lbd', type=float, default=0.5)

    return parser

def parse_args_sgl(parser):
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--lbd', type=float, default=0.1)

    return parser

def parse_args_mixgcf(parser):
    parser.add_argument('--n_negs', type=int, default=16)
    return parser

def parse_args_dau(parser):
    parser.add_argument('--gamma', type=float, default=1)
    return parser

def parse_args_duo_emb(parser):
    parser.add_argument('--last_model_path', type=str, default=None)
    return parser

parser = parse_args()
args = parser.parse_known_args()[0]
if args.pre_model == args.f_model:
    args.model = args.pre_model
elif args.pre_model != 'LightGCN':
    args.model = args.pre_model

if args.model == 'SimGCL':
    parser = parse_args_simgcl(parser)
elif args.model == 'SGL':
    parser = parse_args_sgl(parser)
elif args.model == 'MixGCF':
    parser = parse_args_mixgcf(parser)
elif args.model == 'DirectAU':
    parser = parse_args_dau(parser)

if args.phase.startswith("duo_emb_"):
    parser = parse_args_duo_emb(parser)

args = parser.parse_args()
if args.pre_model == args.f_model:
    args.model = args.pre_model
elif args.pre_model != 'LightGCN':
    args.model = args.pre_model