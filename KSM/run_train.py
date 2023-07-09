'''
 run 'torchrun --standalone --nnodes=1 --nproc_per_node=2 run_train.py' to start training process with 2 GPUs
'''

import sys
sys.path.append('..')

from src.utils import set_random_seed
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import os
import random
from src.data.collator import Collator_fn, MoleculeDataset
from src.model.ksm import KSGNN
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import Trainer
from src.trainer.result_tracker import Result_Tracker
import warnings
import pandas as pd
import json
from src.trainer.metrics import Evaluate

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12312'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
warnings.filterwarnings("ignore")

metric = Evaluate()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def cal_final_results(args):
    res = pd.read_csv(args.save_dir + '/results.csv')
    metrics = ['rmse', 'mae', 'sd', 'r']
    res_ls = {m: [] for m in metrics}
    try:
        for m in metrics:
            res_ls[m] = [float(res[res['repeat'] == i][m].values[0]) for i in range(args.n_repeat)]
    except:
        for m in metrics:
            res_ls[m] = [float(res[res['repeat'] == str(i)][m].values[0]) for i in range(args.n_repeat)]
    avg_res = [round(np.mean(res_ls[m]), 4) for m in metrics]
    std_res = [round(np.std(res_ls[m]), 4) for m in metrics]
    res.loc[len(res)] = ['avg'] + avg_res
    res.loc[len(res)] = ['std'] + std_res
    res.to_csv(args.save_dir + '/results.csv', index=False)
    print('Average results: ', avg_res)
    print(args.save_dir)


def load_snapshot(args, model, snapshot_path):
    snapshot = torch.load(snapshot_path, map_location=args.device)
    model.load_state_dict(snapshot["model_state"])
    args.epoch_run = snapshot["epoch_run"]
    try:
        args.step_count = snapshot['step_count']
    except:
        args.step_count = (len(train_loader.dataset) // args.batch_size) * args.epoch_run
    print(f"Resuming training from snapshot at Epoch {args.epoch_run}")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--n_repeat", type=int, default=3)
    parser.add_argument("--dist_train", type=int, default=False, help='Distributed training')
    parser.add_argument("--only_test", type=int, default=False, help='Only test, no training process.')
    ## data path
    parser.add_argument("--dataset", type=str, default='PDBBind2020')
    parser.add_argument('--data_path', default='/media/data2/lm/Experiments/3D_DTI/dataset/', type=str)
    parser.add_argument('--refined_path', default='/original/PDBbind_v2020_refined/refined-set/', type=str, help='refined set path')
    parser.add_argument('--core_path', default='/original/CASF-2016/coreset/', type=str, help='core set path')
    parser.add_argument('--split_type', type=str, default='random_split', help='random split or temporal split')
    parser.add_argument('--processed_dir', type=str, default='./dataset/processed_{}_debug/', help='Preprocessed dataset')
    ## dataset prepare
    parser.add_argument("--smiles_feat", type=str, default='string', help='fingerpring (fp) or string')
    parser.add_argument("--cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--inner_cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--n_angle", type=int, default=6, help='number of angle domains')
    parser.add_argument("--add_fea", type=int, default=0, help='add feature manner, 1, 2, others')
    parser.add_argument("--is_mask", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--p", type=float, default=0.5, help='Mask probability')
    parser.add_argument('--smiles_kernel', type=int, default=3, help='Convolutional kernel size of SMILES')
    parser.add_argument('--prot_kernel', type=int, default=4, help='Convolutional kernel size of protein_sequences')
    parser.add_argument('--l2_norm', type=int, default=0, help='l2 normalization of features in the prediction module')
    parser.add_argument('--ln', type=int, default=0, help='LayerNorm')
    ## model config
    # Basic config
    parser.add_argument("--graph_pool", type=str, default='mean', choices=['sum', 'max', 'mean'], help='Pooling graph node')
    parser.add_argument("--pool", type=str, default="DiffPool", choices=["SAGPool", "DiffPool", None], help='Pooling operation')
    parser.add_argument("--readout", type=str, default='last',
                        choices=['sum', 'max', 'mean', 'concat', 'last', 'linear', 'gru', 'lstm', 'bi-gru', 'bi-lstm'],
                        help='Readout operation for outputs of different layers')
    parser.add_argument("--pool_gnn", type=str, default='gat', choices=['graphsage', 'gcn', 'gat'])
    parser.add_argument("--pool_attn", type=int, default=1, help='pooling coarsened bond nodes with attention')
    parser.add_argument("--gat_ln", type=int, default=0, help='LayerNorm for pool_gnn of GAT')
    parser.add_argument("--pool_ratio", type=float, default=0.5, help='Pooling ratio for SAGPool')
    parser.add_argument("--pool_layer", type=int, default=1, help='Pooling layers for DiffPool')
    parser.add_argument("--pool_rel", type=int, default=1, help='Using relatin features for pooling operation')
    parser.add_argument("--pool_dim", type=int, default=64, help='Pooling hidden dim for DiffPool')
    parser.add_argument("--assign_node", type=int, default=100, help='Assignment node for DiffPool')
    parser.add_argument("--init_emb", type=int, default=1, help='Embed node/edge embeddings')
    parser.add_argument("--n_layer", type=int, default=3, help='number of layers')
    parser.add_argument("--embed_type", type=str, default='float', help='int, float, both')
    parser.add_argument("--embed_dim", type=int, default=128, help='angle domain, bong length embedding dim')
    parser.add_argument("--hidden_size", type=int, default=128, help='head_dim * num_head')
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--input_drop", type=float, default=0.)
    parser.add_argument("--attn_drop", type=float, default=0.)
    parser.add_argument("--feat_drop", type=float, default=0.)
    parser.add_argument("--fusion_mode", type=str, default='cat', help='sum, max, min, mean, attn (attention), cat (concatenation)')
    # Conifg for GNN
    parser.add_argument("--a2a_topk", type=int, default=-1, help='Top-K neighbors for each atom node')
    parser.add_argument("--bab_topk", type=int, default=-1, help='Top-K neighbors for each bond node')
    parser.add_argument("--leaky_relu", type=float, default=0.)
    parser.add_argument("--layer_norm", type=int, default=1, help='Whether or not use LayerNorm in GNN')
    parser.add_argument("--feed_forward", type=int, default=0, help='Whether or not use the FeedForward layer in GNN')
    parser.add_argument("--residual", type=int, default=1, help='Whether or not to to use the residual connection in GNN')
    parser.add_argument("--is_rel", type=int, default=1, help='Whether the edge has representation')
    ## training config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epoch", type=int, default=3, help='Total epochs')
    parser.add_argument("--warmup_epoch", type=int, default=100, help='Warmup epochs')
    parser.add_argument("--epoch_run", type=int, default=0, help='Start epoch')
    parser.add_argument("--step_count", type=int, default=0, help='Start step')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5, help='weight decay')
    parser.add_argument("--patience", type=int, default=30, help='Early stopping')
    parser.add_argument("--max_norm", type=int, default=5, help='Clips gradient norm of an iterable of parameters.')
    ## save path
    parser.add_argument("--summary_writer", type=bool, default=0)
    parser.add_argument("--save_dir", type=str,
                        default=#'/media/data0/lm/Experiments/3D_DTI/GeoTrans/'
                                './results/{}/{}_graph_pool_{}_pool_attn{}_nlayer{}_embed_{}_dim{}_hsize{}_{}_{}'
                                '_pool_node{}_readout_{}_head{}_lr{}_weight_decay{}_leaky{}/', )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    args.processed_dir = args.processed_dir.format(args.smiles_feat)
    torch.backends.cudnn.benchmark = True
    # # Distributed setting
    if args.dist_train:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device('cuda', local_rank)
    else:
        torch.cuda.set_device(args.device)
    set_random_seed(args.seed)

    # Save path
    args.save_dir = args.save_dir.format(args.split_type, args.smiles_feat, args.graph_pool, args.pool_attn, args.n_layer,
                                         args.embed_type, args.embed_dim, args.hidden_size, args.pool, args.pool_gnn,
                                         args.assign_node, args.readout,
                                         args.n_head, args.lr, args.weight_decay,
                                         args.leaky_relu)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + '/model/', exist_ok=True)
    os.makedirs(args.save_dir + '/tensorboard/', exist_ok=True)
    os.makedirs(args.save_dir + '/outputs/', exist_ok=True)

    if args.dist_train:
        if local_rank == 0:
            sys.stdout = Logger(args.save_dir + 'log.txt')
            print(args.save_dir)
            f_csv = open(args.save_dir + '/results.csv', 'a')
            f_csv.write('repeat,rmse,mae,sd,r\n')
            f_csv.close()
    else:
        sys.stdout = Logger(args.save_dir + 'log.txt')
        print(args.save_dir)
        f_csv = open(args.save_dir + '/results.csv', 'a')
        f_csv.write('repeat,rmse,mae,sd,r\n')
        f_csv.close()

    # Loss
    criterion = MSELoss(reduction='mean')

    for repeat in range(args.n_repeat):
        print(f'This is repeat {repeat}...')
        args.repeat = repeat
        args.epoch_run = 0
        # check wthether the model has been trained
        if not args.only_test:
            if os.path.exists(args.save_dir + f'/model/best_model_repeat{repeat}.pt'):
                print('The model has been trained!!!')
                continue

        # Dataset prepare
        collator_train = Collator_fn(args, training=True)
        collator_val_test = Collator_fn(args, training=False)
        train_dataset = MoleculeDataset(args, graph_path=f'{args.processed_dir}/{args.dataset}/{args.split_type}/repeat{repeat}_train_{args.cutoff}_{args.n_angle}_graph.pkl',
                                        train_val=f'train_repeat{repeat}')
        val_dataset = MoleculeDataset(args, graph_path=f'{args.processed_dir}/{args.dataset}/{args.split_type}/repeat{repeat}_val_{args.cutoff}_{args.n_angle}_graph.pkl',
                                      train_val=f'val_repeat{repeat}')
        test_dataset = MoleculeDataset(args, graph_path=f'{args.processed_dir}/{args.dataset}/{args.split_type}/test_{args.cutoff}_{args.n_angle}_graph.pkl',
                                       train_val='test')
        if args.dist_train:
            train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset),
                                      batch_size=args.batch_size, num_workers=args.n_threads,
                                      worker_init_fn=seed_worker, drop_last=False, collate_fn=collator_train)
            val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset),
                                      batch_size=args.batch_size, num_workers=args.n_threads,
                                      worker_init_fn=seed_worker, drop_last=False, collate_fn=collator_val_test)
            test_loader = DataLoader(test_dataset, sampler=DistributedSampler(test_dataset),
                                  batch_size=args.batch_size, num_workers=args.n_threads,
                                  worker_init_fn=seed_worker, drop_last=False, collate_fn=collator_val_test)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=True,
                                      drop_last=False, worker_init_fn=seed_worker, collate_fn=collator_train)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_threads,
                                    drop_last=False, worker_init_fn=seed_worker, collate_fn=collator_val_test)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_threads,
                                     drop_last=False, worker_init_fn=seed_worker, collate_fn=collator_val_test)

        # Model
        model = KSGNN(args=args, d_hidden=args.hidden_size, n_layer=args.n_layer, n_heads=args.n_head, n_ffn_dense_layers=2,
                      input_drop=args.input_drop, attn_drop=args.attn_drop, feat_drop=args.feat_drop,
                      readout_mode=args.readout)
        model = model.to(args.device)
        if os.path.exists(args.save_dir + f'/model/model_repeat{repeat}.pt'):
            load_snapshot(args, model, snapshot_path=args.save_dir + f'/model/model_repeat{repeat}.pt')
        if args.dist_train:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=7000, tot_updates=40000, step_count=args.step_count, lr=args.lr, end_lr=1e-9, power=1)

        if args.dist_train:
            if args.summary_writer & (local_rank == 0):
                summary_writer = SummaryWriter(args.save_dir + f'/tensorboard/repest{repeat}')
            else:
                summary_writer = None
        else:
            if args.summary_writer:
                summary_writer = SummaryWriter(args.save_dir + f'/tensorboard/repest{repeat}')
            else:
                summary_writer = None

        trainer = Trainer(args, optimizer, lr_scheduler, criterion, summary_writer, device=args.device, local_rank=0)
        if not args.only_test:
            trainer.fit(model, train_loader, val_loader, test_loader)
        # the process may be terminated while training with distribution.
        # Then run 'killall python' in the terminal can kill the zombie processes
        preds, true_pks = trainer.predict(model, test_loader)
        rmse, mae, r, sd = metric.evaluate(true_pks, preds)
        print(round(rmse, 4), round(mae, 4), round(sd, 4), round(r, 4))
        res = dict()
        res['pred'] = preds.tolist()
        res['ground_truth'] = true_pks.tolist()
        json.dump(res, open(args.save_dir + f'/outputs/labels000{repeat}.json', 'w'))

        if args.dist_train:
            if local_rank == 0:
                ls = [repeat, round(rmse, 4), round(mae, 4), round(sd, 4), round(r, 4)]
                f_csv = open(args.save_dir + '/results.csv', 'a')
                f_csv.write(','.join(map(str, ls)) + '\n')
                f_csv.close()
                if args.summary_writer:
                    summary_writer.close()
        else:
            ls = [repeat, round(rmse, 4), round(mae, 4), round(sd, 4), round(r, 4)]
            f_csv = open(args.save_dir + '/results.csv', 'a')
            f_csv.write(','.join(map(str, ls)) + '\n')
            f_csv.close()
            if args.summary_writer:
                summary_writer.close()

    # compute mean and std
    if args.dist_train:
        if local_rank == 0:
            cal_final_results(args)
            print('Done!!!!!!!!')
    else:
        cal_final_results(args)
    print('Done!!!!!!!!')

