import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import argparse
import random
import subprocess
import tempfile
import time
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import distributed, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel

import dataset_factory
import utils
from modeling import Reranker
from msmarco_eval import calc_mrr
from utils import add_prefix, build_engine, load_qid, read_embed, search

SEED = 2023
best_mrr=-1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
def define_args():
    import argparse
    parser = argparse.ArgumentParser('BERT-ranker model')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--q_max_seq_len', type=int, default=160)
    parser.add_argument('--p_max_seq_len', type=int, default=160)
    parser.add_argument('--model_name_or_path', type=str, default="../../data/bert-base-uncased/")
    parser.add_argument('--reranker_model_name_or_path', type=str, default="../../data/bert-base-uncased/")
    parser.add_argument('--warm_start_from', type=str, default="")
    parser.add_argument('--model_out_dir', type=str, default="output")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_step_proportion', type=float, default=1.0)
    parser.add_argument('--report', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--qrels', type=str, default="../../data/marco/qrels.train.debug.tsv")
    parser.add_argument('--dev_qrels', type=str, default="../../data/marco/qrels.train.debug.tsv")
    parser.add_argument('--top1000', type=str, default="../../data/marco/run.msmarco-passage.train.debug.tsv")
    parser.add_argument('--dev_top1000', type=str, default="../../data/marco/run.msmarco-passage.train.debug.tsv")
    parser.add_argument('--collection', type=str, default="../../data/marco/collection.debug.tsv")
    parser.add_argument('--query', type=str, default="../../data/marco/train.query.debug.txt")
    parser.add_argument('--dev_query', type=str, default="../../data/marco/train.query.debug.txt")
    parser.add_argument('--min_index', type=int, default=0)
    parser.add_argument('--max_index', type=int, default=256)
    parser.add_argument('--sample_num', type=int, default=128)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_checkpoint', type=bool, default=True)
    parser.add_argument('--negatives_x_device', type=bool, default=True)
    parser.add_argument('--untie_encoder', type=bool, default=True)
    parser.add_argument('--add_pooler', type=bool, default=False)
    parser.add_argument('--Temperature', type=float, default=1.0)

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args

def merge(eval_cnts, file_pattern='output/res.step-%d.part-0%d'):
    f_list = []
    total_part = torch.distributed.get_world_size()
    for part in range(total_part):
        f0 = open(file_pattern % (eval_cnts, part))
        f_list+=f0.readlines()
    f_list = [l.strip().split("\t") for l in f_list]
    dedup = defaultdict(dict)
    for qid,pid,score in f_list:
        dedup[qid][pid] = float(score)
    mp = defaultdict(list)
    for qid in dedup:
        for pid in dedup[qid]:
            mp[qid].append((pid, dedup[qid][pid]))
    for qid in mp:
        mp[qid].sort(key=lambda x:x[1], reverse=True)
    with open(file_pattern.replace('.part-0%d','')%eval_cnts, 'w') as f:
        for qid in mp:
            for idx, (pid, score) in enumerate(mp[qid]):
                f.write(str(qid)+"\t"+str(pid)+'\t'+str(idx+1)+"\t"+str(score)+'\n')
    for part in range(total_part):
        os.remove(file_pattern % (eval_cnts, part))

def train_cross_encoder(args, model, optimizer):
    epoch = 0
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        print(f'Starting training, upto {args.epoch} epochs, LR={args.learning_rate}', flush=True)

    # 加载数据集
    dev_dataset = dataset_factory.CrossEncoderDevDataset(args)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, collate_fn=dev_dataset._collate_fn, sampler=dev_sampler, num_workers=4)
    validate_multi_gpu(model, dev_loader, epoch, args)

    train_dataset = dataset_factory.CrossEncoderTrainDataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    for epoch in range(1, args.epoch+1):
        train_dataset.set_epoch(epoch)  # 选择的negative根据epoch后移
        train_sampler.set_epoch(epoch)  # shuffle batch
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset._collate_fn, sampler=train_sampler, num_workers=4, drop_last=True)
        train_iteration_multi_gpu(model, optimizer, train_loader, epoch, args)
        torch.distributed.barrier()
        del train_loader
        if epoch%1==0:
            validate_multi_gpu(model, dev_loader, epoch, args)
            torch.distributed.barrier()

def validate_multi_gpu(model, dev_loader, epoch, args):
    global best_mrr
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    with torch.no_grad():
        model.eval()
        scores_lst = []
        qids_lst = []
        pids_lst = []
        for record1, record2 in tqdm(dev_loader):
            with autocast():
                scores = model(_prepare_inputs(record1))
            qids = record2['qids']
            pids = record2['pids']
            scores_lst.append(scores.detach().cpu().numpy().copy())
            qids_lst.append(qids.copy())
            pids_lst.append(pids.copy())
        qids_lst = np.concatenate(qids_lst).reshape(-1)
        pids_lst = np.concatenate(pids_lst).reshape(-1)
        scores_lst = np.concatenate(scores_lst).reshape(-1)
        with open("output/res.step-%d.part-0%d"%(epoch, local_rank), 'w') as f:
            for qid,pid,score in zip(qids_lst, pids_lst, scores_lst):
                f.write(str(qid)+'\t'+str(pid)+'\t'+str(score)+'\n')
        torch.distributed.barrier()
        if local_rank==0:
            merge(epoch)
            metrics = calc_mrr(args.dev_qrels, 'output/res.step-%d'%epoch)
            mrr = metrics['MRR @10']
            if mrr>best_mrr:
                print("*"*50)
                print("new top")
                print("*"*50)
                best_mrr = mrr
                torch.save(model.module.lm.state_dict(), os.path.join(args.model_out_dir, "reranker.p"))



def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt

def _prepare_inputs(record):
    prepared = {}
    local_rank = torch.distributed.get_rank()
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared

def train_iteration_multi_gpu(model, optimizer, data_loader, epoch, args):
    total = 0
    model.train()
    total_loss = 0.
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    start = time.time()
    local_start = time.time()
    all_steps_per_epoch = len(data_loader)
    step = 0
    scaler = GradScaler()
    for record in data_loader:
        record = _prepare_inputs(record)
        if args.fp16:
            with autocast():
                loss = model(record)
        else:
            loss = model(record)
        torch.distributed.barrier()
        reduced_loss = reduce_tensor(loss.data)
        total_loss += reduced_loss.item()
        # optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step+=1
        if step%args.report==0 and local_rank==0:
            seconds = time.time()-local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print("epoch:%d training step: %d/%d, mean loss: %.5f, current loss: %.5f,"%(epoch, step, all_steps_per_epoch, total_loss/step, loss.cpu().detach().numpy()),"report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time()-start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    if local_rank==0:
        # model.save(os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        torch.save(model.module.state_dict(), os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        seconds = time.time()-start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print(f'train epoch={epoch} loss={total_loss}')
        print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
        print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))

if __name__ == '__main__':
    args = define_args()
    args = vars(args)
    args = utils.HParams(**args)
    args.reranker_model_name_or_path = args.model_name_or_path
    # 加载到多卡
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    if local_rank==0:
        args.print_config()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = Reranker(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    params = {'params': [v for k, v in params]}
    optimizer = torch.optim.Adam([params], lr=args.learning_rate, weight_decay=0.0)

    if args.warm_start_from:
        print('warm start from ', args.warm_start_from)
        state_dict = torch.load(args.warm_start_from, map_location=device)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.','')] = state_dict.pop(k)
        model.load_state_dict(state_dict)


    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    os.makedirs(args.model_out_dir, exist_ok=True)

    train_cross_encoder(args, model, optimizer)
