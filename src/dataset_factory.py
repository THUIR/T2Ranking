import os
import random

import numpy as np
import pandas as pd
import pytorch_pretrained_bert
import torch
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DataCollatorForWholeWordMask

class PassageDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t", quoting=3)
        self.collection.columns=['pid', 'para']
        self.collection = self.collection.fillna("NA")        
        self.collection.index = self.collection.pid 
        total_cnt = len(self.collection)
        shard_cnt = total_cnt//self.n_procs
        if self.rank!=self.n_procs-1:
            self.collection = self.collection[self.rank*shard_cnt:(self.rank+1)*shard_cnt]
        else:
            self.collection = self.collection[self.rank*shard_cnt:]
        self.num_samples = len(self.collection)
        print('rank:',self.rank,'samples:',self.num_samples)

    def _collate_fn(self, psgs):
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        return p_records

    def __getitem__(self, idx):
        cols = self.collection.iloc[idx]
        para = cols.para
        psg = para
        return psg

    def __len__(self):
        return self.num_samples


class QueryDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.args = args
        self.collection = pd.read_csv(args.dev_query, sep="\t", quoting=3)
        self.collection.columns = ['qid','qry']
        self.collection = self.collection.fillna("NA")
        self.num_samples = len(self.collection)
        
    def _collate_fn(self, qrys):
        return self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)

    def __getitem__(self, idx):
        return self.collection.iloc[idx].qry

    def __len__(self):
        return self.num_samples
    

class CrossEncoderTrainDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t", quoting=3)
        self.collection.columns=['pid', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t")
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.top1000, sep="\t")
        self.top1000.columns=['qid','pid','index', 'score']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with f(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args.min_index:self.args.max_index]
        interval = len(pids)//sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = [(query, self.collection.loc[pos_id]['para'])]
        for neg_pid in sample_neg_pids:
            data.append((query, self.collection.loc[neg_pid]['para']))
        return data

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        for qp_pairs in sample_list:
            for q,p in qp_pairs:
                qrys.append(q)
                psgs.append(p)
        features = self.tokenizer(qrys, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return features

    def __len__(self):
        return self.num_samples

class CrossEncoderDevDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t", quoting=3)
        self.collection.columns=['pid', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.dev_query,sep="\t")
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.dev_top1000, sep="\t", header=None)
        self.num_samples = len(self.top1000)


    def __getitem__(self, idx):
        cols = self.top1000.iloc[idx]
        qid = cols[0]
        pid = cols[1]
        return self.query.loc[qid]['text'], self.collection.loc[pid]['para'], qid, pid

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qids = []
        pids = []
        for q,p,qid,pid in sample_list:
            qrys.append(q)
            psgs.append(p)
            qids.append(qid)
            pids.append(pid)
        features = self.tokenizer(qrys, psgs,  padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_len)
        return features, {"qids":np.array(qids),"pids":np.array(pids)}
        
    def __len__(self):
        return self.num_samples

class DualEncoderTrainDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = pd.read_csv(args.collection,sep="\t", quoting=3)
        self.collection.columns=['pid','para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid 
        self.collection.pop('pid')
        self.query = pd.read_csv(args.query,sep="\t")
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(args.top1000, sep="\t")
        if len(self.top1000.columns)==3:
            self.top1000.columns=['qid','pid','index']
        else:
            self.top1000.columns=['qid','pid','index','score']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)
        self.min_index = args.min_index
        self.max_index = args.max_index
        qrels={}
        with open(args.qrels,'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = args.sample_num-1   
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch
        print(self.epoch)
    
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.args.min_index:self.args.max_index]
        if len(pids)<sample_num:
            pad_num = sample_num - len(pids)
            pids+=[random.randint(0, 2303643) for _ in range(pad_num)]  # 用random neg补充
        interval = len(pids)//sample_num
        offset = self.epoch%interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = self.collection.loc[pos_id]
        psgs = [data['para']]
        for neg_pid in sample_neg_pids:
            data = self.collection.loc[neg_pid]
            psgs.append(data['para'])
        return [query], psgs

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        for q, p in sample_list:
            qrys+=q 
            psgs+=p 
        q_records = self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        return {"query_inputs":q_records, "passage_inputs":p_records}

    def __len__(self):
        return self.num_samples

