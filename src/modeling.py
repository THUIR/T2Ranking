import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer,BertForMaskedLM,
                          AutoModelForSequenceClassification, BertModel, BertLayer,
                          PreTrainedModel)
from transformers.file_utils import ModelOutput
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from utils import filter_stop_words

logger = logging.getLogger(__name__)




class Reranker(nn.Module):
    def __init__(self, args):
        super(Reranker, self).__init__()
        self.lm = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=1, output_hidden_states=True)
        if args.gradient_checkpoint:
            self.lm.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        ret = self.lm(**batch, return_dict=True)
        logits = ret.logits
        if self.training:
            scores = logits.view(-1, self.args.sample_num)  # q_batch_size, sample_num
            target_label = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target_label)
            return loss
        return logits


class DualEncoder(nn.Module):
    def __init__(self, args):
        super(DualEncoder, self).__init__()
        self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, output_hidden_states=True, add_pooling_layer=False)
        self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, output_hidden_states=True, add_pooling_layer=False)
        if args.gradient_checkpoint:
            self.lm_q.gradient_checkpointing_enable()
            self.lm_p.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def encode_query(self, query_inputs):
        qry_out = self.lm_q(**query_inputs, return_dict=True)
        q_hidden = qry_out.hidden_states[-1]
        q_reps = q_hidden[:, 0]
        return q_reps

    def encode_passage(self, passage_inputs):
        mlm_labels = None
        psg_out = self.lm_p(**passage_inputs, return_dict=True)
        p_hidden = psg_out.hidden_states[-1]  
        p_reps = p_hidden[:, 0]
        return p_reps

    def forward(self, query_inputs=None, passage_inputs=None):
        if self.training:
            q_reps = self.encode_query(query_inputs)
            p_reps = self.encode_passage(passage_inputs)

            if self.args.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if self.args.negatives_in_device:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
            else:
                p_reps = p_reps.view(-1, self.args.sample_num, 768)
                scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))
                scores = scores.squeeze(1)
                target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target)
            return loss
        else:
            if query_inputs is not None:
                return self.encode_query(query_inputs)
            else:
                return self.encode_passage(passage_inputs)

