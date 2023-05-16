import json
import six
import os
import faiss
import numpy as np
class HParams(object):
    """Hyper paramerter"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise ValueError('key(%s) not in HParams.' % key)
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.to_dict())

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    @classmethod
    def from_json(cls, json_str):
        """doc"""
        d = json.loads(json_str)
        if type(d) != dict:
            raise ValueError('json object must be dict.')
        return HParams.from_dict(d)

    def get(self, key, default=None):
        """doc"""
        return self.__dict__.get(key, default)

    @classmethod
    def from_dict(cls, d):
        """doc"""
        if type(d) != dict:
            raise ValueError('input must be dict.')
        hp = HParams(**d)
        return hp

    def to_json(self):
        """doc"""
        return json.dumps(self.__dict__)

    def to_dict(self):
        """doc"""
        return self.__dict__
    
    def print_config(self):
        for key,value in self.__dict__.items():
            print(key+":",value)

    def join(self, other):
        """doc"""
        if not isinstance(other, HParams):
            raise ValueError('input must be HParams instance.')
        self.__dict__.update(**other.__dict__)
        return self


def _get_dict_from_environ_or_json_or_file(args, env_name):
    if args == '':
        return None
    if args is None:
        s = os.environ.get(env_name)
    else:
        s = args
        if os.path.exists(s):
            s = open(s).read()
    if isinstance(s, six.string_types):
        try:
            r = eval(s)
        except SyntaxError as e:
            raise ValueError('json parse error: %s \n>Got json: %s' %
                             (repr(e), s))
        return r
    else:
        return s  #None


def parse_file(filename):
    """useless api"""
    d = _get_dict_from_environ_or_json_or_file(filename, None)
    if d is None:
        raise ValueError('file(%s) not found' % filename)
    return d

def build_engine(p_emb_matrix, dim):
    index = faiss.IndexFlatIP(dim)
    index.add(p_emb_matrix.astype('float32'))
    return index
from tqdm import tqdm
def read_embed(file_name, dim=768, bs=100):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        with tqdm(total=len(emb_np)//bs+1) as pbar:
            while(i < len(emb_np)):
                vec_list = emb_np[i:i+bs]
                i += bs
                pbar.update(1)
                yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in tqdm(inp):
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                assert len(vector) == dim
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list

def load_qid(file_name):
    qid_list = []
    with open(file_name) as inp:
        for line in inp:
            line = line.strip()
            qid = line.split('\t')[0]
            try:
                int(qid)
                qid_list.append(qid)
            except:
                pass
    return qid_list

import torch
def topk_query_passage(query_vector, passage_vector, k):
    """
    对query vector和passage vector进行内积计算，并返回top k的索引

    Args:
        query_vector (torch.Tensor): query向量，形状为 (batch_size, query_dim)
        passage_vector (torch.Tensor): passage向量，形状为 (batch_size, passage_dim)
        k (int): 返回的top k值

    Returns:
        torch.Tensor: top k值的索引，形状为 (batch_size, k)
    """
    # 计算query向量和passage向量的内积
    scores = torch.matmul(query_vector, passage_vector.t())  # 形状为 (batch_size, batch_size)

    # 对每个batch进行排序，取top k值
    res_dist, res_p_id = torch.topk(scores, k=k, dim=1)  # 形状为 (batch_size, k)

    return res_dist.cpu().numpy(), res_p_id.cpu().numpy()

def search(index, emb_file, qid_list, outfile, top_k):
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in read_embed(emb_file):
            q_emb_matrix = np.array(batch_vec)
            q_emb_matrix = torch.from_numpy(q_emb_matrix)
            q_emb_matrix = q_emb_matrix.cuda()
            res_dist, res_p_id = topk_query_passage(q_emb_matrix, index, top_k)
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, j+1, score))
                q_idx += 1

def merge(total_part, shift, top, eval_cnts):
    f_list = []
    for part in range(total_part):
        f0 = open('output/res.top%d.part%d.step%d' % (top, part, eval_cnts))
        f_list.append(f0)

    line_list = []
    for part in range(total_part):
        line = f_list[part].readline()
        line_list.append(line)

    out = open('output/res.top%d.step%d' % (top, eval_cnts), 'w')
    last_q = ''
    ans_list = {}
    while line_list[-1]:
        cur_list = []
        for line in line_list:
            sub = line.strip().split('\t')
            cur_list.append(sub)

        if last_q == '':
            last_q = cur_list[0][0]
        if cur_list[0][0] != last_q:
            rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
            for i in range(top):
                out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
            ans_list = {}
        for i, sub in enumerate(cur_list):
            ans_list[int(sub[1]) + shift*i] = float(sub[-1])
        last_q = cur_list[0][0]

        line_list = []
        for f0 in f_list:
            line = f0.readline()
            line_list.append(line)

    rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
    for i in range(top):
        out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
    out.close()


def add_prefix(state_dict, prefix='module.'):
    if all(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped_state_dict = {}
    for key in list(state_dict.keys()):
        key2 = prefix + key
        stripped_state_dict[key2] = state_dict.pop(key)
    return stripped_state_dict

def filter_stop_words(txts):
    stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    txts = [t.split() for t in txts]
    txts = [list(set(list(filter(lambda x:x not in stop_words,t)))) for t in txts]
    rets = []
    for t in txts:
        rets+=t
    return list(set(rets))
