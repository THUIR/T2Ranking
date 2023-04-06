#!/bin/bash
dataset=marco
sample_num=64
batch_size=16
echo "batch size ${batch_size}"
dev_batch_size=256
min_index=0
max_index=256
max_seq_len=332
q_max_seq_len=32
p_max_seq_len=128
model_name_or_path=checkpoint/bert-base-chinese/
top1000=data/train.mined.tsv
dev_top1000='yourself retrieved top1000 file'
dev_query=data/queries.dev.tsv
collection=data/collection.tsv
qrels=data/qrels.retrieval.train.tsv
query=data/queries.train.tsv
learning_rate=3e-5
### 下面是永远不用改的
warmup_proportion=0.1
eval_step_proportion=0.01
report_step=100
epoch=20
fp16=true
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
# pip install https://paddle-wheel.bj.bcebos.com/benchmark/torch-1.12.0%2Bcu113-cp37-cp37m-linux_x86_64.whl 
# pip install transformers
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m torch.distributed.launch \
    --log_dir ${log_dir} \
    --nproc_per_node=8 \
    src/train_cross_encoder.py \
    --model_name_or_path=${model_name_or_path} \
    --batch_size=${batch_size} \
    --warmup_proportion=${warmup_proportion} \
    --eval_step_proportion=${eval_step_proportion} \
    --report=${report_step} \
    --qrels=${qrels} \
    --query=${query} \
    --dev_query=${dev_query} \
    --collection=${collection} \
    --top1000=${top1000} \
    --min_index=${min_index} \
    --max_index=${max_index} \
    --epoch=${epoch} \
    --sample_num=${sample_num} \
    --dev_batch_size=${dev_batch_size} \
    --max_seq_len=${max_seq_len} \
    --learning_rate=${learning_rate} \
    --q_max_seq_len=${q_max_seq_len} \
    --p_max_seq_len=${p_max_seq_len} \
    --dev_top1000=${dev_top1000} \
    --warm_start_from=${warm_start_from} \
    | tee ${log_dir}/train.log
echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="
