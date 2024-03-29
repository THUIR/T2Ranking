#!/bin/bash
dataset=marco
sample_num=2
batch_size=64
echo "batch size ${batch_size}"
max_index=200
retriever_model_name_or_path=checkpoint/bert-base-chinese/
top1000=data/train.mined.tsv
warm_start_from=data/dual-encoder-trained-with-hard-negatives.p
learning_rate=2e-5
### 下面是永远不用改的
dev_batch_size=256
min_index=0
max_seq_len=332
q_max_seq_len=32
p_max_seq_len=300
dev_query=data/queries.dev.tsv
collection=data/collection.tsv
qrels=data/qrels.retrieval.train.tsv
dev_qrels=data/qrels.retrieval.dev.tsv
query=data/queries.train.tsv
warmup_proportion=0.1
eval_step_proportion=0.01
report_step=100
epoch=200
fp16=true
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
master_port=29500
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m torch.distributed.launch \
    --log_dir ${log_dir} \
    --nproc_per_node=8 \
    --master_port=${master_port} \
    src/train_dual_encoder.py \
    --retriever_model_name_or_path=${retriever_model_name_or_path} \
    --batch_size=${batch_size} \
    --warmup_proportion=${warmup_proportion} \
    --eval_step_proportion=${eval_step_proportion} \
    --report=${report_step} \
    --qrels=${qrels} \
    --dev_qrels=${dev_qrels} \
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
    --warm_start_from=${warm_start_from} \
    | tee ${log_dir}/train.log

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="

