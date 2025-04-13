#!/bin/bash

# 定义参数
OUTPUT_DIR="./saved_models"

MODEL_TYPE="roberta"
TOKENIZER_NAME="microsoft/graphcodebert-base"
MODEL_NAME_OR_PATH="microsoft/graphcodebert-base"

#MODEL_TYPE="longformer"
#TOKENIZER_NAME="allenai/longformer-base-4096"
#MODEL_NAME_OR_PATH="allenai/longformer-base-4096"

TRAIN_DATA_FILE="../dataset/train.jsonl"
EVAL_DATA_FILE="../dataset/valid.jsonl"
TEST_DATA_FILE="../dataset/test.jsonl"

# 执行训练命令（自动捕获日志）
python run.py \
  --output_dir=$OUTPUT_DIR \
  --model_type=$MODEL_TYPE \
  --tokenizer_name=$TOKENIZER_NAME \
  --model_name_or_path=$MODEL_NAME_OR_PATH \
  --do_eval \
  --do_test \
  --do_train \
  --train_data_file=$TRAIN_DATA_FILE \
  --eval_data_file=$EVAL_DATA_FILE \
  --test_data_file=$TEST_DATA_FILE \
  --block_size=4096 \
  --train_batch_size=128 \
  --eval_batch_size=128 \
  --max_grad_norm=1.0 \
  --evaluate_during_training \
  --gnn=GraphSAGE \
  --learning_rate=5e-4 \
  --epoch=100 \
  --hidden_size=256 \
  --num_GNN_layers=2 \
  --format=uni \
  --window_size=5 \
  --seed=123456 \
  --use_contrastive \
  --contrastive_weight=0.01 \
  --temperature=0.5 \
  2>&1 | tee ~/training_log.txt