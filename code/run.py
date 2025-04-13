from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import logging
import multiprocessing
import os
import random

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import *

cpu_cont = multiprocessing.cpu_count()
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'longformer': (LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer)
}


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


class InputFeatures(object):  # 封装原始数据
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
        patience (int): 监控指标不再提升的轮数，超过该值停止训练
        verbose (bool): 是否打印早停信息
        delta (float): 性能提升的最小值，只有超过该值才认为是提升
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs of no improvement.")
        return self.early_stop


# 清理代码中的无关内容
def clean_code(string):
    # 移除单行注释
    string = re.sub(r"//.*?$", "", string, flags=re.MULTILINE)
    # 移除多行注释
    string = re.sub(r"/\*.*?\*/", "", string, flags=re.DOTALL)
    # 移除空行
    string = re.sub(r"^\s*$\n?", "", string, flags=re.MULTILINE)
    return string.strip()


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(clean_code(js['func']).split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []

        # 如果提供了文件路径，则打开文件并处理每一行
        if file_path:
            # 计算文件的总行数
            with open(file_path) as f:
                total_lines = sum(1 for _ in f)

            # 重新打开文件，进行数据处理
            with open(file_path) as f:
                # 使用 tqdm 设置 total 为文件总行数
                for line in tqdm(f, desc="Processing lines", total=total_lines, unit="line"):
                    js = json.loads(line.strip())
                    feature = convert_examples_to_features(js, tokenizer, args)
                    self.examples.append(feature)  # 每个函数产生一个合并后的特征

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            logger.info("*** Total Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Sample ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 模型训练
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    # 最大步数、保存步数、预热步数和日志步数
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    # 训练轮数
    args.num_train_epochs = args.epoch
    model.to(args.device)

    # 准备优化器和学习率调度器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    # fp16训练
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 多GPU和分布式
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # 加载上次训练的检查点
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    # 初始化全局步数和损失变量
    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        seed = idx * 100 + args.seed  # 增强随机性
        bar = tqdm(train_dataloader, total=len(train_dataloader), leave=True, unit="batch")
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            # for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device).double()
            labels = batch[1].to(args.device).double()
            model.train()
            loss, logits = model(inputs, labels, seed)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)

            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            # logger.info("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                #avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                avg_loss = round((tr_loss - logging_loss) / max((global_step - tr_nb), 1), 4)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        bar.close()
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            tqdm.write(f"  {key} = {round(value, 4)}")
                        # Save model checkpoint

                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'

                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model

                        output_dir1 = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir1)

                        # 保存 config.json
                        config_output_dir = os.path.join(output_dir, 'config.json')
                        with open(config_output_dir, 'w') as f:
                            f.write(model.config.to_json_string())
                        # 保存 idx_file.txt (保存当前的epoch)
                        idx_file_output_dir = os.path.join(output_dir, 'idx_file.txt')
                        with open(idx_file_output_dir, 'w') as f:
                            f.write(str(idx))
                        # 保存 step_file.txt (保存当前的step)
                        step_file_output_dir = os.path.join(output_dir, 'step_file.txt')
                        with open(step_file_output_dir, 'w') as f:
                            f.write(str(global_step))  # 保存当前的训练步数
                        logger.info("Saving model checkpoint to %s", output_dir1)

        avg_loss = round(train_loss / tr_num, 5)
        logger.info("Epoch {} loss {}".format(idx + 1, avg_loss))

        # 添加早停判断（每个 epoch 后）
        if args.evaluate_during_training:
            eval_results = evaluate(args, model, tokenizer)
            val_loss = eval_results['eval_loss']

            if early_stopping(val_loss, model):
                logger.info("Early stopping triggered at epoch {}".format(idx + 1))
                break


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation *****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []

    with tqdm(total=len(eval_dataloader), desc="Evaluating", unit="batch", leave=True) as bar:
        for batch in eval_dataloader:
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit = model(inputs, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

            nb_eval_steps += 1
            bar.update(1)

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5

    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    auc = roc_auc_score(labels, logits[:, 0])  # Use logits for AUC
    fpr, tpr, thresholds = roc_curve(labels, logits[:, 0])
    fpr_at_05 = fpr[np.argmin(np.abs(tpr - 0.5))] if len(fpr) > 0 else 0.0

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_precision": round(precision, 4),
        "eval_recall": round(recall, 4),
        "eval_f1": round(f1, 4),
        "eval_auc": round(auc, 4),
        "eval_fpr_at_05": round(fpr_at_05, 4),
    }
    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    logits = []
    labels = []
    # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5

    test_acc = np.mean(labels == preds)

    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + '\t1\n')
            else:
                f.write(example.idx + '\t0\n')

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    auc = roc_auc_score(labels, logits[:, 0])

    fpr, tpr, thresholds = roc_curve(labels, logits[:, 0])
    fpr_at_05 = fpr[np.where(thresholds >= 0.5)][0] if len(np.where(thresholds >= 0.5)[0]) > 0 else 0.0


    result = {
        "test_acc": round(test_acc, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "test_auc": round(auc, 4),
        "test_fpr_at_05": round(fpr_at_05, 4),
    }
    with open(os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_test_result.txt"), 'w') as f:
        for key, value in result.items():
            f.write(f"{key} = {value}\n")
    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../dataset/train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=100,
                        help="train epochs")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument("--model", default="GNNs", type=str, help="")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")

    parser.add_argument("--format", default="uni", type=str,
                        help="idx for index-focused method, uni for unique token-focused method")
    parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percet of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percet of training sample")

    # GRAND parameters
    parser.add_argument("--dropnode", default="0.2", type=float, help="节点丢弃概率")
    parser.add_argument("--lam", default="1.0", type=float, help="一致性损失")
    parser.add_argument("--temp", default=0.8, type=float, help="平滑度")
    parser.add_argument("--S", default="4", type=int, help="数据增强样本数")
    parser.add_argument("--K", default="4", type=int, help="图传播步数")
    parser.add_argument("--input_droprate", default=0.2, type=float, help="输入特征丢弃率")
    parser.add_argument("--hidden_droprate", default=0.2, type=float, help="隐藏层丢弃率")

    parser.add_argument("--use_contrastive", action='store_true', help="是否使用GRACE对比学习")
    parser.add_argument("--contrastive_weight", type=float, default=0.1, help="对比损失的权重")
    parser.add_argument("--temperature", type=float, default=0.5, help="对比损失的温度参数")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(args.n_gpu, 1)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(args.n_gpu, 1)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(seed=args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')

    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # 从预训练的模型配置文件中加载模型的配置
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.block_size <= 0:
        args.block_size = 512

    # 确保 block_size 不超过模型允许的最大值
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,
                                            ignore_mismatched_sizes=True)
    else:
        model = model_class(config)

    ## model
    if args.model == "original":
        model = Model(model, config, tokenizer, args)
    else:  # GNNs
        model = GNN(model, config, tokenizer, args)

    model = model.double()  # 转换所有参数和缓冲区为double类型

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test_result = test(args, model, tokenizer)

        logger.info("***** Test results *****")
        for key in sorted(test_result.keys()):
            logger.info("  %s = %s", key, str(round(test_result[key], 4)))

    return results


if __name__ == "__main__":
    main()
