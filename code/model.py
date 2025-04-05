import torch
import math
import torch.nn as nn
from sympy.codegen import Print
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss, MSELoss
from utils import preprocess_features, preprocess_adj
from utils import *
import ast
import networkx as nx
import numpy as np
import scipy.sparse as sp
from pycparser import c_parser, c_ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}


# GraphSAGE实现
class GraphSAGE(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_layers, dropout, aggregator_type='mean', residual=True):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator_type = aggregator_type
        self.out_dim = hidden_size
        self.residual = residual  # 是否使用残差连接

        self.input_projection = nn.Linear(feature_dim_size, hidden_size)

        # 定义每一层的权重矩阵
        self.weights = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = feature_dim_size if layer == 0 else hidden_size
            self.weights.append(nn.Linear(input_dim, hidden_size))

    def forward(self, x, adj, mask=None):
        """
        x: 节点特征矩阵 (batch_size, num_nodes, feature_dim_size)
        adj: 邻接矩阵 (batch_size, num_nodes, num_nodes)
        mask: 掩码矩阵 (batch_size, num_nodes, 1)
        """
        h = x  # 保存输入特征，用于残差连接

        if self.residual:
            x = self.input_projection(x)  # 调整 x 的维度

        for layer in range(self.num_layers):
            # 聚合邻居信息
            if self.aggregator_type == 'mean':
                # 均值聚合：邻接矩阵与节点特征矩阵的乘积
                neighbor_agg = torch.matmul(adj, h)  # 聚合邻居节点特征
            else:
                raise ValueError(f"Unsupported aggregator type: {self.aggregator_type}")

            # 线性变换，更新特征
            h = self.weights[layer](neighbor_agg)  # 基于聚合后的邻居信息

            # 残差连接
            if self.residual and layer > 0:  # 第一层不添加残差连接
                h = h + x  # 将输入特征与当前层输出特征相加

            # 激活函数和 dropout
            if layer < self.num_layers - 1:  # 最后一层不加激活函数
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)

        return h


weighted_graph = False
print('using default unweighted graph')


# 图构建函数
def build_graph(shuffle_doc_words_list, word_embeddings, window_size=3):
    # print('using window size = ', window_size)
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p_id = window[p]
                    word_q_id = window[q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[p])
            col.append(doc_word_id_map[q])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        x_adj.append(adj)

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[int(k)])
        x_feature.append(features)

    return x_adj, x_feature


# 另一种图构建
def build_graph_text(shuffle_doc_words_list, word_embeddings, window_size=3):
    # print('using window size = ', window_size)
    x_adj = []
    x_feature = []
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        doc_len = len(doc_words)

        row = []
        col = []
        weight = []
        features = []

        if doc_len > window_size:
            for j in range(doc_len - window_size + 1):
                for p in range(j + 1, j + window_size):
                    for q in range(j, p):
                        row.append(p)
                        col.append(q)
                        weight.append(1.)
                        #
                        row.append(q)
                        col.append(p)
                        weight.append(1.)
        else:  # doc_len < window_size
            for p in range(1, doc_len):
                for q in range(0, p):
                    row.append(p)
                    col.append(q)
                    weight.append(1.)
                    #
                    row.append(q)
                    col.append(p)
                    weight.append(1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_len, doc_len))
        if weighted_graph == False:
            adj[adj > 1] = 1.
        x_adj.append(adj)
        #
        for word in doc_words:
            feature = word_embeddings[word]
            features.append(feature)
        x_feature.append(features)

    return x_adj, x_feature

# AST图构建函数
def build_graph_ast_from_source(code_list, word_embeddings):
    x_adj = []
    x_feature = []
    vocab_size = word_embeddings.shape[0]
    parser = c_parser.CParser()

    for code_str in code_list:
        try:
            # 清理代码（可选）
            code_str = code_str.strip()
            if not code_str.endswith('}'):
                code_str += ' }'  # 处理不完整函数尾巴

            wrapped_code = f"void dummy() {code_str}"  # 包装为合法函数
            ast_tree = parser.parse(wrapped_code)

            node_list = []
            edges = []

            def add_nodes_edges(node, parent_id=None):
                node_id = len(node_list)
                node_list.append(node)
                if parent_id is not None:
                    edges.append((parent_id, node_id))
                for _, child in node.children():
                    add_nodes_edges(child, node_id)

            add_nodes_edges(ast_tree)

            N = len(node_list)
            row, col = zip(*edges) if edges else ([], [])
            adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(N, N))
            x_adj.append(adj)

            features = []
            for node in node_list:
                node_type = type(node).__name__
                type_id = hash(node_type) % vocab_size
                features.append(word_embeddings[type_id])
            x_feature.append(np.array(features))

        except Exception as e:
            x_adj.append(sp.csr_matrix((1, 1)))
            x_feature.append(np.zeros((1, word_embeddings.shape[1])))

    return x_adj, x_feature



def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.matmul(z, z.T) / temperature  # 2B x 2B

    # 创建标签
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    # mask 自己
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    sim = sim.masked_fill(mask, float('-inf'))

    # 正样本对索引偏移量
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size),
                           torch.arange(0, batch_size)]).to(z1.device)

    loss = F.cross_entropy(sim, positives)
    return loss


class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.double())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GNNReGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # 根据模型类型动态获取嵌入层
        if hasattr(self.encoder, 'roberta'):  # 如果是 Roberta 模型
            self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        elif hasattr(self.encoder, 'longformer'):  # 如果是 Longformer 模型
            self.w_embeddings = self.encoder.longformer.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        else:
            raise ValueError("Unsupported model type. Expected Roberta or Longformer.")

        if args.gnn == "GraphSAGE":  # 添加 GraphSAGE 支持
            self.gnn = GraphSAGE(feature_dim_size=args.feature_dim_size,
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_GNN_layers,
                                 dropout=config.hidden_dropout_prob,
                                 aggregator_type='mean',
                                 residual=not args.remove_residual)

        else:
            self.gnn = GraphSAGE(feature_dim_size=args.feature_dim_size,
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_GNN_layers,
                                 dropout=config.hidden_dropout_prob,
                                 aggregator_type='mean',
                                 residual=not args.remove_residual)

        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

    def forward(self, input_ids=None, labels=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # 构造图
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings,
                                         window_size=self.args.window_size)
        elif self.args.format == "ast":
            adj, x_feature = build_graph_ast(input_ids.cpu().detach().numpy(), self.w_embeddings, self.tokenizer)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings,
                                              window_size=self.args.window_size)
        # 初始化
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)

        if self.args.use_contrastive:
            # 两个增强视图
            feat1 = torch.tensor(node_drop(adj_feature.numpy(), drop_prob=0.2)).to(device).double()
            feat2 = torch.tensor(node_drop(adj_feature.numpy(), drop_prob=0.2)).to(device).double()

            adj1 = torch.tensor(edge_perturb(adj.numpy(), perturb_prob=0.2)).to(device).double()
            adj2 = torch.tensor(edge_perturb(adj.numpy(), perturb_prob=0.2)).to(device).double()

            mask = adj_mask.to(device).double()

            out1 = self.gnn(feat1, adj1, mask)
            out2 = self.gnn(feat2, adj2, mask)

            g1 = torch.mean(out1, dim=1)
            g2 = torch.mean(out2, dim=1)

            c_loss = contrastive_loss(g1, g2, temperature=self.args.temperature)

        # 运行GNN
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        if self.args.gnn == "GraphSAGE":
            # 使用求和池化或均值池化
            graph_embeddings = torch.sum(outputs, dim=1)  # 求和池化，维度为 (batch_size, hidden_size)
            # 或者使用均值池化
            # graph_embeddings = torch.mean(outputs, dim=1)  # 均值池化，维度为 (batch_size, hidden_size)

            logits = self.classifier(graph_embeddings)  # 分类器输入维度为 (batch_size, hidden_size)

        else:
            logits = self.classifier(outputs)

        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            if self.args.use_contrastive:
                loss = loss + self.args.contrastive_weight * c_loss  # 加入对比损失
            return loss, prob
        else:
            return prob
