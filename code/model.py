import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss, MSELoss
from utils import preprocess_features, preprocess_adj
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}

# 带有残差连接的GGNN实现
class ReGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op='mul', alpha_weight=1.0):
        super(ReGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            if self.residual:
                x = x + self.gatedGNN(x.double(),
                                      adj.double()) * mask.double()  # add residual connection, can use a weighted sum
            else:
                x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum/mean and max pooling

        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


# 带有残差连接的GCN实现
class ReGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op="mul", alpha_weight=1.0):
        super(ReGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(GraphConvolution(feature_dim_size, hidden_size, dropout, act=act))
            else:
                self.gnnlayers.append(GraphConvolution(hidden_size, hidden_size, dropout, act=act))
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else:
                if self.residual:
                    x = x + self.gnnlayers[idx_layer](x, adj) * mask  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnnlayers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


# 无残差连接的GGNN实现
class GGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu):
        super(GGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        return x



# GraphConvolution实现
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.double(), self.weight.double())
        output = torch.matmul(adj.double(), support.double())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)

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


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # 输入形状: (batch_size, num_nodes, in_features)
        batch_size, num_nodes, _ = input.size()

        # 线性变换: (batch_size, num_nodes, in_features) -> (batch_size, num_nodes, out_features)
        h = torch.matmul(input, self.W)  # 形状: (batch_size, num_nodes, out_features)
        N = h.size(1)

        # 计算注意力系数
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, h.size(-1)),
                             h.repeat(1, N, 1)], dim=-1)

        # 计算注意力分数 e_ij
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1).view(-1, N, N))  # 形状: (batch_size, num_nodes, num_nodes)

        # 掩码处理
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 加权求和
        h_prime = torch.matmul(attention, h)  # 形状: (batch_size, num_nodes, out_features)

        # 输出结果
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, alpha=0.2, nheads=8, residual=True):
        super(GAT, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.dropout = dropout
        self.nheads = nheads

        # 多头注意力层
        self.attentions = [GATLayer(feature_dim_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 输出注意力层
        self.out_att = GATLayer(hidden_size * nheads, hidden_size, dropout=dropout, alpha=alpha, concat=False)

        # 线性变换层，将多头注意力的输出维度从 hidden_size * nheads 映射到 hidden_size
        self.linear = nn.Linear(hidden_size * nheads, hidden_size)

        self.ln = nn.Linear(hidden_size, hidden_size)
        self.act = nn.functional.relu

        # 添加 out_dim 属性
        self.out_dim = hidden_size  # GAT 的输出维度是 hidden_size

    def forward(self, inputs, adj, mask):
        # 输入形状: (batch_size, num_nodes, feature_dim_size)
        x = inputs
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                # 第一层：多头注意力
                x = F.dropout(x, self.dropout, training=self.training)
                x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # 形状: (batch_size, num_nodes, hidden_size * nheads)
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                # 后续层：输出注意力 + 残差连接
                if self.residual:
                    # 将多头注意力的输出维度从 hidden_size * nheads 映射到 hidden_size
                    x_residual = self.linear(x)  # 形状: (batch_size, num_nodes, hidden_size)
                    x = x_residual + F.dropout(self.out_att(x, adj), self.dropout, training=self.training)
                else:
                    x = F.dropout(self.out_att(x, adj), self.dropout, training=self.training)
        # 线性变换和激活函数
        x = self.act(self.ln(x))  # 形状: (batch_size, num_nodes, hidden_size)
        return x


class GRAND(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, residual=True, diffusion_steps=4):
        super(GRAND, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.diffusion_steps = diffusion_steps
        self.dropout = dropout
        self.out_dim = hidden_size

        self.linear = nn.Linear(feature_dim_size, hidden_size)
        self.weights = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_GNN_layers)])

    def forward(self, x, adj, mask=None):
        h = self.linear(x)

        for layer in range(self.num_GNN_layers):
            diffusion = h
            for _ in range(self.diffusion_steps):
                diffusion = torch.matmul(adj, diffusion)
            h = self.weights[layer](diffusion)

            if self.residual and layer > 0:
                h = h + diffusion

            if layer < self.num_GNN_layers - 1:
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

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob



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

        self.tokenizer = tokenizer
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(feature_dim_size=args.feature_dim_size,
                              hidden_size=args.hidden_size,
                              num_GNN_layers=args.num_GNN_layers,
                              dropout=config.hidden_dropout_prob,
                              residual=not args.remove_residual,
                              att_op=args.att_op)
        elif args.gnn == "GraphSAGE":  # 添加 GraphSAGE 支持
            self.gnn = GraphSAGE(feature_dim_size=args.feature_dim_size,
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_GNN_layers,
                                 dropout=config.hidden_dropout_prob,
                                 aggregator_type='mean',
                                 residual=not args.remove_residual)
        elif args.gnn == "GAT":  # 添加 GAT 支持
            self.gnn = GAT(feature_dim_size=args.feature_dim_size,
                           hidden_size=args.hidden_size,
                           num_GNN_layers=args.num_GNN_layers,
                           dropout=config.hidden_dropout_prob,
                           alpha=0.2,
                           nheads=8,
                           residual=not args.remove_residual)
        elif args.gnn == "GRAND":
            self.gnn = GRAND(feature_dim_size=args.feature_dim_size,
                             hidden_size=args.hidden_size,
                             num_GNN_layers=args.num_GNN_layers,
                             dropout=config.hidden_dropout_prob,
                             residual=not args.remove_residual,
                             diffusion_steps=4)
        else:
            self.gnn = ReGCN(feature_dim_size=args.feature_dim_size,
                             hidden_size=args.hidden_size,
                             num_GNN_layers=args.num_GNN_layers,
                             dropout=config.hidden_dropout_prob,
                             residual=not args.remove_residual,
                             att_op=args.att_op)
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

    def forward(self, input_ids=None, labels=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # 构造图
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings,
                                         window_size=self.args.window_size)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings,
                                              window_size=self.args.window_size)
        # 初始化
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # 运行GNN
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        if self.args.gnn == "GraphSAGE":
            # 使用求和池化或均值池化
            graph_embeddings = torch.sum(outputs, dim=1)  # 求和池化，维度为 (batch_size, hidden_size)
            # 或者使用均值池化
            # graph_embeddings = torch.mean(outputs, dim=1)  # 均值池化，维度为 (batch_size, hidden_size)

            logits = self.classifier(graph_embeddings)  # 分类器输入维度为 (batch_size, hidden_size)
        elif self.args.gnn == "GAT":
            # 使用求和池化或均值池化
            graph_embeddings = torch.sum(outputs, dim=1)  # 求和池化，维度为 (batch_size, hidden_size)
            # 或者使用均值池化
            # graph_embeddings = torch.mean(outputs, dim=1)  # 均值池化，维度为 (batch_size, hidden_size)
            logits = self.classifier(graph_embeddings)  # 分类器输入维度为 (batch_size, hidden_size)
        elif self.args.gnn == "GRAND":
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
            return loss, prob
        else:
            return prob


# modified from https://github.com/saikat107/Devign/blob/master/modules/model.py
class DevignModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DevignModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer

        self.gnn = GGGNN(feature_dim_size=args.feature_dim_size, hidden_size=args.hidden_size,
                         num_GNN_layers=args.num_GNN_layers, num_classes=args.num_classes,
                         dropout=config.hidden_dropout_prob)

        self.conv_l1 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 3).double()
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 1).double()
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2).double()

        self.concat_dim = args.feature_dim_size + args.hidden_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3).double()
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2).double()
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1).double()
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2).double()

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes).double()
        self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes).double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings)
        # initilization
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(device).double()
        # run over GGGN
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(),
                           adj_mask.to(device).double()).double()
        #
        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        prob = self.sigmoid(avg)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
