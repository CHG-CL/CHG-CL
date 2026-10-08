import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from entmax import entmax_bisect

from gMLP import gMLP
from aggregator import *

# device conf
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")


class GateUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GateUnit, self).__init__()
        self.W_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.W_2 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x1, x2):
        g = torch.sigmoid(self.W_1(x1) + self.W_2(x2))
        out = g * x1 + (1 - g) * x2
        return out


class SessionGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0., k=1):
        super(SessionGAT, self).__init__()
        self.gat = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat,
                             dropout=dropout)
        self.k = k

    def forward(self, x):
        B, D = x.shape
        scale = 1.0 / math.sqrt(D)
        attn = torch.einsum("bd, sd -> bs", x, x)
        attn = torch.softmax(attn * scale, dim=1)
        # 为每个会话节点选出余弦距离最小的K个邻居会话节点，即这些邻居节点和当前节点有边连接
        # print("attn' shape: ", attn.shape)
        _, indices = torch.topk(attn, dim=1, k=self.k)
        adj = torch.zeros_like(attn)
        adj.scatter_(1, indices.long(), 1)
        edge_index, _ = dense_to_sparse(adj)
        out = self.gat(x, edge_index)
        return out


class CombineGraph(Module):
    def __init__(self, opt, num_node):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.hop = opt.n_iter
        self.aggr_gate = GateUnit(self.dim, self.dim)
        # Aggregator
        self.attribute_agg = AttributeAggregator(self.dim, self.opt.alpha, opt, self.opt.dropout_attribute)
        self.local_agg = nn.ModuleList()
        self.mirror_agg = nn.ModuleList()
        for i in range(self.hop):
            agg = LocalAggregator(self.dim, self.opt.alpha)
            self.local_agg.append(agg)
            agg = MirrorAggregator(self.dim)
            self.mirror_agg.append(agg)

        # high way net
        self.highway = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.highway1 = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.highway2 = nn.Linear(self.dim * 2, self.dim, bias=False)
        # embeddings
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        # self.w_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim)
        self.glu4 = nn.Linear(self.dim, self.dim)
        self.glu5 = nn.Linear(self.dim, self.dim)

        self.gate = GateUnit(self.dim, self.dim)
        self.sessionGAT = SessionGAT(self.dim, self.dim, heads=opt.sessionGAT_heads, concat=True,
                                     dropout=opt.dropout, k=opt.k)
        self.s_ln = nn.LayerNorm(self.dim, eps=1e-8)
        self.b_ln = nn.LayerNorm(self.dim, eps=1e-8)
        self.linear_transform = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.aggregation = opt.aggregation
        # Multi
        self.activate = F.relu
        dim = self.dim
        self.LN = nn.LayerNorm(dim)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.attention_mlp = nn.Linear(dim, dim)
        self.alpha_w = nn.Linear(dim, 1)
        # 0.2-->0.5
        self.dropout = nn.Dropout(opt.atten_dropout)
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.linear2_1 = nn.Linear(2 * dim, dim, bias=True)
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = int(dim / self.num_attention_heads)
        self.multi_alpha_w = nn.Linear(self.attention_head_size, 1)
        # loss function
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.gmlp = gMLP(d_model=dim, d_ffn=dim * 2, seq_len=opt.max_seq_length,
                         num_layers=opt.gmlp_layers, dropout=opt.gmlp_dropout, norm_eps=opt.layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def Multi_Self_attention(self, q, k, v, sess_len):
        is_dropout = True
        if is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))  # [b,s+1,d]
        else:
            q_ = self.activate(self.attention_mlp(q))

        query_layer = self.transpose_for_scores(q_)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        alpha_ent = self.get_alpha2(query_layer[:, :, -1, :], seq_len=sess_len)

        attention_probs = entmax_bisect(attention_scores, alpha_ent, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        att_v = context_layer.view(*new_context_layer_shape)

        if is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v

        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        x_n = att_v[:, :-1, :]  # [b,s,d]
        return c, x_n

    def get_alpha2(self, x=None, seq_len=70):  # x [b,n,d/n]
        alpha_ent = torch.sigmoid(self.multi_alpha_w(x)) + 1  # [b,n,1]
        alpha_ent = self.add_value(alpha_ent).unsqueeze(2)  # [b,n,1,1]
        alpha_ent = alpha_ent.expand(-1, -1, seq_len, -1)  # [b,n,s,1]
        return alpha_ent

    def add_value(self, value):

        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    def compute_score(self, item_embedding, pos_emb, attribute_embedding, mask, item_weight, h_local):
        # hl = h_local.unsqueeze(1).repeat(1, item_embedding.size(1), 1)
        # hp = item_embedding + pos_emb
        # nh = torch.sigmoid(self.glu1(hp) + self.glu2(hm) + self.glu3(hl))
        # beta = torch.matmul(nh, self.w)
        # beta = beta * mask
        # s_ = beta * hp

        # s_local = torch.sum(hp * mask, 1) / torch.sum(mask, 1)
        # s_global = torch.sum(s_, 1)
        # # s_final = self.gate2(s_global, s_local)
        # g = torch.sigmoid(self.highway(torch.cat([h_local, s_local], dim=-1)))
        # s_local = g * h_local + (1 - g) * s_local
        attribute_embedding = self.gate(torch.sum(item_embedding, 1), attribute_embedding)
        hp = item_embedding + pos_emb
        item_global_embedding = torch.sum(mask * item_embedding, dim=1)
        hm = attribute_embedding.unsqueeze(1).repeat(1, item_embedding.size(1), 1)
        hl = h_local.unsqueeze(1).repeat(1, item_embedding.size(1), 1)
        # hp = item_embedding + pos_emb
        nh = torch.sigmoid(self.glu1(hp) + self.glu2(hm) + self.glu3(hl))
        beta = torch.matmul(nh, self.w)
        beta = beta * mask
        s_final = beta * hp

        s_final_item = torch.sum(hp * mask, 1) / torch.sum(mask, 1)
        s_final = torch.sum(s_final, 1)

        # # s_final = self.gate2(s_global, s_local)
        # g = torch.sigmoid(self.highway(torch.cat([h_local, s_local], dim=-1)))
        # s_local = g * h_local + (1 - g) * s_local
        # s_final = self.gate(item_global_embedding, attribute_embedding)
        s_cross_embedding = self.sessionGAT(item_global_embedding)
        s_final = s_cross_embedding + s_final
        s_final = self.s_ln(s_final)
        item_weight = self.b_ln(item_weight)
        scores = torch.matmul(s_final, item_weight.transpose(1, 0))
        return scores

    def similarity_loss(self, hf, hf_SSL, simi_mask):
        h1 = hf
        h2 = hf_SSL
        h1 = h1.unsqueeze(2).repeat(1, 1, h1.size(1), 1)
        h2 = h2.unsqueeze(1).repeat(1, h2.size(1), 1, 1)
        hf_similarity = torch.sum(torch.mul(h1, h2), dim=3) / self.opt.temp
        loss = -torch.log(torch.softmax(hf_similarity, dim=2) + 1e-8)
        simi_mask = simi_mask == 1
        loss = torch.sum(loss * simi_mask, dim=2)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss

    def tglobal_attention(self, target, k, v, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),
                             self.atten_w0.t())
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    def get_alpha(self, x=None):
        # x[b,1,d]
        alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
        alpha_global = self.add_value(alpha_global)
        return alpha_global  # [b,1,1]

    def compute_score_and_ssl_loss(self, h, h_local, h_mirror, mask, hf_SSL1, hf_SSL2, simi_mask):
        mask = mask.float().unsqueeze(-1)
        batch_size = h.shape[0]
        len = h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        b = self.embedding.weight[1:]

        simi_loss = self.similarity_loss(hf_SSL1, hf_SSL2, simi_mask)

        zeros = torch.cuda.FloatTensor(h.shape[0], 1, self.dim).fill_(0)  # [b,1,d]
        # session_target = torch.cat([h + pos_emb, zeros], 1)  # [b,s+1,d]
        session_target = torch.cat([h + pos_emb, zeros], 1)  # [b,s+1,d]
        sess_len = session_target.shape[1]
        # spars_self_attention
        target_emb, item_embedding = self.Multi_Self_attention(session_target, session_target, session_target, sess_len)
        item_embedding = self.gmlp(item_embedding)
        item_embedding+=h
        # target_attention_mechanism
        q = target_emb  # [b,1,d]
        k = h_mirror
        v = h_mirror
        alpha_line = self.get_alpha(x=target_emb)
        line_c = self.tglobal_attention(q, k, v, alpha_ent=alpha_line)  # [b,1,d]
        c = torch.selu(line_c).squeeze()
        attribute_embedding = (c / torch.norm(c, dim=-1).unsqueeze(1))

        scores = self.compute_score(item_embedding, pos_emb, attribute_embedding, mask, b, h_local)

        return simi_loss, scores

    def forward(self, inputs, adj, last_item_mask, as_items, as_items_SSL, simi_mask):
        # preprocess
        mask_item = inputs != 0
        attribute_num = len(as_items)
        h = self.embedding(inputs)
        h_as = []
        h_as_SSL = []
        as_mask = []
        as_mask_SSL = []
        for k in range(attribute_num):
            nei = as_items[k]
            nei_SSL = as_items_SSL[k]
            nei_emb = self.embedding(nei)
            nei_emb_SSL = self.embedding(nei_SSL)
            h_as.append(nei_emb)
            h_as_SSL.append(nei_emb_SSL)
            as_mask.append(as_items[k] != 0)
            as_mask_SSL.append(as_items_SSL[k] != 0)

        # attribute
        hf_1, hf_2, hf = self.attribute_agg(h, h_as, as_mask, h_as_SSL, as_mask_SSL)

        # GNN
        x = h
        mirror_nodes = hf
        for i in range(self.hop):
            # aggregate neighbor info
            x = self.local_agg[i](x, adj, mask_item)
            mirror_nodes = self.local_agg[i](mirror_nodes, adj, mask_item)

        # highway
        # g = torch.sigmoid(self.highway(torch.cat([h, x], dim=2)))
        # x_dot = g * h + (1 - g) * x

        # hidden
        hidden = x

        # local representation
        h_local = torch.masked_select(x, last_item_mask.unsqueeze(2).repeat(1, 1, x.size(2))).reshape(
            mask_item.size(0), -1)

        # mirror
        h_mirror = mirror_nodes
        # calculate score
        simi_loss, scores = self.compute_score_and_ssl_loss(hidden, h_local, h_mirror, mask_item, hf_1, hf_2, simi_mask)

        return simi_loss, scores


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, opt):
    adj, items, targets, last_item_mask, as_items, as_items_SSL, simi_mask = data
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    last_item_mask = trans_to_cuda(last_item_mask)
    for k in range(opt.attribute_kinds):
        as_items[k] = trans_to_cuda(as_items[k]).long()
        as_items_SSL[k] = trans_to_cuda(as_items_SSL[k]).long()
    targets_cal = trans_to_cuda(targets).long()
    simi_mask = trans_to_cuda(simi_mask).long()

    simi_loss, scores = model(items, adj, last_item_mask, as_items, as_items_SSL, simi_mask)
    # scores = model(items, adj, last_item_mask, as_items, as_items_SSL, simi_mask)
    loss = model.loss_function(scores, targets_cal - 1)
    loss = loss + opt.phi * simi_loss

    return targets, scores, loss


def adjust_learning_rate(optimizer, decay_rate, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * decay_rate
    lr * decay_rate


def train_test(model, opt, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=False)
    for i, data in enumerate(tqdm(train_loader)):
        targets, scores, loss = forward(model, data, opt)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    if opt.decay_count < opt.decay_num:
        model.scheduler.step()
        opt.decay_count += 1

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=int(opt.batch_size),
                                              shuffle=False, pin_memory=False)
    result_20 = []
    hit_20, mrr_20 = [], []
    result_10 = []
    hit_10, mrr_10 = [], []
    for data in test_loader:
        targets, scores, loss = forward(model, data, opt)
        targets = targets.numpy()
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
        for score, target in zip(sub_scores_20, targets):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        sub_scores_10 = scores.topk(10)[1]
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        for score, target in zip(sub_scores_10, targets):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result_20.append(np.mean(hit_20) * 100)
    result_20.append(np.mean(mrr_20) * 100)

    result_10.append(np.mean(hit_10) * 100)
    result_10.append(np.mean(mrr_10) * 100)

    return result_10, result_20
