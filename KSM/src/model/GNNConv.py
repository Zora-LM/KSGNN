import torch
from torch import nn
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from dgl import DGLGraph
import dgl.function as fn
import numpy as np


class LayerNorm(nn.Module):
    """Construct a layernorm module"""
    def __init__(self, num_features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)


class alpha_calculation(nn.Module):
    "Implements FFN equation."
    def __init__(self, dim):
        super(alpha_calculation, self).__init__()
        self.fc_1 = nn.Linear(dim, 128)
        self.fc_2 = nn.Linear(128, 1, bias=False)
        self.init()

    def forward(self, x):
        x = torch.mean(x, dim=0).view(1, -1)
        return self.fc_2(torch.tanh(self.fc_1(x)))[0][0]

    def init(self):
        gain = 1.414
        nn.init.xavier_normal_(self.fc_1.weight, gain=gain)
        # nn.init.xavier_normal_(self.fc_2.weight, gain=gain)


class GNNlayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 in_rel_feats: int,
                 out_feats: int,
                 num_heads: int,
                 alpha,
                 input_drop,
                 feat_drop,
                 attn_drop,
                 topk_type,
                 top_k=-1,
                 negative_slope=0.2,
                 is_rel=False,  # whether the edge has representation
                 args=None):
        """
        :param in_ent_feats:
        :param in_rel_feats:
        :param out_feats:
        :param num_heads:
        :param alpha:
        :param hop_num:
        :param input_drop:
        :param feat_drop:
        :param attn_drop:
        :param topk_type:
        :param top_k:
        :param negative_slope:
        """
        super(GNNlayer, self).__init__()
        self.topk_type = topk_type
        self._in_ent_feats = in_ent_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self._in_rel_feats = in_rel_feats
        self.alpha = alpha
        self.top_k = top_k
        self.att_dim = self._out_feats // self._num_heads
        self.args = args
        self.is_rel = is_rel

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.input_drop = nn.Dropout(input_drop)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.fc_ent_head = nn.Linear(in_ent_feats, out_feats, bias=False)
        self.fc_ent_tail = nn.Linear(in_ent_feats, out_feats, bias=False)
        self.fc_ent = nn.Linear(in_ent_feats, out_feats, bias=False)
        if self.is_rel:
            self.fc_rel = nn.Linear(in_rel_feats, out_feats, bias=False)

        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self.att_dim)), requires_grad=True)
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self.att_dim)), requires_grad=True)
        if self.is_rel:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, self._num_heads, self.att_dim)), requires_grad=True)
        if self.args.layer_norm:
            self.graph_ent_norm = LayerNorm(num_features=in_ent_feats) # entity feature normalization
            if self.is_rel:
                self.graph_rel_norm = LayerNorm(num_features=in_rel_feats)  # relation feature normalization
        else:
            self.graph_ent_norm = Identity()
            if self.is_rel:
                self.graph_rel_norm = Identity()

        # self.fc_ent_out = nn.Linear(out_feats, out_feats, bias=False)
        if self.args.feed_forward:
            self.ent_feed_forward = PositionwiseFeedForward(model_dim=out_feats, d_hidden=4 * out_feats)  # entity feed forward
            if self.args.layer_norm:
                self.ff_ent_norm = LayerNorm(num_features=out_feats)  # entity feed forward normalization
            else:
                self.ff_ent_norm = Identity()
        else:
            self.ent_feed_forward = Identity()

        if args.residual:
            if in_ent_feats != out_feats:
                self.res_fc_ent = nn.Linear(in_ent_feats, out_feats, bias=False)
                nn.init.xavier_normal_(self.res_fc_ent.weight.data, gain=1.414)
            else:
                self.res_fc_ent = Identity()

        self.reset_parameters()
        self.attention_mask_value = -10e15

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # if isinstance(self.res_fc_ent, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc_ent.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.fc_ent_head.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.fc_ent_tail.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.fc_ent.weight.data, gain=1.414)
        # nn.init.xavier_normal_(self.fc_ent_out.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_t, gain=1.414)
        nn.init.xavier_normal_(self.attn_h, gain=1.414)
        if self.is_rel:
            nn.init.xavier_normal_(self.fc_rel.weight.data, gain=1.414)
            nn.init.xavier_normal_(self.attn_r, gain=1.414)


    def forward(self, graph: DGLGraph, ent_embed, rel_embed=None, drop_edge_ids=None):
        ###Attention computation: pre-normalization structure
        graph = graph.local_var()
        # entity embedding
        h = self.graph_ent_norm(ent_embed)
        # ent_feat_head = self.fc_ent_head(self.input_drop(h)).view(-1, self._num_heads, self.att_dim)
        # ent_feat_tail = self.fc_ent_tail(self.input_drop(h)).view(-1, self._num_heads, self.att_dim)
        ent_feat_head = torch.tanh(self.fc_ent_head(self.input_drop(h))).view(-1, self._num_heads, self.att_dim)
        ent_feat_tail = torch.tanh(self.fc_ent_tail(self.input_drop(h))).view(-1, self._num_heads, self.att_dim)

        eh = (ent_feat_head * self.attn_h).sum(dim=-1).unsqueeze(-1)
        et = (ent_feat_tail * self.attn_t).sum(dim=-1).unsqueeze(-1)
        ent_feat = self.fc_ent(self.input_drop(h)).view(-1, self._num_heads, self.att_dim)
        graph.ndata.update({'ft': ent_feat, 'eh': eh, 'et': et})

        def edge_attention(edges):
            return {'e': self.leaky_relu(edges.src['eh'] + edges.dst['et'] + edges.data['er'])}
        def edge_attention_norel(edges):
            return {'e': self.leaky_relu(edges.src['eh'] + edges.dst['et'])}

        # relation embedding
        if self.is_rel:
            h_r = self.graph_rel_norm(rel_embed)
            # rel_feat = self.fc_rel(self.input_drop(h_r)).view(-1, self._num_heads, self.att_dim)
            rel_feat = torch.tanh(self.fc_rel(self.input_drop(h_r))).view(-1, self._num_heads, self.att_dim)
            er = (rel_feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.edata.update({'er': er})
            graph.apply_edges(edge_attention)
        else:
            graph.apply_edges(edge_attention_norel)

        attations = graph.edata.pop('e')
        if drop_edge_ids is not None:
            attations[drop_edge_ids] = self.attention_mask_value

        if self.top_k <= 0:
            graph.edata['a'] = edge_softmax(graph, attations)
        else:
            if self.topk_type == 'local':
                graph.edata['e'] = attations
                attations = self.topk_attention(graph)
                graph.edata['a'] = edge_softmax(graph, attations)  ##return attention scores
            else:
                graph.edata['e'] = edge_softmax(graph, attations)
                graph.edata['a'] = self.topk_attention_softmax(graph)
        # aggregate message using PPR of not
        ent_rst = self.gat(graph=graph)
        ent_rst = ent_rst.flatten(1)

        # ent_rst = self.fc_ent_out(ent_rst)
        if self.args.residual:
            ent_resval = self.res_fc_ent(self.feat_drop(ent_embed))
            ent_rst = ent_resval + ent_rst
        if self.args.feed_forward:
            ent_rst_ff = self.ent_feed_forward(self.ff_ent_norm(ent_rst))
            ent_rst = ent_rst + self.feat_drop(ent_rst_ff)

        return ent_rst


    def gat(self, graph: DGLGraph):
        graph = graph.local_var()

        attentions = graph.edata.pop('a')
        graph.edata['a_temp'] = self.attn_drop(attentions)
        graph.update_all(fn.u_mul_e('ft', 'a_temp', 'm'), fn.sum('m', 'ft'))
        feat = graph.ndata.pop('ft')
        feat = self.feat_drop(feat)
        return feat

    def topk_attention(self, graph: DGLGraph):
        graph = graph.local_var()# the graph should be added a self-loop edge
        def send_edge_message(edges):
            return {'m_e': edges.data['e']}
        def topk_attn_reduce_func(nodes):
            topk = self.top_k
            attentions = nodes.mailbox['m_e']
            neighbor_num = attentions.shape[1]
            if topk > neighbor_num:
                topk = neighbor_num
            topk_atts, _ = torch.topk(attentions, k=topk, dim=1)
            kth_attn_value = topk_atts[:, topk-1]
            return {'kth_e': kth_attn_value}

        # graph.register_reduce_func(topk_attn_reduce_func)
        # graph.register_message_func(send_edge_message)
        graph.update_all(message_func=send_edge_message, reduce_func=topk_attn_reduce_func)
        def edge_score_update(edges):
            scores, kth_score = edges.data['e'], edges.dst['kth_e']
            scores[scores < kth_score] = self.attention_mask_value
            return {'e': scores}
        graph.apply_edges(edge_score_update)
        topk_attentions = graph.edata.pop('e')
        return topk_attentions

    def topk_attention_softmax(self, graph: DGLGraph):
        graph = graph.local_var()
        def send_edge_message(edges):
            return {'m_e': edges.data['e'], 'm_e_id': edges.data['e_id']}
        def topk_attn_reduce_func(nodes):
            topk = self.top_k
            attentions = nodes.mailbox['m_e']
            edge_ids = nodes.mailbox['m_e_id']
            topk_edge_ids = torch.full(size=(edge_ids.shape[0], topk), fill_value=-1, dtype=torch.long)
            if torch.cuda.is_available():
                topk_edge_ids = topk_edge_ids.cuda()
            attentions_sum = attentions.sum(dim=2)
            neighbor_num = attentions_sum.shape[1]
            if topk > neighbor_num:
                topk = neighbor_num
            topk_atts, top_k_neighbor_idx = torch.topk(attentions_sum, k=topk, dim=1)
            top_k_neighbor_idx = top_k_neighbor_idx.squeeze(dim=-1)
            row_idxes = torch.arange(0, top_k_neighbor_idx.shape[0]).view(-1, 1)
            top_k_attention = attentions[row_idxes, top_k_neighbor_idx]
            top_k_edge_ids = edge_ids[row_idxes, top_k_neighbor_idx]
            top_k_attention_norm = top_k_attention.sum(dim=1)
            topk_edge_ids[:, torch.arange(0, topk)] = top_k_edge_ids
            return {'topk_eid': topk_edge_ids, 'topk_norm': top_k_attention_norm}
        # graph.register_reduce_func(topk_attn_reduce_func)
        # graph.register_message_func(send_edge_message)
        graph.update_all(message_func=send_edge_message, reduce_func=topk_attn_reduce_func)
        topk_edge_ids = graph.ndata['topk_eid'].flatten()
        topk_edge_ids = topk_edge_ids[topk_edge_ids >=0]
        mask_edges = torch.zeros((graph.number_of_edges(), 1))
        if torch.cuda.is_available():
            mask_edges = mask_edges.cuda()
        mask_edges[topk_edge_ids] = 1
        attentions = graph.edata['e'].squeeze(dim=-1)
        attentions = attentions * mask_edges
        graph.edata['e'] = attentions.unsqueeze(dim=-1)
        def edge_score_update(edges):
            scores = edges.data['e']/edges.dst['topk_norm']
            return {'e': scores}
        graph.apply_edges(edge_score_update)
        topk_attentions = graph.edata.pop('e')
        return topk_attentions