import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn import AvgPooling, GraphConv, MaxPooling, SumPooling

import numpy as np
from copy import deepcopy

from src.model.GNNConv import GNNlayer
from src.model.layer_pool import SAGPool, DiffPoolBatchedGraphLayer, BatchedDiffPool, BatchedGraphSAGE


feat_name_dict = {'atomic_num': 9, 'hyb': 6, 'heavydegree': 5, 'heterodegree': 5, 'smarts': 32, 'partialcharge': 0}

def MinMax_normalization(x):
    x_min = torch.min(x, dim=1, keepdim=True)[0]
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    out = (x - x_min) / ((x_max - x_min) + 1e-6)
    return out


def init_params(module):
    if isinstance(module, nn.Linear):
        # nn.init.xavier_normal_(module.weight.data, gain=1.414)
        module.weight.data.normal_(mean=0.0, std=0.1)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.1)
        

class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, act):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, act, d_hidden_feats=d_out_feats*4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x

class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, act, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers-2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = act
    
    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers-2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class Attn_Fusion(nn.Module):
    def __init__(self, d_input):
        super(Attn_Fusion, self).__init__()
        self.mlp = MLP(d_in_feats=d_input*2, d_out_feats=2, n_dense_layers=2, d_hidden_feats=128, act=nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h1, h2):
        h = torch.cat([h1, h2], dim=-1)
        attn = self.mlp(h)
        attn = self.softmax(attn)
        x = h1 * attn[:, 0].unsqueeze(-1) + h2 * attn[:, 1].unsqueeze(-1)

        return x


class Feat_Fusion(nn.Module):
    def __init__(self, dim, mode):
        super(Feat_Fusion, self).__init__()
        self.mode = mode

        if mode == 'cat':
            self.proj = nn.Linear(2*dim, dim)
        elif mode == 'attn':
            self.fusion = Attn_Fusion(d_input=dim)

    def forward(self, h1, h2):
        h = None
        if self.mode == 'mean':
            pass
        elif self.mode == 'max':
            pass
        elif self.mode == 'sum':
            pass
        elif self.mode == 'cat':
            h_cat = torch.cat([h1, h2], dim=-1)
            h = self.proj(h_cat)
        elif self.mode == 'attn':
            h = self.fusion(h1, h2)

        return h


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """

    def __init__(self, args):
        super(ConvPoolBlock, self).__init__()
        self.args = args
        d_hidden = args.hidden_size
        self.conv = GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=args.n_head, alpha=0.05,
                             hop_num=args.n_hop, input_drop=self.args.input_drop, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                             negative_slope=args.leaky_relu, topk_type='local', top_k=args.topk, is_rel=args.is_rel, args=self.args)
        self.pool = SAGPool(d_hidden, ratio=args.pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, angle_h):
        graph.edata['angle_h'] = angle_h
        out = F.relu(self.conv(graph, feature, angle_h))
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, g_out, graph.edata.pop('angle_h')

class GNNBlock(torch.nn.Module):
    """
    A combination of two GNN layers. One for aton-atom graph, and the other for bond-bond graph.
    """

    def __init__(self, args):
        super(GNNBlock, self).__init__()
        self.args = args
        d_hidden = args.hidden_size
        self.gnn1 = GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=args.n_head, alpha=0.05,
                             input_drop=self.args.input_drop, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                             negative_slope=args.leaky_relu, topk_type='local', top_k=args.a2a_topk, is_rel=args.is_rel, args=self.args)
        self.gnn2 = GNNlayer(in_ent_feats=d_hidden, in_rel_feats=d_hidden, out_feats=d_hidden, num_heads=args.n_head, alpha=0.05,
                             input_drop=self.args.input_drop, feat_drop=args.feat_drop,
                             attn_drop=args.attn_drop, negative_slope=args.leaky_relu, topk_type='local', top_k=args.bab_topk,
                             is_rel=args.is_rel, args=self.args)
        self.triplet_emb = TripletEmbedding(d_hidden, nn.ReLU())  # GELU

    def forward(self, g1, g2, node_feat, len_h, angle_h):
        g1_out = F.relu(self.gnn1(g1, node_feat, len_h))

        bond_nodes = g1.edges()
        head_h, end_h = g1_out[bond_nodes[0]], g1_out[bond_nodes[1]]
        triple_h = self.triplet_emb(head_h, len_h, end_h)
        g2_out = F.relu(self.gnn2(g2, triple_h, angle_h))

        return g1_out, g2_out

class GNNModel(nn.Module):
    def __init__(self, args, d_hidden, n_layer=2, n_heads=4, n_ffn_dense_layers=4, feat_drop=0., attn_drop=0., act=nn.ReLU()):
        super(GNNModel, self).__init__()
        self.args = args
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        # Angle embedding
        if self.args.embed_type == 'float':
            self.angle_emb = BondAngleFloatRBF(args=args, bond_angle_float_names=['bond_angle'], embed_dim=d_hidden, input_drop=self.args.input_drop)
        elif self.args.embed_type == 'int':
            self.angle_emb = BondAngleEmbedding(args.n_angle, args.embed_dim, d_hidden, args.input_drop)
        elif self.args.embed_type == 'both':
            self.angle_emb = nn.ModuleList()
            self.angle_emb.append(BondAngleEmbedding(args.n_angle, args.embed_dim, d_hidden, args.input_drop))
            self.angle_emb.append(BondAngleFloatRBF(args=args, bond_angle_float_names=['bond_angle'], embed_dim=d_hidden, input_drop=self.args.input_drop))

        # Molecule GNN
        # SAGPool
        if self.args.pool == 'SAGPool':
            self.mol_T_layers = nn.ModuleList([GNNBlock(args) for _ in range(n_layer)])
            self.pool = SAGPool(d_hidden, ratio=args.pool_ratio)
            if self.args.pool_layer > 1:
                self.pool_layers = nn.ModuleList()
                for _ in range(self.args.pool_layer - 1):
                    self.pool_layers.append(SAGPool(d_hidden, ratio=args.pool_ratio))
        # DiffPool
        elif self.args.pool == 'DiffPool':
            self.mol_T_layers = nn.ModuleList([GNNBlock(args) for _ in range(n_layer)])
            self.pool = DiffPoolBatchedGraphLayer(args=args, input_dim=d_hidden, assign_dim=args.assign_node,
                                                  output_feat_dim=args.pool_dim,
                                                  activation=F.relu, dropout=args.feat_drop,
                                                  aggregator_type='meanpool')

        # No Pooling operation
        else:
            self.mol_T_layers = nn.ModuleList([GNNBlock(args) for _ in range(n_layer)])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        if args.pool_attn:
            self.pool_attn = nn.Linear(d_hidden, 1)
        if self.args.graph_pool == 'sum':
            self.readout = SumPooling()
        elif self.args.graph_pool == 'mean':
            self.readout = AvgPooling()
        elif self.args.graph_pool == 'max':
            self.readout = MaxPooling()
        self.act = act

    def forward(self, a2a_g=None, bab_g=None, node_h=None, bond_h=None):
        if self.args.embed_type == 'both':
            angle_h = self.angle_emb[0](bab_g.edata['bond_angle']) + self.angle_emb[1](bab_g.edata['bond_angle'])
        else:
            angle_h = self.angle_emb(bab_g.edata['bond_angle'])

        hidden_h, hidden_bond_h = [], []
        # SAGPool
        if self.args.pool == 'SAGPool':
            for i in range(self.n_layer):
                node_h, bond_h = self.mol_T_layers[i](a2a_g, bab_g, node_h, bond_h, angle_h)
                if i < self.n_layer - 1:
                    feat_node = self.readout(a2a_g, node_h)
                    feat_bond = self.readout(bab_g, bond_h)
                    hidden_h.append(feat_node)
                    hidden_bond_h.append(feat_bond)
            bond_rep = bond_h
            bab_g, bond_h, _ = self.pool(bab_g, bond_h)
            if self.args.pool_layer > 1:
                for p in range(self.args.pool_layer - 1):
                    bab_g, bond_h, _ = self.pool_layers[p](bab_g, bond_h)
            feat_node = self.readout(a2a_g, node_h)
            feat_bond = self.readout(bab_g, bond_h)
            hidden_h.append(feat_node)
            hidden_bond_h.append(feat_bond)

        # DiffPool
        elif self.args.pool == 'DiffPool':
            for i in range(self.n_layer):
                node_h, bond_h = self.mol_T_layers[i](a2a_g, bab_g, node_h, bond_h, angle_h)
                if i < self.n_layer - 1:
                    feat_node = self.readout(a2a_g, node_h)
                    feat_bond = self.readout(bab_g, bond_h)
                    hidden_h.append(feat_node)
                    hidden_bond_h.append(feat_bond)
            bond_rep = bond_h
            adj, bond_h = self.pool(bab_g, bond_h, angle_h)
            feat_node = self.readout(a2a_g, node_h)
            if self.args.pool_attn:
                a = self.pool_attn(bond_h)
                weights = F.softmax(a, dim=1)
                feat_bond = torch.sum(weights * bond_h, dim=1)
            else:
                feat_bond = torch.mean(bond_h, dim=1)
            hidden_h.append(feat_node)
            hidden_bond_h.append(feat_bond)

        # No pooling operation
        else:
            for i in range(self.n_layer):
                node_h, bond_h = self.mol_T_layers[i](a2a_g, bab_g, node_h, bond_h, angle_h)
                feat_node = self.readout(a2a_g, node_h)
                feat_bond = self.readout(bab_g, bond_h)
                hidden_h.append(feat_node)
                hidden_bond_h.append(feat_bond)
            bond_rep = bond_h

        return hidden_h, hidden_bond_h, bond_rep


    def _device(self):
        return next(self.parameters()).device


class AtomEmbedding(nn.Module):
    def __init__(self, args, d_hidden, input_drop):
        super(AtomEmbedding, self).__init__()
        self.args = args

        if self.args.init_emb:
            self.embed_list = nn.ModuleList()
            for name in feat_name_dict.keys():
                if name == 'partialcharge':
                    continue
                embed = nn.Embedding(feat_name_dict[name], self.args.embed_dim)
                self.embed_list.append(embed)

            centers = np.arange(0, 2, 0.1)
            gamma = 10.
            self.rbf = RBF(centers, gamma, device=self.args.device)
            self.charge_linear = nn.Linear(len(centers), d_hidden)

        self.in_proj = nn.Linear(self.args.embed_dim, d_hidden)
        self.input_dropout = nn.Dropout(input_drop)


    def forward(self, node_feat):
        if self.args.init_emb:
            emb_h = 0.
            for i, name in enumerate(feat_name_dict.keys()):
                if name == 'partialcharge':
                    continue
                emb_h += self.embed_list[i](node_feat[name])
            rbf_x = self.rbf(node_feat['partialcharge'])
            rbf_h = self.charge_linear(rbf_x)
            pair_node_h = self.in_proj(emb_h) + rbf_h
            h = self.input_dropout(pair_node_h)
        else:
            pair_node_h = self.in_proj(node_feat)
            h = self.input_dropout(pair_node_h)
        return h


class BondEmbedding(nn.Module):
    def __init__(self, cut_dist, embed_dim, d_hidden, input_drop):
        super(BondEmbedding, self).__init__()
        self.cut_dist = cut_dist
        self.dist_embed = nn.Embedding(cut_dist + 1, embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_hidden)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, dist_feat):
        x = torch.clip(dist_feat.squeeze(), 1.0, self.cut_dist-1e-6).long()
        edge_h = self.dist_embed(x)
        edge_h = self.in_proj(edge_h)
        return self.input_dropout(edge_h)


class BondAngleEmbedding(nn.Module):
    def __init__(self, n_angle, embed_dim, d_hidden, input_drop):
        super(BondAngleEmbedding, self).__init__()
        self.n_angle = n_angle
        self.angle_unit = torch.FloatTensor([np.pi])[0] / n_angle
        self.angle_embed = nn.Embedding(n_angle + 1, embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_hidden)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, angle_feat):
        angle_domain = angle_feat / self.angle_unit
        x = torch.clip(angle_domain.squeeze(), 1.0, self.n_angle-1e-6).long()
        angle_h = self.angle_embed(x)
        angle_h = self.in_proj(angle_h)
        return self.input_dropout(angle_h)


class TripletEmbedding(nn.Module):
    def __init__(self, d_hidden, act=nn.ReLU()):
        super(TripletEmbedding, self).__init__()
        # self.in_proj = MLP(d_hidden*3, d_hidden, 2, act)
        self.in_proj = nn.Sequential(nn.Linear(d_hidden*3, d_hidden), act)

    def forward(self, head_h, edge_h, end_h):
        triplet_h = torch.cat([head_h, edge_h, end_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        return triplet_h


class Bar_Predictor(nn.Module):
    def __init__(self, d_input, d_output):
        super(Bar_Predictor, self).__init__()
        self.mlp = MLP(d_in_feats=d_input, d_out_feats=128, n_dense_layers=2, act=nn.ReLU(), d_hidden_feats=512)
        self.act = nn.ReLU()
        self.reg = nn.Linear(128, 1)
        self.cls = nn.Linear(128, d_output)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_i, h_j):
        h = torch.cat([h_i, h_j], dim=-1)
        h_hidden = self.act(self.mlp(h))
        reg = self.reg(h_hidden)
        cls = self.softmax(self.cls(h_hidden))

        return reg.flatten(), cls


class Blr_Predictor(nn.Module):
    def __init__(self, d_input, d_output):
        super(Blr_Predictor, self).__init__()
        self.mlp = MLP(d_in_feats=d_input, d_out_feats=128, n_dense_layers=2, act=nn.ReLU(), d_hidden_feats=512)
        self.act = nn.ReLU()
        self.reg = nn.Linear(128, 1)
        self.cls = nn.Linear(128, d_output)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_hidden = self.act(self.mlp(x))
        reg = self.reg(x_hidden)
        cls = self.softmax(self.cls(x_hidden))

        return reg.flatten(), cls


class KSGNN(nn.Module):
    def __init__(self, args, d_hidden=256, n_layer=2, n_heads=4, n_ffn_dense_layers=2, input_drop=0., feat_drop=0.,
                 attn_drop=0., act=nn.ReLU(), readout_mode='mean'):
        super(KSGNN, self).__init__()
        self.args = args
        self.d_hidden = d_hidden
        self.readout_mode = readout_mode

        # Input
        self.node_emb = AtomEmbedding(args, d_hidden, input_drop)
        if self.args.embed_type == 'float':
            self.bond_len_emb = BondFloatRBF(args=args, bond_float_names=['bond_length'], embed_dim=d_hidden, input_drop=input_drop)
        elif self.args.embed_type == 'int':
            self.bond_len_emb = BondEmbedding(args.inner_cutoff, args.embed_dim, d_hidden, input_drop)
        elif self.args.embed_type == 'both':
            self.bond_len_emb = nn.ModuleList()
            self.bond_len_emb.append(BondEmbedding(args.inner_cutoff, args.embed_dim, d_hidden, input_drop))
            self.bond_len_emb.append(BondFloatRBF(args=args, bond_float_names=['bond_length'], embed_dim=d_hidden, input_drop=input_drop))

        # Encode ringerprints, molecular descriptors, and protein sequences
        if args.smiles_feat == 'fp':
            self.fp_encoder = nn.Sequential(
                nn.Linear(512, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden*2),
                nn.ReLU(),
                nn.Linear(d_hidden*2, d_hidden),
                nn.ReLU(),
            )
            self.md_encoder = nn.Sequential(
                nn.Linear(200, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden * 2),
                nn.ReLU(),
                nn.Linear(d_hidden * 2, d_hidden),
                nn.ReLU(),
            )
        else:
            # self.smiles_encoder = nn.Sequential(
            #     nn.Linear(100, d_hidden * 2),
            #     nn.ReLU(),
            #     nn.Linear(d_hidden * 2, d_hidden),
            #     nn.ReLU()
            # )
            smiles_emb = np.load('./dataset/lig_vocab_emb.npy') # nn.Embedding(num_embeddings=65, embedding_dim=args.embed_dim)
            self.smiles_emb = torch.FloatTensor(smiles_emb).to(args.device)
            self.smiles_encoder = nn.Sequential(
                nn.Conv1d(in_channels=300, out_channels=d_hidden, kernel_size=args.smiles_kernel, stride=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_hidden, out_channels=d_hidden * 2, kernel_size=args.smiles_kernel, stride=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=d_hidden * 2, out_channels=d_hidden, kernel_size=args.smiles_kernel, stride=1),
                nn.ReLU(),
            )

        # seq_emb = np.load('../dataset/prot_vocab_emb.npy') # nn.Embedding(num_embeddings=26, embedding_dim=args.embed_dim)
        # self.seq_emb = torch.FloatTensor(seq_emb).to(args.device)
        # self.seq_encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=300, out_channels=d_hidden, kernel_size=args.prot_kernel, stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=d_hidden, out_channels=d_hidden*2, kernel_size=args.prot_kernel, stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=d_hidden*2, out_channels=d_hidden, kernel_size=args.prot_kernel, stride=1),
        #     nn.ReLU(),
        # )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        # self.seq_encoder = nn.Sequential(
        #     nn.Linear(512, d_hidden * 2),
        #     nn.ReLU(),
        #     nn.Linear(d_hidden * 2, d_hidden),
        #     nn.ReLU()
        # )
        if args.ln:
            self.smiles_ln = nn.LayerNorm(normalized_shape=d_hidden)
            self.seq_ln = nn.LayerNorm(normalized_shape=d_hidden)


        # Model
        self.model = GNNModel(args, d_hidden, n_layer, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, act)

        # Prediction module

        if self.args.readout == 'concat':
            in_dim = out_dim = d_hidden * n_layer
        else:
            in_dim = out_dim = d_hidden
        if self.args.readout == 'gru':
            self.out = nn.GRU(in_dim, out_dim, batch_first=True)
            self.out_bond = nn.GRU(in_dim, out_dim, batch_first=True)
        elif self.args.readout == 'lstm':
            self.out = nn.LSTM(in_dim, out_dim, batch_first=True)
            self.out_bond = nn.LSTM(in_dim, out_dim, batch_first=True)
        elif self.args.readout == 'bi-gru':
            self.out = nn.GRU(in_dim, out_dim // 2, bidirectional=True, batch_first=True)
            self.out_bond = nn.GRU(in_dim, out_dim // 2, bidirectional=True, batch_first=True)
        elif self.args.readout == 'bi-lstm':
            self.out = nn.LSTM(in_dim, out_dim // 2, bidirectional=True, batch_first=True)
            self.out_bond = nn.LSTM(in_dim, out_dim // 2, bidirectional=True, batch_first=True)
        elif self.args.readout == 'linear':
            self.out = nn.Linear(args.n_layer, 1)
            self.out_bond = nn.Linear(args.n_layer, 1)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim*3 + 1280, 512),
            nn.Dropout(args.dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # self.apply(lambda module: init_params(module))

    def forward(self, a2a_g=None, bab_g=None, fps=None, mds=None, strings=None, seqs=None, feed_dict=None):

        # Input
        if self.args.init_emb:
            feat = dict()
            for name in feat_name_dict:
                feat[name] = a2a_g.ndata[name]
        else:
            feat = a2a_g.ndata['feat']
        bond_len = a2a_g.edata['bond_len']
        node_h = self.node_emb(feat)
        if self.args.embed_type == 'both':
            bond_h = self.bond_len_emb[0](bond_len) + self.bond_len_emb[1](bond_len)
        else:
            bond_h = self.bond_len_emb(bond_len)

        # Fingerprints, molecular descriptors, and protein sequences
        if self.args.smiles_feat == 'fp':
            x_fps = self.fp_encoder(fps)
            # mds = MinMax_normalization(mds)
            mds = F.normalize(mds, p=2, dim=1)
            x_mds = self.md_encoder(mds)
            x_lig = x_fps + x_mds
        else:
            # x_lig = self.smiles_encoder(strings) #MLP
            smiles_emb = self.smiles_emb[strings]
            x_lig = self.smiles_encoder(smiles_emb.permute(0, 2, 1))
            x_lig = self.pool(x_lig).squeeze()
            x_lig = x_lig.view(-1, self.d_hidden)
        # seq_emb = self.seq_emb[seqs]
        # x_seqs = self.seq_encoder(seq_emb.permute(0, 2, 1))
        # x_seqs = self.pool(x_seqs).squeeze()
        # x_seqs = x_seqs.view(-1, self.d_hidden)

        # x_seqs = self.seq_encoder(seqs)
        if self.args.ln:
            x_lig = self.smiles_ln(x_lig)
            x_seqs = self.seq_ln(x_seqs)

        # Model
        feat, feat_bond, bond_h = self.model(a2a_g=a2a_g, bab_g=bab_g, node_h=node_h, bond_h=bond_h)
        # readout
        readout = None
        if self.args.readout == 'sum':
            readout_node = torch.stack(feat).sum(dim=0)
            readout_bond = torch.stack(feat_bond).sum(dim=0)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'mean':
            readout_node = torch.stack(feat).mean(dim=0)
            readout_bond = torch.stack(feat_bond).mean(dim=0)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'max':
            readout_node = torch.stack(feat).max(dim=0)[0]
            readout_bond = torch.stack(feat_bond).max(dim=0)[0]
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'concat':
            readout_node = torch.concat(feat, dim=-1)
            readout_bond = torch.concat(feat_bond, dim=-1)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'last':
            readout_node = feat[-1]
            readout_bond = feat_bond[-1]
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'linear':
            feat = torch.stack(feat, dim=-1)
            readout_node = self.out(feat).squeeze(dim=0)
            feat_bond = torch.stack(feat_bond, dim=-1)
            readout_bond = self.out(feat_bond).squeeze(dim=0)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'gru':
            feat = torch.stack(feat, dim=1)
            _, hidden = self.out(feat)
            readout_node = hidden.squeeze(dim=0)
            feat_bond = torch.stack(feat_bond, dim=1)
            _, hidden_bond = self.out(feat_bond)
            readout_bond = hidden_bond.squeeze(dim=0)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'lstm':
            feat = torch.stack(feat, dim=1)
            _, (hidden, _) = self.out(feat)
            readout_node = self.out(feat).squeeze(dim=0)
            feat_bond = torch.stack(feat_bond, dim=1)
            _, (hidden_bond, _) = self.out(feat_bond)
            readout_bond = self.out(feat_bond).squeeze(dim=0)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'bi-gru':
            feat = torch.stack(feat, dim=1)
            _, hidden = self.out(feat)
            readout_node = hidden.permute(1, 0, 2).reshape(feat.shape[0], -1)
            feat_bond = torch.stack(feat_bond, dim=1)
            _, hidden_bond = self.out(feat_bond)
            readout_bond = hidden_bond.permute(1, 0, 2).reshape(feat_bond.shape[0], -1)
            readout = torch.cat([readout_node, readout_bond], dim=-1)
        elif self.args.readout == 'bi-lstm':
            feat = torch.stack(feat, dim=1)
            _, (hidden, _) = self.out(feat)
            readout_node = hidden.permute(1, 0, 2).reshape(feat.shape[0], -1)
            feat_bond = torch.stack(feat_bond, dim=1)
            _, (hidden_bond, _) = self.out(feat_bond)
            readout_bond = hidden_bond.permute(1, 0, 2).reshape(feat_bond.shape[0], -1)
            readout = torch.cat([readout_node, readout_bond], dim=-1)

        # Predict
        if self.args.l2_norm:
            readout, x_lig, x_seqs = F.normalize(readout, p=2), F.normalize(x_lig, p=2), F.normalize(x_seqs, p=2)
        readout = torch.concat([readout, x_lig, seqs], dim=-1)
        pred = self.predictor(readout)
        if feed_dict is None:
            return pred

        return pred


    def forward_tune(self, g):

        # Input
        node_h = self.node_emb(g.ndata['begin_end'])
        edge_h = self.edge_emb(g.ndata['edge'])
        triplet_h = self.triplet_emb(node_h, edge_h)

        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h

        # Readout
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)

        #Predict
        pred = self.predictor(readout)
        return pred


class RBF(nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma, device='cpu'):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.FloatTensor(centers), [1, -1]).to(device)
        self.gamma = torch.FloatTensor([gamma]).to(device)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        # self.canters = self.cent
        return torch.exp(-self.gamma * torch.square(x - self.centers))
    
    
class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, args, bond_float_names, embed_dim, input_drop, rbf_params=None):
        super(BondFloatRBF, self).__init__()
        self.args = args
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, args.cutoff, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=self.args.device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)
        self.input_drop = nn.Dropout(input_drop)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            # x = bond_float_features[name]
            rbf_x = self.rbf_list[i](bond_float_features)
            out_embed += self.linear_list[i](rbf_x)
        return self.input_drop(out_embed)

class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, args, bond_angle_float_names, embed_dim, input_drop=0., rbf_params=None):
        super(BondAngleFloatRBF, self).__init__()
        self.args = args
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=self.args.device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)
        self.input_drop = nn.Dropout(input_drop)

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            # x = bond_angle_float_features[name]
            rbf_x = self.rbf_list[i](bond_angle_float_features)
            out_embed += self.linear_list[i](rbf_x)
        return self.input_drop(out_embed)

