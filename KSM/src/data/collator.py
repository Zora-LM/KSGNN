import os
import dgl
import torch
import numpy as np
import pandas as pd
import pickle
import random
import json

random.seed(22)

from src.data.featurizer import mask_geognn_graph
from torch.utils.data import Dataset


prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]


class MoleculeDataset(Dataset):
    def __init__(self, args, graph_path, index=None, train_val='train'):
        self.args = args
        self.graph_path = graph_path
        graphs, labels, global_feats = self.load()

        if 'csar' in train_val:
            prot_feat_dict = json.load(open('./dataset/csar_seq_esm_1b_rep.json','r'))
            data = pd.read_csv(args.data_path + '/CSAR_sequence_smiles.csv')
        else:
            prot_feat_dict = json.load(open('./dataset/pdbbind_seq_esm_1b_rep.json','r'))
            data = pd.read_csv(args.data_path + f'/dataset_split/random_split/{train_val}.csv')
        if args.debug:
            data = data[:10]


        prot_feat = [torch.FloatTensor(prot_feat_dict[row['pdbid']]) for _, row in data.iterrows()]
        global_feats[1] = prot_feat

        if index is not None:
            graphs[0], graphs[1] = list(np.array(graphs[0])[index]), list(np.array(graphs[1])[index])
            labels = list(np.array(labels[index]))
            global_feats[0], global_feats[1] = list(np.array(global_feats[0])[index]), list(np.array(global_feats[1])[index])

        self.a2a_graphs, self.bab_graphs = graphs
        if args.smiles_feat == 'fp':
            self.fps, self.mds, self.seqs = global_feats
        else:
            self.strings, self.seqs = global_feats
            # self.strings = [s.float() for s in self.strings]
            # self.seqs = [s.float() for s in self.seqs]
        self.labels = labels

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return graphs and label. """
        if self.args.smiles_feat == 'fp':
            return self.a2a_graphs[idx], self.bab_graphs[idx], self.labels[idx], \
                   self.fps[idx], self.mds[idx], self.seqs[idx]
        else:
            return self.a2a_graphs[idx], self.bab_graphs[idx], self.labels[idx], \
                   self.strings[0], self.seqs[idx]

    def load(self):
        """ Load the generated graphs. """
        print('Loading processed complex data...')
        with open(self.graph_path, 'rb') as f:
            graphs, labels, global_feats = pickle.load(f)
        return graphs, labels, global_feats


def preprocess_batch(batch_num, data_list, ssl_tasks=None):
    batch_num = np.concatenate([[0], batch_num], axis=-1)
    cs_num = np.cumsum(batch_num)

    Ba_bond_i, Ba_bond_j, Ba_bond_angle, Bl_bond, Bl_bond_length = [], [], [], [], []
    for i, data in enumerate(data_list):
        bond_node_count = cs_num[i]
        if 'Bar' in ssl_tasks:
            Ba_bond_i.append(data['Ba_bond_i'] + bond_node_count)
            Ba_bond_j.append(data['Ba_bond_j'] + bond_node_count)
            Ba_bond_angle.append(data['Ba_bond_angle'])
        if 'Blr' in ssl_tasks:
            Bl_bond.append(data['Bl_bond_node'] + bond_node_count)
            Bl_bond_length.append(data['Bl_bond_length'])

    feed_dict = dict()
    if 'Bar' in ssl_tasks:
        feed_dict['Ba_bond_i'] = torch.LongTensor(np.concatenate(Ba_bond_i, 0).reshape(-1))
        feed_dict['Ba_bond_j'] = torch.LongTensor(np.concatenate(Ba_bond_j, 0).reshape(-1))
        feed_dict['Ba_bond_angle'] = torch.FloatTensor(np.concatenate(Ba_bond_angle, 0).reshape(-1, 1))
    if 'Blr' in ssl_tasks:
        feed_dict['Bl_bond'] = torch.LongTensor(np.concatenate(Bl_bond, 0).reshape(-1))
        feed_dict['Bl_bond_length'] = torch.FloatTensor(np.concatenate(Bl_bond_length, 0).reshape(-1, 1))

    # add_factors = np.concatenate([[cs_num[i]] * batch_num_target[i] for i in range(len(cs_num) - 1)], axis=-1)
    return feed_dict


class Collator_fn(object):
    def __init__(self, args, training=False):
        self.args = args
        self.training = training

    def __call__(self, samples):
        '''
        Generate batched a2a graphs and bab graphs
        '''

        batched_fps, batched_mds, batched_strings = None, None, None
        if self.args.smiles_feat == 'fp':
            a2a_graphs, bab_graphs, labels, fps, mds, seqs = map(list, zip(*samples))
            batched_fps = torch.stack(fps)
            batched_mds = torch.stack(mds)
        else:
            a2a_graphs, bab_graphs, labels, strings, seqs = map(list, zip(*samples))
            batched_strings = torch.stack(strings)

        batched_bab_graph = dgl.batch(bab_graphs)
        pk_values = torch.FloatTensor(labels)
        batched_seqs = torch.stack(seqs)

        if self.training & self.args.is_mask:
            masked_a2a_graphs = []
            for g in a2a_graphs:
                p = random.random()
                if self.args.p > p:
                    masked_a2a_g = mask_geognn_graph(g, mask_ratio=self.args.mask_ratio)
                    masked_a2a_graphs.append(masked_a2a_g)
                else:
                    masked_a2a_graphs.append(g)
            batched_a2a_graph = dgl.batch(masked_a2a_graphs)
        else:
            batched_a2a_graph = dgl.batch(a2a_graphs)

        return batched_a2a_graph, batched_bab_graph, pk_values, batched_fps, batched_mds, batched_strings, batched_seqs

