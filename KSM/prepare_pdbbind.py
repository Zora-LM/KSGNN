import os
import torch
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import dgl
from src.data.featurizer import mol2graph
from src.data.featurizer import Compound3DKit, Featurizer, gen_feature, cons_lig_pock_graph_with_spatial_context, bond_node_prepare
from rdkit import Chem
from scipy import sparse as sp
import argparse

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
import selfies as sf

###################################################################################
prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]

fea_name_list = ['atomic_num', 'hyb', 'heavydegree', 'heterodegree', 'partialcharge', 'smarts']

# protein
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: i+1 for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 512

# ligands
smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
              "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
              "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
              "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
              "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
              "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
              "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
              "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
max_smiles_len = 100

class ComplexDataset:
    def __init__(self, args, mol_path, save_path=None, set_path=None, save_file=True):
        self.args = args
        self.mol_path = mol_path
        self.save_file = save_file
        self.save_path = save_path

        data = pd.read_csv(set_path)
        if args.DEBUG:
            data = data[:10]
        # print('Dataset size: ', len(data))
        self.pdbid = data['pdbid'].tolist()
        self.affinity = data['-logKd/Ki'].tolist()
        self.smiles_seq = pd.read_csv(args.data_path + '/smiles_sequence.csv')

        self.labels = []
        self.a2a_graphs = []
        self.bab_graphs = []
        if args.smiles_feat == 'fp':
            self.fps = []  #fingerprint
            self.mds = []  # moledular descriptor
        else:
            self.strings = []  ## smiles sequence

        self.seqs = []  ## protein sequence

        self.process_data()

    def process_data(self):
        """ Generate complex interaction graphs. """

        # Check cache file
        if os.path.exists(self.save_path):
            print('The prossed dataset is saved in ', self.save_path)
        else:
            print('Processing raw protein-ligand complex data...')
            for i, (pdb_name, pk) in enumerate(tqdm(zip(self.pdbid, self.affinity))):
                # print('pdb name ', pdb_name)
                a2a_g, bab_g, global_feats = self.build_graph(pdb_name)
                if global_feats is None:
                    continue
                self.a2a_graphs.append(a2a_g)
                self.bab_graphs.append(bab_g)
                self.labels.append(pk)
                if self.args.smiles_feat == 'fp':
                    self.fps.append(global_feats['fp'])
                    self.mds.append(global_feats['md'])
                else:
                    self.strings.append(global_feats['string'])

                self.seqs.append(global_feats['seq'])

            self.labels = np.array(self.labels)
            if self.save_file:
                self.save()

    def build_graph(self, name):
        featurizer = Featurizer(save_molecule_codes=False)

        data = dict()
        # atomic feature generation
        lig_poc_dict = gen_feature(self.mol_path, name, featurizer)  # dict of {coords, features, atoms, edges} for ligand and pocket
        ligand = (lig_poc_dict['lig_fea'], lig_poc_dict['lig_fea_dict'], lig_poc_dict['lig_co'], lig_poc_dict['lig_atoms'], lig_poc_dict['lig_eg'])
        pocket = (lig_poc_dict['pock_fea'], lig_poc_dict['pock_fea_dict'], lig_poc_dict['pock_co'], lig_poc_dict['pock_atoms'], lig_poc_dict['pock_eg'])

        # get complex coods, features, atoms
        mol = cons_lig_pock_graph_with_spatial_context(args, ligand, pocket, add_fea=self.args.add_fea, theta=self.args.cutoff,
                                                       theta2=args.inner_cutoff, keep_pock=False, pocket_spatial=True)
        num_lig_atoms, num_atoms, coords, features, features_dict, bond_nodes, bond_len, atoms = mol

        # get bond node (atom-atom edges), bond length
        # # strategy 1: get bond node (atom-atom edges) according to distance
        bond_nodes, bond_len = bond_node_prepare(mol, self.args.cutoff)
        # # strategy 2: use known edges as bond node
        for n in fea_name_list:
            if n == 'partialcharge':
                features_dict[n] = torch.FloatTensor(features_dict[n])
            else:
                features_dict[n] = torch.LongTensor(features_dict[n])
        features = torch.FloatTensor(features)

        # Fingerprints, molecular descriptors, and protein sequence
        global_feats = {}
        smiles_seq = self.smiles_seq[self.smiles_seq['pdbid'] == name]
        smiles, sequence = smiles_seq['smiles'].tolist()[0], smiles_seq['sequence'].tolist()[0]

        if self.args.smiles_feat == 'fp':
            fp, md = self.mol_preprocess(smiles)
            if fp is None:
                return None, None, None
            global_feats['fp'] = torch.FloatTensor(fp)
            global_feats['md'] = torch.FloatTensor(md)
        else:
            string = self.smiles_string_preprocess(smiles)
            global_feats['string'] = torch.LongTensor(string)

        seq = self.sequence_preprocess(sequence)
        global_feats['seq'] = torch.LongTensor(seq)

        # G1: atom-distance-atom graph
        g = dgl.graph(data=(bond_nodes[:, 0], bond_nodes[:, 1]), num_nodes=num_atoms)
        for n in fea_name_list:
            g.ndata[n] = features_dict[n]
        # g.ndata['feat'] = features
        g.edata['bond_len'] = torch.FloatTensor(bond_len)

        # G2: bond-angle-bond graph
        data['atom_pos'] = np.array(coords, dtype='float32')
        data['edges'] = np.array(bond_nodes, dtype="int64")
        data['bond_len'] = bond_len
        BondAngleGraph_edges, bond_angles, bond_angle_dirs = Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'], dir_type='HT')

        bab_g = dgl.graph(data=(BondAngleGraph_edges[:, 0], BondAngleGraph_edges[:, 1]), num_nodes=len(bond_nodes))
        bab_g.edata['bond_angle'] = torch.FloatTensor(bond_angles)

        return g, bab_g, global_feats


    def mol_preprocess(self, smiles):
        '''Extracting fingerprints and molecular descriptors'''

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp, md = None, None
            # fp = np.random.randint(0, 2, 512)
            # md = np.random.random(200)
        else:
            fp = np.array(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))

            generator = RDKit2DNormalized()
            features_map = generator.process(smiles)
            md = np.array(list(features_map))[1:]
            # print(len(md[np.isnan(md)]))
            md[np.isnan(md)] = 0
        return fp, md

    def sequence_preprocess(self, seq):
        '''
        Encode protein sequence
        '''

        x = np.zeros(max_seq_len)
        for i, ch in enumerate(seq[:max_seq_len]):
            x[i] = seq_dict[ch]

        return x

    def smiles_string_preprocess(self, string):
        '''
        Encode smiles sequence
        '''

        x = np.zeros(max_smiles_len)
        for i, ch in enumerate(string[:max_smiles_len]):
            x[i] = smiles_dict[ch]

        return x


    def save(self):
        """ Save the generated graphs. """
        print('Saving processed complex data...')
        graphs = [self.a2a_graphs, self.bab_graphs]
        if self.args.smiles_feat == 'fp':
            global_feat = [self.fps, self.mds, self.seqs]
        else:
            global_feat = [self.strings, self.seqs]

        with open(self.save_path, 'wb') as f:
            pickle.dump((graphs, self.labels, global_feat), f)

def pairwise_atomic_types(path, processed_dict, atom_types, atom_types_):
    keys = [(i, j) for i in atom_types_ for j in atom_types]
    for name in tqdm(os.listdir(path)):
        if len(name) != 4:
            continue
        ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (path, name, name)))
        pocket = next(pybel.readfile('pdb', '%s/%s/%s_protein.pdb' % (path, name, name)))
        coords_lig = np.vstack([atom.coords for atom in ligand])
        coords_poc = np.vstack([atom.coords for atom in pocket])
        atom_map_lig = [atom.atomicnum for atom in ligand]
        atom_map_poc = [atom.atomicnum for atom in pocket]
        dm = distance_matrix(coords_lig, coords_poc)
        # print(coords_lig.shape, coords_poc.shape, dm.shape)
        ligs, pocks = dist_filter(dm, 12)
        # print(len(ligs),len(pocks))

        fea_dict = {k: 0 for k in keys}
        for x, y in zip(ligs, pocks):
            x, y = atom_map_lig[x], atom_map_poc[y]  # x-atom.atomicnum of ligand, y-atom.atomicnum of pocket
            if x not in atom_types or y not in atom_types_: continue
            fea_dict[(y, x)] += 1

        processed_dict[name]['type_pair'] = list(fea_dict.values())

    return processed_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preprocess for PDBBind")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--DEBUG", action='store_true', default=True, help='Debug mode')
    parser.add_argument("--n_repeat", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='PDBBind2020')
    parser.add_argument('--data_path', default='/media/data2/lm/Experiments/3D_DTI/dataset/', type=str)
    parser.add_argument('--refined_path', default='/original/PDBbind_v2020_refined/refined-set/', type=str, help='refined set path')
    parser.add_argument('--core_path', default='/original/CASF-2016/coreset/', type=str, help='core set path')
    parser.add_argument('--split_type', type=str, default='random_split', help='random split or temporal split')
    parser.add_argument('--smiles_feat', type=str, default='string', help='fringerprint or sequence')
    parser.add_argument("--cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--inner_cutoff", type=int, default=5, help='threshold of atom distance')
    parser.add_argument("--n_angle", type=int, default=6, help='number of angle domains')
    parser.add_argument("--add_fea", type=int, default=0, help='add feature manner, 1, 2, others')
    parser.add_argument("--save_dir", type=str, default='./dataset/processed_{}_debug/{}/{}/', help='Processed dataset path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.save_dir = args.save_dir.format(args.smiles_feat, args.dataset, args.split_type)
    os.makedirs(args.save_dir, exist_ok=True)

    # preprocess train set and val set
    for i in range(0, args.n_repeat):
        print('This is repeat ', i)
        for s in ['train', 'val']:
            ComplexDataset(args, mol_path=args.data_path + args.refined_path,
                           set_path=f'{args.data_path}/dataset_split/{args.split_type}/{s}_repeat{i}.csv',
                           save_path=args.save_dir + f'repeat{i}_{s}_{args.cutoff}_{args.n_angle}_graph.pkl')

    # Preprocess core set
    ComplexDataset(args, mol_path=args.data_path + args.core_path,
                   set_path=f'{args.data_path}/dataset_split/{args.split_type}/test.csv',
                   save_path=args.save_dir + f'test_{args.cutoff}_{args.n_angle}_graph.pkl')
