'''
This script from https://pypi.org/project/fair-esm/0.4.2/

# example of data format
# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]

'''

import torch
import esm
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

device = 'cuda:0'
# Prepare data
fpath = './3D_DTI/dataset/smiles_sequence.csv'  #CSAR_sequence_smiles.csv' #
df = pd.read_csv(fpath)
data = [(row['pdbid'], row['sequence']) for i, row in df.iterrows()]

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
model.to(device)
batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
model.eval()  # disables dropout for deterministic results

batch_labels, batch_strs, batch_tokens = batch_converter(data)
data_loader = DataLoader(batch_tokens, batch_size=2, shuffle=False, num_workers=4, drop_last=False)

# Extract per-residue representations
token_representations = []
with torch.no_grad():
    for i, samples in enumerate(data_loader):
        print(i)
        samples = samples.to(device)
        res = model(samples, repr_layers=[33], return_contacts=True)["representations"][33]
        token_representations.append(res.cpu().numpy())
token_representations = np.concatenate(token_representations, axis=0)
print(token_representations.shape)

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sep_rep = dict()
for i, (pdbid, seq) in enumerate(data):
    rep = token_representations[i, 1: min(len(seq) + 1, 1022+1)].mean(0)
    sep_rep[pdbid] = rep.tolist()

json.dump(sep_rep, open('./dataset/pdbbind_seq_esm_1b_rep.json', 'w'))


# # Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), attention_contacts in zip(data, results["contacts"]):
#     plt.matshow(attention_contacts[: len(seq), : len(seq)])
#     plt.title(seq)
#     plt.show()