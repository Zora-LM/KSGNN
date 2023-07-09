import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


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

def get_sentences(df):
    lig_corpus, prot_corpus = [], []
    for i, row in df.iterrows():
        smiles, seq = row['smiles'], row['sequence']
        smiles_sentence = [str(smiles_dict[w]) for w in smiles]
        seq_sentence = [str(seq_dict[w]) for w in seq]
        lig_corpus.append(smiles_sentence)
        prot_corpus.append(seq_sentence)

    return lig_corpus, prot_corpus

def get_word_emb(sentences, length, vector_size):
    feats = np.random.rand(length, vector_size)
    model = Word2Vec(sentences, vector_size=vector_size, window=8, sg=1, min_count=10, workers=10, hs=0, negative=6)
    vectors = model.wv.vectors
    key2idx = model.wv.key_to_index
    key2idx = np.array([[int(k), key2idx[k]] for k in key2idx])
    feats[key2idx[:, 0]] = vectors[key2idx[:, 1]]

    return feats


if __name__ == '__main__':
    fpath = './3D_DTI/dataset/'
    train_set = '/original/PDBbind_v2020_refined/refined-set/' # using the training set to learn vocab vectors
    data = os.listdir(fpath + train_set)
    data = [p for p in data if len(p) == 4]
    df = pd.read_csv(fpath + '/smiles_sequence.csv')
    df = df[df['pdbid'].isin(data)]
    df = df.reset_index(drop=True)

    lig_corpus, prot_corpus = get_sentences(df)
    lig_emb = get_word_emb(lig_corpus, length=len(smiles_dict) + 1, vector_size=300)
    prot_emb = get_word_emb(prot_corpus, length=len(seq_dict) + 1, vector_size=300)
    np.save('../dataset/lig_vocab_emb', lig_emb)
    np.save('../dataset/prot_vocab_emb', prot_emb)

    print('Done!!!')

