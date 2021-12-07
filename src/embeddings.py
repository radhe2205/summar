import json

import numpy as np
import torch
from torch import nn


class GloveEmbedding(nn.Module):
    def __init__(self, embedding_dim, embedding_path = None, reload = False):
        super(GloveEmbedding, self).__init__()

        self.total_words = 400003
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(400004, embedding_dim, padding_idx=-1).requires_grad_(False)
        self.beg_end_marker = nn.Embedding(2, embedding_dim).requires_grad_(True)
        self.start_idx = 400001
        self.end_idx = 400002
        # self.start_emb = nn.Parameter(torch.randn(embedding_dim))
        # self.end_emb = nn.Parameter(torch.randn(embedding_dim))

        # Whether the embedding be loaded from file OR model.
        if reload:
            if embedding_path is not None:
                embedding_vec, wordtoidx = load_embeddings(embedding_path, embedding_dim)
                self.wordtoidx = wordtoidx
                with torch.no_grad():
                    self.embeddings.weight.data = embedding_vec

    def get_embeddings(self, idxes):
        idxes[idxes == -1] = self.embeddings.padding_idx
        embedding_vec = self.embeddings(idxes)
        embedding_vec[idxes == self.start_idx] = self.beg_end_marker(torch.tensor(0).to(idxes.device))
        embedding_vec[idxes == self.end_idx] = self.beg_end_marker(torch.tensor(1).to(idxes.device))
        return embedding_vec

def load_embeddings(embedding_path, embedding_dim = 300):
    embedding_vec = torch.zeros((400004, embedding_dim))
    wordtoidx = {}
    word_count = 0
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            embedding_vec[word_count] = torch.from_numpy(np.asarray(vals[1:], "float32"))
            wordtoidx[word] = word_count
            word_count += 1
    print("Finished loading embeddings")
    wordtoidx["<start>"] = 400001
    wordtoidx["<end>"] = 400002
    return embedding_vec, wordtoidx

def save_vocab(wordtoidx, vocab_path):
    with open(vocab_path, "w") as f:
        f.write(json.dumps(wordtoidx))

def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = f.read()
        vocab = json.loads(vocab)
    return vocab
