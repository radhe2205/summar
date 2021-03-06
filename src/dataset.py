import torch
from torch.utils.data import Dataset


class SummaryDataset(Dataset):
    def __init__(self, texts, summaries, wordtoidx, text_len, summary_len):
        super(SummaryDataset, self).__init__()
        self.wordtoidx = wordtoidx # TODO: Make sure to add index of <start> and <end> token
        self.text_idxes = torch.empty(len(texts), text_len).long().fill_(wordtoidx["<padding>"])
        self.summary_idxes = torch.empty(len(texts), summary_len).long().fill_(wordtoidx["<padding>"])

        for i, (text, summary) in enumerate(zip(texts, summaries)):
            for j, word in enumerate(text.split()):
                if word not in wordtoidx:
                    word = "<unk>"
                self.text_idxes[i,j] = wordtoidx[word]
            for j, word in enumerate(summary.split()):
                if word not in wordtoidx:
                    word = "<unk>"
                self.summary_idxes[i,j] = wordtoidx[word]

    def __len__(self):
        return self.text_idxes.shape[0]

    def __getitem__(self, item):
        return self.text_idxes[item], self.summary_idxes[item]


