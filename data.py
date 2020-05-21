import numpy as np
import json

import torch
from torch.utils.data import Dataset


def char_to_idx(t, index):
    char_to_idx = index['char_to_idx']
    indexed = [char_to_idx[x] for x in t]

    return indexed



class TextData(Dataset):
    def __init__(self, data_fpath, index_fpath):
        with open(index_fpath, 'r') as fh:
            self.index = json.load(fh)
            self.ntoks = max(self.index['char_to_idx'].values())
            
        self.data = []
        with open(data_fpath, 'r') as fh:
            for line in fh:
                line = line.strip().split('\t')
                if len(line) != 3:
                    continue
                _, trans, label = line
                trans_one_hot = char_to_idx(trans, self.index)
                self.data.append((trans, trans_one_hot, int(label)))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trans, trans_one_hot, label = self.data[idx]

        return {
            'transcription': trans,
            'trans_one_hot': trans_one_hot,
            'label': label
        }


def collate_text(batch):
    transcriptions = [x['transcription'] for x in batch]
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.int32)

    max_width = 1014 # max([len(x['trans_one_hot']) for x in batch])
    n = len(batch)
    padded_one_hots = np.zeros((n, max_width), dtype=np.int32)

    for padded_one_hot, one_hot in zip(padded_one_hots, [x['trans_one_hot'] for x in batch]):
        padded_one_hot[:len(one_hot)] = one_hot

    return {
        'transcriptions': transcriptions,
        'labels': labels.float(),
        'transcriptions_one_hot': torch.from_numpy(padded_one_hots).long()
    }
