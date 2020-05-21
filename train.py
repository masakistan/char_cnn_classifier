import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import TextData, collate_text
from text_classifier import TextClassifier


def train(data_fpath, index_fpath):
    dataset = TextData(data_fpath, index_fpath)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_text
    )

    criterion = nn.BCEWithLogitsLoss()

    print('number of tokens {}'.format(dataset.ntoks))

    m = TextClassifier(dataset.ntoks)
    optimizer = optim.Adam(
        m.parameters(),
        lr=0.0001
    )

    for eidx in range(10):
        for bidx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            #print(bidx, batch['transcriptions_one_hot'])

            out = m(batch['transcriptions_one_hot'])
            loss = criterion(out, batch['labels'])
            print('epoch:  {}\tstep: {}\tloss: {:.4f}'.format(eidx, bidx, loss.item()))
            print('pred', (F.sigmoid(out).data > 0.5).numpy().astype(np.int))
            print('true', batch['labels'].numpy().astype(np.int))
            loss.backward()
            optimizer.step()


train(sys.argv[1], sys.argv[2])
