import sys

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
            #print('pred', F.sigmoid(out))
            #print('true', batch['labels'])
            loss = criterion(out, batch['labels'])
            print('loss', loss)
            loss.backward()
            optimizer.step()


train(sys.argv[1], sys.argv[2])
