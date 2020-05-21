import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.Module):
    def __init__(self, ntoks):
        super(TextClassifier, self).__init__()

        self.max_pool = nn.MaxPool1d(3)

        self.embedding = nn.Embedding(ntoks, 128)
        self.conv1 = nn.Conv1d(
            128,
            256,
            7,
        )
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(
            256,
            256,
            7,
        )
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(
            256,
            256,
            3,
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(
            256,
            256,
            3,
        )
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(
            256,
            256,
            3,
        )
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(
            256,
            256,
            3,
        )
        self.bn6 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(8704, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        '''
        input:
            x: N x L
        '''
        x_embed = self.embedding(x)  # N x L x 128
        x_embed = x_embed.permute(0, 2, 1) # N x 128 x L
        x = self.max_pool(F.relu(self.bn1(self.conv1(x_embed))))
        x = self.max_pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.max_pool(F.relu(self.bn6(self.conv6(x))))
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.flatten(x)

        return x

