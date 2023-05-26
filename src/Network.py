import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from cw_torch.gamma import silverman_rule_of_thumb_sample
from cw_torch.metric import cw, cw_normality
from tqdm import tqdm

from models.resnet18_encoder import *
from models.resnet20_cifar import *


class Net(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000']:
            self.encoder = resnet18(False, args)
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.init_cw_architecture()

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)

        self.fc_frozen = nn.Linear(self.num_features, self.pre_allocate)
        self.fc_frozen.requires_grad_(False)
        
        nn.init.orthogonal_(self.fc.weight)

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        x_enc = self.encode(x)
        if 'cos' in self.args.mode:
            x = F.linear(F.normalize(x_enc, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.args.mode:
            x = self.fc(x_enc)
            x = self.args.temperature * x

        return x, x_enc

    def predict(self, x):
        x_enc = self.encode(x)
        if 'cos' in self.args.mode:
            x = F.linear(F.normalize(x_enc, p=2, dim=-1), F.normalize(self.fc_frozen.weight, p=2, dim=-1))
            x = self.args.temperature * x
        else:
            raise RuntimeError("only cos mode supported atm")
        return x

    def get_cw_loss(self, embed):
        if self.args.cw_architecture == "generator":
            noise = torch.normal(0, 1, [embed.shape[0], self.num_features*2]).to(self.args.device)
            generated = self.cw_architecture(noise)
            gamma = silverman_rule_of_thumb_sample(torch.cat([embed, generated], dim=0))
            return cw(embed, generated, gamma=gamma)

        if self.args.cw_architecture in ["encoder", "encoder_same_dim"]:
            encoded = self.cw_architecture(embed)
            gamma = silverman_rule_of_thumb_sample(torch.cat([encoded], dim=0))
            return cw_normality(encoded, gamma=gamma)


    def replace_fc_weights(self, trainset):
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=self.args.num_workers, pin_memory=True, shuffle=False)
        embed_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                embed = self.encode(data)

                embed_list.append(embed)
                label_list.append(label)
        embedding_list = torch.cat(embed_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        print(f"Replacing embeddings for classes: {torch.unique(label_list).cpu().detach().numpy().tolist()}")
        for class_index in torch.unique(label_list):
            data_index = (label_list == class_index).nonzero()
            embedding_this_class = embedding_list[data_index.squeeze(-1)].mean(0)

            self.fc_frozen.weight.data[class_index] = embedding_this_class

    def cw_architecture_train(self, cw_train):
        self.requires_grad_(not cw_train)
        self.cw_architecture.requires_grad_(cw_train)

    def init_cw_architecture(self):
        if self.args.cw_architecture == "generator":
            self.cw_architecture = nn.Sequential(
                nn.Linear(self.num_features*2, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_features)
            )

        elif self.args.cw_architecture == "encoder":
            self.cw_architecture = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.BatchNorm1d(self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.num_features//2),
                nn.BatchNorm1d(self.num_features//2),
                nn.ReLU(),
                nn.Linear(self.num_features//2, self.num_features//4),
                nn.BatchNorm1d(self.num_features//4),
                nn.Linear(self.num_features//4, 5)
            )

        if self.args.cw_architecture == "encoder_same_dim":
            self.cw_architecture = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.BatchNorm1d(self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.num_features),
                nn.BatchNorm1d(self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.num_features),
                nn.BatchNorm1d(self.num_features),
                nn.Linear(self.num_features, self.num_features)
            )

        self.cw_architecture.requires_grad_(False)
