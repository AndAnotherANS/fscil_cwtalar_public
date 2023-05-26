import torch.optim

import torch.nn as nn
from copy import deepcopy

from .training_loops import *
from utils import *
from data.dataloader.data_utils import *
import seaborn as sns
from src.Network import Net


class PseudoParallelWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)


class FSCILTrainer:
    def __init__(self, args):
        self.args = args
        self.args, self.dataset = set_up_datasets(self.args)

        self.model = Net(self.args)

        self.test_dataloaders = []

        if self.args.device == "cuda":
            if self.args.num_gpu > 1:
                self.model = nn.DataParallel(self.model, None)
            else:
                self.model = PseudoParallelWrapper(self.model)
            self.model = self.model.cuda()
        else:
            self.model = PseudoParallelWrapper(self.model)
            self.model = self.model.cpu()

    def get_optimizer(self, sess):

        if sess == 0:
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        self.args.base_lr,
                                        momentum=0.9,
                                        nesterov=True,
                                        weight_decay=self.args.decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), self.args.incremental_lr)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args, self.dataset)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, self.dataset, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args

        accuracy_matrix = np.zeros([args.sessions, args.sessions])
        for session in range(args.sessions):
            self.model = self.model.train()
            train_set, trainloader, testloader = self.get_dataloader(session)

            print('new classes for this session:\n', np.unique(train_set.targets))
            optimizer, scheduler = self.get_optimizer(session)

            self.test_dataloaders.append(testloader)
            
            if session == 0:
                for epoch in range(args.base_epochs):
                    train_loss, train_acc = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    test_acc = self.test()[0]
                    log_wandb(args, {"Base_Session/train_loss": train_loss,
                                     "Base_Session/train_acc": train_acc,
                                     "Base_Session/test_acc": test_acc})
                    print(f"Session 0 Epoch {epoch}: train loss={train_loss}, train accuracy={train_acc}, test accuracy={test_acc}")
                    scheduler.step()


            else:  # incremental learning sessions
                incremental_train(self.model, trainloader, optimizer, scheduler, args, session)

            if args.mode == "cos":
                self.model.module.replace_fc_weights(train_set)


            self.model = self.model.eval()
            accuracy_matrix[session] = self.test()


        fig, ax = plt.subplots(figsize=(11, 10))
        heatmap = sns.heatmap(accuracy_matrix, annot=True, ax=ax)
        image = wandb.Image(heatmap.get_figure(), caption="Accuracy per task heatmap")

        log_wandb(args, {"Heatmaps": image})



    def test(self):
        accuracies, losses = [], []
        bar = tqdm(self.test_dataloaders)
        bar.set_description("Testing sessions:")
        for session, loader in enumerate(bar):
            ta = test_one_task(self.model.module, loader, session, self.args)
            accuracies.append(ta)
        print(accuracies)

        accuracies = accuracies + [0] * (self.args.sessions - len(accuracies))
        return np.array(accuracies)
