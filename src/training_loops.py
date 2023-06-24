import torch
import wandb

from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.to(args.device) for _ in batch]

        logits, embed = model(data)
        logits_ = logits[:, :args.base_class]
        ce_loss = F.cross_entropy(logits_, train_label.long())

        total_loss = ce_loss

        acc = count_acc(logits_, train_label)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f}, ce_loss={:.4f}, acc={:.4f}'.format(epoch,
                                                                               lrc,
                                                                               total_loss.item(),
                                                                               acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return tl, ta


def train_encoder(model, trainloader, args):
    tqdm_epoch = tqdm(range(args.first_cw_train_epochs))

    # train cw architecture from classifier
    model.module.cw_architecture_train(True)

    optimizer_cw = torch.optim.Adam(model.module.cw_architecture.parameters(), lr=args.cw_pretraining_lr)
    pretrain_avg = Averager()
    for epoch in tqdm_epoch:
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.to(args.device) for _ in batch]

            _, embed = model(data)

            cw_loss = model.module.get_cw_loss(embed)
            total_loss = cw_loss

            tqdm_epoch.set_description(f"CW pretraining loss {total_loss.item()}")

            pretrain_avg.add(cw_loss.cpu().detach().item())

            optimizer_cw.zero_grad()
            total_loss.backward()
            optimizer_cw.step()

            if hasattr(args, "logging_freq") and epoch % args.logging_freq == 0:
                log_wandb(args, {f"Encoder_training/pretrain_loss": pretrain_avg.item()})
                pretrain_avg.reset()

def incremental_train(model, trainloader, optimizer, scheduler, args, session):

    model = model.train()
    classes = get_classes(session, args)

    # train classifier from data and previous generator
    tqdm_epoch = tqdm(range(args.incremental_epochs))
    tl = Averager()
    ta = Averager()

    model.module.cw_architecture_train(False)

    ce_avg = Averager()
    cw_avg = Averager()
    acc_avg = Averager()

    for epoch in tqdm_epoch:
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.to(args.device) for _ in batch]


            logits_current, embed = model(data)
            logits_previous = model.module.predict(data)

            logits_ = torch.concatenate([logits_previous[:, :np.min(classes)], logits_current[:, classes]], 1)

            ce_loss = F.cross_entropy(logits_, train_label.long())
            cw_loss = model.module.get_cw_loss(embed)
            total_loss = ce_loss + args.incremental_cw_coefficient * cw_loss

            acc = count_acc(logits_, train_label)

            ce_avg.add(ce_loss.cpu().detach().item())
            cw_avg.add(cw_loss.cpu().detach().item())
            acc_avg.add(acc)

            lrc = scheduler.get_last_lr()[0]
            tqdm_epoch.set_description(
                'Session {}, epo {}, lrc={:.4f},total loss={:.4f}, cw_loss={:.4f}, ce_loss={:.4f}, acc={:.4f}'.format(session, epoch,
                                                                                                                 lrc,
                                                                                                                 total_loss.item(),
                                                                                                                 cw_loss.item(),
                                                                                                                 ce_loss.item(),
                                                                                                                 acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if hasattr(args, "logging_freq") and epoch % args.logging_freq == 0:
            base = f"Incremental_sessions/Session_{session}/CW_coeff_{args.incremental_cw_coefficient}_encoder_dim_{args.encoder_latent_dim}"
            log_wandb(args, {base + "/ce_loss": ce_avg.item(),
                             base + "/accuracy": acc_avg.item(),
                             base + "/cw_loss": cw_avg.item()})
            cw_avg.reset()
            ce_avg.reset()
            acc_avg.reset()

    tl = tl.item()
    ta = ta.item()
    return tl, ta


def test_one_task(model, testloader, session, args):
    ta = Averager()
    model = model.eval()


    classes = get_classes_cumulative(session, args)

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, train_label = [_.to(args.device) for _ in batch]

            train_label = train_label - classes[0]
            logits = model.predict(data)
            logits_ = logits[:, classes]

            acc = count_acc(logits_, train_label)

            ta.add(acc)

    ta = ta.item()
    return ta


def get_classes(session, args):
    if session == 0:
        classes = np.arange(0, args.base_class)
    else:
        classes = np.arange(args.base_class + (session - 1) * args.way, args.base_class + session * args.way)
    return classes

def get_classes_cumulative(session, args):
    classes = np.arange(0, args.base_class + session * args.way)
    return classes

