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
    if model.module.last_cw_encoder is not None:
        args.first_cw_train_epochs = 3000
    tqdm_epoch = tqdm(range(args.first_cw_train_epochs))

    # train cw architecture from classifier
    model.module.encoder_train(True)

    optimizer_cw = torch.optim.Adam(model.module.cw_encoder.parameters(), lr=args.cw_pretraining_lr)
    pretrain_avg = Averager()
    for epoch in tqdm_epoch:
        for i, batch in enumerate(trainloader, 1):
            data, train_label = [_.to(args.device) for _ in batch]

            _, embed = model(data)

            cw_loss = model.module.get_cw_loss(embed)
            l1_loss = model.module.get_l1_loss()
            total_loss = cw_loss + args.l1_coeff * l1_loss

            tqdm_epoch.set_description(f"CW pretraining loss {total_loss.item()}")

            pretrain_avg.add(cw_loss.cpu().detach().item())

            optimizer_cw.zero_grad()
            total_loss.backward()
            optimizer_cw.step()

            if hasattr(args, "logging_freq") and epoch % args.logging_freq == 0:
                log_wandb(args, {f"Encoder_training/pretrain_loss": pretrain_avg.item()})
                pretrain_avg.reset()

    model.module.store_previous_encoder()
    model.module.encoder_train(False)

def incremental_train(model, trainloader, optimizer, scheduler, args, session):

    model = model.train()
    classes = get_classes(session, args)

    # train classifier from data and previous encoder
    tqdm_epoch = tqdm(range(args.incremental_epochs))
    tl = Averager()
    ta = Averager()


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


    train_encoder(model, trainloader, args)

    tl = tl.item()
    ta = ta.item()
    return tl, ta


def test_tasks_up_to_session(model, testloader, session, args):
    accuracy_counters = [Averager() for _ in range(session + 1)]
    model = model.eval()


    classes, cls_max = get_classes_cumulative(session, args)

    logits_all, targets_all = [], []

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, train_label = [_.to(args.device) for _ in batch]

            logits = model.predict(data)
            logits_all.append(logits[:, :cls_max].detach().cpu())
            targets_all.append(train_label.detach().cpu())


    logits_all = torch.concatenate(logits_all, 0)
    targets_all = torch.concatenate(targets_all, 0)

    for sess_cls, acc_counter_sess in zip(classes, accuracy_counters):
        logits_sess, targets_sess = logits_all[torch.isin(targets_all, sess_cls)], targets_all[torch.isin(targets_all, sess_cls)]

        acc = count_acc(logits_sess, targets_sess)

        acc_counter_sess.add(acc)
    return np.array([ta.item() for ta in accuracy_counters])


def get_classes(session, args):
    if session == 0:
        classes = np.arange(0, args.base_class)
    else:
        classes = np.arange(args.base_class + (session - 1) * args.way, args.base_class + session * args.way)
    return classes

def get_classes_cumulative(session, args):
    classes = [torch.arange(0, args.base_class)] + \
              [torch.arange(args.base_class + i * args.way, args.base_class + (i+1) * args.way) for i in range(session)]
    return classes, args.base_class + session * args.way

