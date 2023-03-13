from models.selector import *
from utils.util import *
from data_loader import *
from config import get_arguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random


def train_step_clean(opt, train_loader, model_clean, model_backdoor, disen_estimator, optimizer, adv_optimizer,
                         criterion, epoch):
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    losses = AverageMeter()
    disen_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model_clean.train()
    model_backdoor.eval()

    if opt.disentangle:
        for idx, (img, target, indicator) in enumerate(train_loader, start=1):
            if opt.cuda:
                img = img.type(torch.FloatTensor)
                img = img.to(opt.device)

            output1, z_hidden = model_clean(img, True)
            with torch.no_grad():
                output2, r_hidden = model_backdoor(img, True)

            # Train discriminator
            # stop gradient propagation to encoder
            r_hidden, z_hidden = r_hidden.detach(), z_hidden.detach()
            # max dis_loss
            dis_loss = - disen_estimator(r_hidden, z_hidden)
            disen_losses.update(dis_loss.item(), img.size(0))
            adv_optimizer.zero_grad()
            dis_loss.backward()
            adv_optimizer.step()
            # Lipschitz constrain for Disc of WGAN
            disen_estimator.spectral_norm()

    for idx, (img, target, indicator) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.type(torch.FloatTensor)
            img = img.to(opt.device)
            target = target.to(opt.device)

        output1, z_hidden = model_clean(img, True)
        with torch.no_grad():
            output2, r_hidden = model_backdoor(img, True)
            loss_bias = criterion1(output2, target)
            loss_d = criterion1(output1, target).detach()

        r_hidden = r_hidden.detach()
        dis_loss = disen_estimator(r_hidden, z_hidden)

        weight = loss_bias / (loss_d + loss_bias + 1e-8)

        weight = weight * weight.shape[0] / torch.sum(weight)
        loss = torch.mean(weight * criterion1(output1, target))
        if opt.disentangle:
            loss += dis_loss

        prec1, prec5 = accuracy(output1, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Clean Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                 top1=top1, top5=top5))


def train_step_backdoor(opt, train_loader, model_backdoor, optimizer, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_backdoor.train()

    for idx, (img, target, _) in enumerate(train_loader, start=1):
        if opt.cuda:
            img = img.type(torch.FloatTensor)
            img = img.to(opt.device)
            target = target.to(opt.device)

        output = model_backdoor(img)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % opt.print_freq == 0:
            print('Backdoor Epoch[{0}]:[{1:03}/{2:03}] '
                  'loss:{losses.val:.4f}({losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_loader), losses=losses,
                                                                 top1=top1, top5=top5))


def test(opt, test_clean_loader, test_bad_loader, model_clean, criterion, epoch):
    test_process = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    weight_record = np.array([])
    criterion1 = nn.CrossEntropyLoss(reduction='none')

    model_clean.eval()

    for idx, (img, target, indicator) in enumerate(test_clean_loader, start=1):
        if opt.cuda:
            img = img.type(torch.FloatTensor)
            img = img.to(opt.device)
            target = target.to(opt.device)

        with torch.no_grad():
            output = model_clean(img)
            loss = criterion(output, target)
            loss1 = criterion1(output, target)
            weight_record = np.concatenate([weight_record, loss1.cpu().numpy()])

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_clean = [top1.avg, top5.avg, losses.avg]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (img, target, indicator) in enumerate(test_bad_loader, start=1):
        if opt.cuda:
            img = img.type(torch.FloatTensor)
            img = img.to(opt.device)
            target = target.to(opt.device)

        with torch.no_grad():
            output = model_clean(img)
            loss = criterion(output, target)
            loss1 = criterion1(output, target)
            weight_record = np.concatenate([weight_record, loss1.cpu().numpy()])

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    acc_bd = [top1.avg, top5.avg, losses.avg]

    print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
    print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

    # save training progress
    log_root = opt.log_root + '/CBD.csv'
    test_process.append((epoch, acc_clean[0], acc_bd[0], acc_clean[2], acc_bd[2]))
    df = pd.DataFrame(test_process,columns=("Epoch", "Test_clean_acc", "Test_bad_acc", "Test_clean_loss", "Test_bad_loss"))
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')
    return acc_clean, acc_bd


def train(opt):
    # Load models
    print('----------- Model Initialization --------------')
    model_clean, _ = select_model(dataset=opt.dataset, model_name=opt.model_name,
                                                  pretrained=False,
                                                  n_classes=opt.num_class)
    model_clean.to(opt.device)
    model_backdoor, _ = select_model(dataset=opt.dataset, model_name=opt.model_name,
                                 pretrained=False,
                                 n_classes=opt.num_class)
    model_backdoor.to(opt.device)
    hidden_dim = model_clean.nChannels
    disen_estimator = DisenEstimator(hidden_dim, hidden_dim, dropout=0.2)
    disen_estimator.to(opt.device)

    print('Finish Loading Models...')

    # initialize optimizer
    adv_params = list(disen_estimator.parameters())
    adv_optimizer = Adam(adv_params, lr=0.2)
    adv_scheduler = StepLR(adv_optimizer, step_size=20, gamma=0.1)
    optimizer = torch.optim.SGD(model_clean.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay, nesterov=True)
    optimizer_backdoor = torch.optim.SGD(model_backdoor.parameters(), lr=opt.lr, momentum=opt.momentum,
                                     weight_decay=opt.weight_decay, nesterov=True)

    # define loss functions
    if opt.cuda:
        criterion = nn.CrossEntropyLoss().to(opt.device)
    else:
        criterion = nn.CrossEntropyLoss()

    print('----------- Data Initialization --------------')

    _, poisoned_data_loader = get_backdoor_loader(opt)
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    print('----------- Training Backdoored Model --------------')
    for epoch in range(0, 5):
        learning_rate(optimizer, epoch, opt)
        train_step_backdoor(opt, poisoned_data_loader, model_backdoor, optimizer_backdoor, criterion, epoch + 1)
        test(opt, test_clean_loader, test_bad_loader, model_backdoor, criterion, epoch + 1)
    print('----------- Training Clean Model --------------')
    for epoch in range(0, opt.tuning_epochs):
        learning_rate(optimizer, epoch, opt)
        adv_scheduler.step()
        train_step_clean(opt, poisoned_data_loader, model_clean, model_backdoor, disen_estimator, optimizer,
                                   adv_optimizer, criterion, epoch + 1)
        test(opt, test_clean_loader, test_bad_loader, model_clean, criterion, epoch + 1)


def learning_rate(optimizer, epoch, opt):
    if epoch < 20:
        lr = 0.1
    elif epoch < 70:
        lr = 0.01
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, opt):
    if is_best:
        filepath = os.path.join(opt.weight_root, opt.model_name + r'_epoch{}.tar'.format(epoch))
        torch.save(state, filepath)
    print('[info] Finish saving the model')


def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    train(opt)


if __name__ == '__main__':
    main()
