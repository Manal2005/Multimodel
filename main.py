import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time
from random import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

start_time = time.time()
print(start_time)

if __name__ == '__main__':

    random_name = str(random())
    random_seed = 336
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt = parse_opts()
    n_folds = 5
    test_accuracies = []

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #pretrained = opt.pretrain_path != 'None'

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    for fold in range(n_folds):

        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        torch.manual_seed(opt.manual_seed)
        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)

        if not opt.no_train:
            training_data = get_training_set(opt)
            train_loader = torch.utils.data.DataLoader(training_data, batch_size = opt.batch_size, shuffle=True)
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
        if not opt.no_val:
          validation_data = get_validation_set(opt)

          val_loader = torch.utils.data.DataLoader(
                       validation_data,
                       batch_size=opt.batch_size,
                       shuffle=False,
                       num_workers=opt.n_threads,
                       pin_memory=True)

        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
          #torch.serialization.add_safe_globals([np.dtype])
          checkpoint = torch.load(opt.resume_path)
          assert opt.arch == checkpoint['arch']
          best_prec1 = checkpoint['best_prec1']
          opt.begin_epoch = checkpoint['epoch']
          model.load_state_dict(checkpoint['state_dict'])

        for i in range(opt.begin_epoch, opt.n_epochs + 1):
          if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger)
            state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
            save_checkpoint(state, False, opt, fold)
          if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best, opt, fold)
            plot_overfitting_graph(train_losses, val_losses)

          if opt.test:
            test_data = get_test_set(opt)
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+'_best'+str(fold)+'.pth')
            model.load_state_dict(best_state['state_dict'])

            test_loader = torch.utils.data.DataLoader(
                              test_data,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              num_workers=opt.n_threads,
                              pin_memory=True)
            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt)
            print("Number of batches in data loader:", len(test_loader))

            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
              f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1)

    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
      f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')


time = time.time() - start_time
print(f'time：{time}')
