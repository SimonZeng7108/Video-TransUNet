import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import copy
import matplotlib.pylab as plt
import pickle

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def loss_epoch(dataloader, model, dice_loss, optimizer = None):
        running_loss=0.0
        running_bolus_metric=0.0
        running_pharynx_metric=0.0
        len_data=len(dataloader.dataset)
        len_bolus = len_data
        len_pharynx = len_data

        for i_batch, sampled_batch in enumerate(dataloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)

            bolus_loss_ce = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], label_batch[:, 0, :, :].float())
            pharynx_loss_ce = F.binary_cross_entropy_with_logits(outputs[:, 1, :, :], label_batch[:, 1, :, :].float())
            bolus_loss_dice, bolus_dice_metric, num_zero_metric = dice_loss(outputs[:, 0, :, :], label_batch[:, 0, :, :])
            pharynx_loss_dice, pharynx_dice_metric, _ = dice_loss(outputs[:, 1, :, :], label_batch[:, 1, :, :])
            loss_ce = bolus_loss_ce + pharynx_loss_ce
            loss_dice = bolus_loss_dice + pharynx_loss_dice

            loss = 0.5 * loss_ce + 0.5 * loss_dice

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            running_loss += loss
            running_bolus_metric += bolus_dice_metric
            running_pharynx_metric += pharynx_dice_metric
            len_bolus -= num_zero_metric   

        loss=running_loss/float(len_data)
        bolus_metric = running_bolus_metric/float(len_bolus)
        pharynx_metric = running_pharynx_metric/float(len_pharynx)

        return loss, bolus_metric, pharynx_metric


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    from sklearn.model_selection import ShuffleSplit
    from torch.utils.data import Subset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val", transform=None)

    #Split data into train/validation set
    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    indices=range(len(db_train))

    for train_index, val_index in sss.split(indices):
        pass

    train_ds=Subset(db_train, train_index)
    val_ds=Subset(db_val, val_index)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    

    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=15,verbose=1)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))


    loss_history={
        "train": [],
        "val": []}

    metric_history={
        "train": [],
        "val": []}   
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf') 

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:

        model.train()
        train_loss, train_bolus_metric, train_pharynx_metric = loss_epoch(trainloader, model, dice_loss, optimizer)
        logging.info('epoch %d : train_loss : %f, train_bolus_acc: %f, train_pharynx_acc: %f' % (epoch_num, train_loss, train_bolus_metric, train_pharynx_metric))
        train_metrics = (train_bolus_metric + train_pharynx_metric)/2
        loss_history["train"].append(train_loss.detach().cpu().numpy())
        metric_history["train"].append(train_metrics.detach().cpu().numpy())

        model.eval()
        with torch.no_grad():
            val_loss, val_bolus_metric, val_pharynx_metric = loss_epoch(valloader, model, dice_loss)
        logging.info('epoch %d : val_loss : %f, val_bolus_acc: %f, val_pharynx_acc: %f' % (epoch_num, val_loss, val_bolus_metric, val_pharynx_metric))
        val_metrics = (val_bolus_metric + val_pharynx_metric)/2
        loss_history["val"].append(val_loss.detach().cpu().numpy())
        metric_history["val"].append(val_metrics.detach().cpu().numpy())

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + 'best_weights.pth')
            torch.save(model.state_dict(), save_mode_path)

        current_lr = get_lr(optimizer)
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(optimizer):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)


        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            

        vis_epoch = True
        if vis_epoch or epoch_num%100 == 0:
            # plot loss progress
            fig1 = plt.figure(1)
            plt.title("Train-Val Loss")
            plt.plot(range(1,epoch_num+2),loss_history["train"],label="train"if epoch_num == 0 else "", color='cornflowerblue')
            plt.plot(range(1,epoch_num+2),loss_history["val"],label="val"if epoch_num == 0 else "", color='orange')
            plt.ylabel("Loss")
            plt.xlabel("Training Epochs")
            plt.legend()
            fig1.savefig('./test_log/Train-Val Loss.png')
            # plot accuracy progress
            fig2 = plt.figure(2)
            plt.title("Train-Val Accuracy")
            plt.plot(range(1,epoch_num+2),metric_history["train"],label="train"if epoch_num == 0 else "", color='cornflowerblue')
            plt.plot(range(1,epoch_num+2),metric_history["val"],label="val"if epoch_num == 0 else "", color='orange')
            plt.ylabel("Accuracy")
            plt.xlabel("Training Epochs")
            plt.legend()
            fig2.savefig('./test_log/Train-Val Accuracy.png')

        #save history to pickle
        loss_file = open("./test_log/loss.pkl", "wb")
        pickle.dump(loss_history, loss_file)
        loss_file.close()

        metric_file = open("./test_log/metric.pkl", "wb")
        pickle.dump(metric_history, metric_file)
        metric_file.close()





    return "Training Finished!"