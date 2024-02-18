import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from newtools.dataset_udd import DatasetTrain,DatasetVal
from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from tqdm import tqdm, trange
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import time

from metrics import *
from torchvision  import datasets, transforms


if __name__ == "__main__":
    # NOTE! NOTE! change this to not overwrite all log data when you train the model:
    # network = DeepLabV3(model_id=1, project_dir="E:/master/master1/RSISS/deeplabv3/deeplabv3").cuda()
    # x = Variable(torch.randn(2,3,256,256)).cuda() 
    # print(x.shape)
    # y = network(x)
    # print(y.shape)
    # model_id = "terrace"
    model_id = "bisnet_Rlow1"

    num_epochs = 200
    batch_size = 4
    learning_rate = 0.001

    def parse_args():
        parse = argparse.ArgumentParser()
        parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
        parse.add_argument('--port', dest='port', type=int, default=44554,)
        parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
        parse.add_argument('--finetune-from', type=str, default=None,)
        return parse.parse_args()

    args = parse_args()
    cfg = cfg_factory[args.model]
    network = model_factory[cfg.model_type](8)
    network.cuda()
    #network.load_state_dict(torch.load("training_logs/checkpoint/model_terrace_epoch_101.pth"))
    # network.load_state_dict(torch.load("training_logs/model_1/checkpoints/model_1_epoch_9.pth"))
    running_metrics_val = runningScore(8)

    # 数据增强
    transform_list = [
    transforms.CenterCrop(330),
    transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
    transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
    transforms.Resize((512,512), interpolation=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.Pad(10),
    transforms.RandomCrop((330,330)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]
 
    transform_compose = transforms.Compose(transform_list)

    # train_dataset = DatasetTrain(udd_data_path="/mnt/datasets/Terrace/Terrace/Terrace",
    #                             udd_meta_path="/mnt/datasets/Terrace/Terrace/Terrace",transform = transform_compose)
    # val_dataset = DatasetVal(udd_data_path="/mnt/datasets/Terrace/Terrace/Terrace",
    #                             udd_meta_path="/mnt/datasets/Terrace/Terrace/Terrace")

    train_dataset = DatasetTrain(udd_data_path="/mnt/datasets/Rlow1/Rlow",
                                udd_meta_path="/mnt/datasets/Rlow1/Rlow",transform = transform_compose)
    val_dataset = DatasetVal(udd_data_path="/mnt/datasets/Rlow1/Rlow",
                                udd_meta_path="/mnt/datasets/Rlow1/Rlow")

    num_train_batches = int(len(train_dataset)/batch_size)
    num_val_batches = int(len(val_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=1,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=1,drop_last=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # with open("D:/BaiduNetdiskDownload/cityscapes/class_weights.pkl", "rb") as file: # (needed for python3)
    #     class_weights = np.array(pickle.load(file))
    # class_weights = torch.from_numpy(class_weights)
    # class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses_train = []
    epoch_losses_val = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, num_epochs))

        ############################################################################
        # train:
        ############################################################################
        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs) in tqdm(enumerate(train_loader)):
            #current_time = time.time()

            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            # print(imgs.shape)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))
            # print(label_imgs.shape)
            outputs,*outputs_aux = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
            # print(outputs)
            # print("shape of label_imgs: ",label_imgs.shape)
            # print("shape of outputs: ",outputs.shape)

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

            #print (time.time() - current_time)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % "training_logs", "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_bisnet_Rlow1_losses_train.png" % "training_logs")
        plt.close(1)

        # 评估
         
        pre_label = outputs.max(dim=1)[1].data.cpu().numpy()
        true_label = label_imgs.data.cpu().numpy()
        running_metrics_val.update(true_label, pre_label)

        metrics = running_metrics_val.get_scores()
        for k, v in metrics[0].items():
            print(k, v)
        train_miou = metrics[0]['mIou: ']

        print ("####")

        ############################################################################
        # val:
        ############################################################################
        network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs, img_ids) in tqdm(enumerate(val_loader)):
            with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

                outputs,*outputs_aux = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                loss = loss_fn(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % "training_logs", "wb") as file:
            pickle.dump(epoch_losses_val, file)
        print ("val loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_bisnet_Rlow1_losses_val.png" % "training_logs")
        plt.close(1)

        # 评估
        def evaluate(model):
            net = model.eval()
            running_metrics_val = runningScore(8)
            eval_loss = 0
            prec_time = datetime.now()

            network.evaluate()
            for j, sample in enumerate(val_data):
                valImg = Variable(sample['img'].to(device))
                valLabel = Variable(sample['label'].long().to(device))

                outputs = net(valImg)
                outputs = F.log_softmax(outputs, dim=1)
                loss = criterion(outputs, valLabel)
                eval_loss = loss.item() + eval_loss
                pre_label = outputs.max(dim=1)[1].data.cpu().numpy()
                true_label = valLabel.data.cpu().numpy()
                running_metrics_val.update(true_label, pre_label)
            metrics = running_metrics_val.get_scores()
            for k, v in metrics[0].items():
                print(k, v)

            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prec_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
            print(time_str)  

        # save the model weights to disk:
        if epoch % 5 == 0:
            checkpoint_path = "training_logs/model/bisnet_Rlow1" + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
            torch.save(network.state_dict(), checkpoint_path)
