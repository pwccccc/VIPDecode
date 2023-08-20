import numpy as np
import torch
import scipy.io as scio
from models.Lstm import LSTM
from dataset import Data
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from models.ReSNN import ReSNN
from utils import t_t_split, t_v_split, add_bin, add_generated_data
import json
from models.FCSNN import FCSNN
import os
import cfg
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt


torch.manual_seed(666)


def main(args):
    data_path = args.train_path
    generated_data_path = args.generated_path
    bin_size = args.bin_size
    real_data = add_generated_data(data_path,generated_data_path,bin_size=bin_size)
    test_data = add_generated_data(args.test_path, bin_size = bin_size)
    # test_data = torch.load('./data1 test_data.pth')
    # real_data['sample_label'] = datam['sample_label']
    
    # data = scio.loadmat(data_path)
    # global data

    if generated_data_path == None:
        train_size = 80
        valid_size = 40
        test_size = 40
    else:
        train_size = 160
        valid_size = 80
        test_size = 40
    batch_size = args.batch_size
    
    # real_data = add_bin(data['spk_arr_mat'])
    # real_data = data['spk_arr_mat'][:, 500:1700, :]
    test_accs = []
    for _ in range(20):
        # net = Model(input_size = real_data['data'].shape[2])
        if args.model_name == 'LSTM':

            net = LSTM(real_data['data'].shape[1], real_data['data'].shape[2])
            # train_data, test_data = t_t_split(real_data['spk_arr_mat'], real_data['sample_label'])

            optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay=args.decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            criterion = nn.CrossEntropyLoss()
            train_accuracy = []
            valid_accuracy = []
            
            losses = []
            for epoch in range(args.epoch):
                net.train()
                train1_data, valid_data = t_v_split(real_data['data'],real_data['label'])
                train_acc = 0.
                valid_acc = 0.
                for i in range(3):
                    MyDataset = Data(train1_data['data'][i], train1_data['label'][i])
                    data_loader = DataLoader(MyDataset, batch_size= batch_size, shuffle=True)
                    
                    for idx, batch in enumerate(data_loader):
                        dataTr, label = batch
                        y_pred = net(dataTr)
                        # mean_pred = torch.mean(y_pred,dim=1)
                        loss = criterion(y_pred, (label-1).long())
                        _, pred = torch.max(y_pred.data, dim=1)
                        train_acc += pred.eq((label -1 ).data).cpu().sum()
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                net.eval()
                for i in range(3):
                    dataV = valid_data['data'][i]
                    label = valid_data['label'][i]
                    dataV = torch.tensor(dataV).to(torch.float32)
                    label = torch.tensor(label)
                    _, pred = torch.max(net(dataV).data, dim=1)
                    # _, pred = torch.max(net(dataV).data, dim=1)
                    valid_acc += pred.eq((label -1 ).data).cpu().sum()

                train_correct = train_acc / train_size/3
                valid_correct = valid_acc / valid_size/3
                train_accuracy.append(train_correct)
                valid_accuracy.append(valid_correct)
                if epoch <= 100:

                    scheduler.step()
                print("epoch {} train accuracy = {}, valid accuracy = {}".format(epoch, 100. * train_correct, 100.*valid_correct))
            
            net.eval()
            datat = test_data['data']
            label = test_data['label']
            datat = torch.tensor(datat).to(torch.float32)
            label = torch.tensor(label)
            _, pred = torch.max(net(datat).data, dim=1)
            # _, pred = torch.max(net(datat).data, dim=1)
            test_acc = pred.eq((label -1 ).data).cpu().sum()
            print("test accuracy = {}".format(test_acc/test_size*100.))
            test_accs.append(test_acc.numpy()/test_size*100.)

            

        else:
            net = eval(args.model_name)(real_data['data'].shape[1], real_data['data'].shape[2])
            # train_data, test_data = t_t_split(real_data['spk_arr_mat'], real_data['sample_label'])

            optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay=args.decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            criterion = nn.CrossEntropyLoss()
            train_accuracy = []
            valid_accuracy = []
            
            losses = []
            for epoch in range(args.epoch):
                net.train()
                train1_data, valid_data = t_v_split(real_data['data'],real_data['label'])
                train_acc = 0.
                valid_acc = 0.
                for i in range(3):
                    MyDataset = Data(train1_data['data'][i], train1_data['label'][i])
                    data_loader = DataLoader(MyDataset, batch_size= batch_size, shuffle=True)
                    
                    for idx, batch in enumerate(data_loader):
                        dataTr, label = batch
                        y_pred = net(dataTr)
                        mean_pred = torch.mean(y_pred,dim=1)
                        loss = criterion(mean_pred, (label-1).long())
                        _, pred = torch.max(mean_pred.data, dim=1)
                        train_acc += pred.eq((label -1 ).data).cpu().sum()
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                net.eval()
                for i in range(3):
                    dataV = valid_data['data'][i]
                    label = valid_data['label'][i]
                    dataV = torch.tensor(dataV).to(torch.float32)
                    label = torch.tensor(label)
                    _, pred = torch.max(torch.mean(net(dataV), dim = 1).data, dim=1)
                    # _, pred = torch.max(net(dataV).data, dim=1)
                    valid_acc += pred.eq((label -1 ).data).cpu().sum()

                train_correct = train_acc / train_size/3
                valid_correct = valid_acc / valid_size/3
                train_accuracy.append(train_correct)
                valid_accuracy.append(valid_correct)
                if epoch <= 100:

                    scheduler.step()
                print("epoch {} train accuracy = {}, valid accuracy = {}".format(epoch, 100. * train_correct, 100.*valid_correct))

            
            net.eval()
            datat = test_data['data']
            label = test_data['label']
            datat = torch.tensor(datat).to(torch.float32)
            label = torch.tensor(label)
            _, pred = torch.max(torch.mean(net(datat), dim = 1).data, dim=1)
            # _, pred = torch.max(net(datat).data, dim=1)
            test_acc = pred.eq((label -1 ).data).cpu().sum()
            print("test accuracy = {}".format(test_acc/test_size*100.))
            test_accs.append(test_acc.numpy()/test_size*100.)

    print(np.mean(test_accs))


if __name__ == "__main__":
    args = cfg.parse_args()
    main(args)


