from sklearn.model_selection import KFold,ShuffleSplit,train_test_split
import numpy as np
import scipy.io as scio
import torch


def t_t_split(data, label):
    train_data = dict()
    test_data = dict()
    k = data.shape[0]//8
    for i in range(8):

        train_x,  test_x,train_y, test_y = train_test_split(data[k*i:k*i+k], label[k*i:k*i+k], test_size = 0.25)
        if len(train_data) == 0:
            train_data["data"] = train_x
            train_data["label"] = train_y
            test_data["data"] = test_x
            test_data["label"] = test_y

        else:
            train_data["data"] = np.append(train_data["data"], train_x, axis=0)
            train_data["label"] = np.append(train_data["label"], train_y)
            test_data["data"] = np.append(test_data["data"],test_x, axis=0)
            test_data["label"] = np.append(test_data["label"],test_y)
                
        # kf = KFold(n_splits=4, shuffle=True)
        # kf2 = KFold(n_splits=3, shuffle=True)
        # for train,test in kf.split(data):
        #     test_data.append((data[test], label[test]))
        #     train_data.append((data[train], label[train]))
        #     for trainOn , valid in kf2.split((data[key_list[2]])[train]):
        #         train_data.append((data[key_list[0]][trainOn], data[key_list[2]][trainOn]))
        #         valid_data.append((data[key_list[0]][valid], data[key_list[2]][valid]))

    return train_data,  test_data

def t_v_split(data, label):
    train_data = dict()
    valid_data = dict()
    k = data.shape[0]//8
    train_data['data'],train_data['label'], valid_data['data'], valid_data['label'] = [],[],[],[]
    kf = KFold(n_splits=3, shuffle=True)
    for i in range(8):
        if len(train_data['data'])==0:
            for train, valid in kf.split(data[k*i:k*i+k]):
                train_data['data'].append(data[train])
                train_data['label'].append(label[train])
                valid_data['data'].append(data[valid])
                valid_data['label'].append(label[valid])

        else:
            idx = 0
            for train, valid in kf.split(data[k*i:k*i+k]):
                train_data['data'][idx] = np.append(train_data['data'][idx],data[k*i+train],axis=0)
                train_data['label'][idx] = np.append(train_data['label'][idx],label[k*i+train])
                valid_data['data'][idx] = np.append(valid_data['data'][idx],data[k*i+valid],axis=0)
                valid_data['label'][idx] = np.append(valid_data['label'][idx],label[k*i+valid])
                idx += 1

    return train_data, valid_data

def add_bin(data, window_size = 30):
    time_num = int(data.shape[1]/ window_size)
    output = np.zeros((data.shape[0], time_num, data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(time_num):
            for idx in range(window_size):
                output[i][j] += data[i][j * window_size + idx]

        
    return output

def add_generated_data(data_path, fake_data_path = None, bin_size = 30):
    data = torch.load(data_path)

    if fake_data_path != None:
    
        generated_data = torch.load(fake_data_path)
        
        if type(generated_data) == list:

            generated_data = torch.tensor([item.detach().numpy() for item in generated_data])
            generated_data = np.array(generated_data)
            

        ge_bin_size = 1200//generated_data.shape[2] 
        B = generated_data.shape[0]
        assert bin_size % ge_bin_size == 0 and bin_size >= ge_bin_size
        
        if ge_bin_size !=bin_size:

            spike_data = add_bin(data['data'], bin_size)
            print(spike_data[0,:,0],spike_data[0,:,10],spike_data[0,:,30])
            generated_data = add_bin(generated_data.reshape(-1, generated_data.shape[2], generated_data.shape[3]), bin_size//ge_bin_size)
            print(generated_data[0,:,0],generated_data[0,:,10],generated_data[0,:,30])
            # generated_data = generated_data.reshape(B, -1, generated_data.shape[2], generated_data.shape[3])
        else:
            spike_data = add_bin(data['data'], bin_size)
            generated_data = generated_data.reshape(-1, generated_data.shape[2], generated_data.shape[3])
        labels = data['label']
        spike_data = spike_data.reshape(8, 15, spike_data.shape[1], spike_data.shape[2])
        labels = labels.reshape(8, 15, 1)
        
        
        generated_data = generated_data.reshape(B, 8, 15, generated_data.shape[1], generated_data.shape[2])
        generated_data = generated_data.transpose(1,0,2,3,4).reshape(8, -1, generated_data.shape[3], generated_data.shape[4])
        spike_data = np.append(spike_data, generated_data, axis=1).reshape(-1, spike_data.shape[2], spike_data.shape[3])
        # if ge_bin_size !=bin_size:
            

        #     spike_data = add_bin(spike_data, bin_size//ge_bin_size)
        data['data'] = spike_data
        
        labels = np.repeat(labels, B + 1, axis = 1).reshape(-1, labels.shape[2])
        data['label'] = labels

    else:
        spike_data = add_bin(data['data'], bin_size)
        data['data'] = spike_data
    print(data['label'])
    return data