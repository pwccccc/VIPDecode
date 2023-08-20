from torch.utils.data import Dataset
import scipy.io as scio
import torch


class Data(Dataset):
    data_path = "./sort_data/VIP_decode_data.mat"
    key_list = ['spk_arr_mat', 'sdf_arr_mat', 'sample_label']
    def __init__(self,data,label):
        
        
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label[index])

        return data.to(torch.float32), label

    def __len__(self):
        return len(self.data)