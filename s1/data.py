import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    # exchange with the corrupted mnist dataset
    train0 = np.load(r"E:\DTU\MLOPs\dtu_mlops\data\corruptmnist\train_0.npz")
    # print(train1['images'])
    train1 = np.load(r"E:\DTU\MLOPs\dtu_mlops\data\corruptmnist\train_1.npz")
    train2 = np.load(r"E:\DTU\MLOPs\dtu_mlops\data\corruptmnist\train_2.npz")
    train3 = np.load(r"E:\DTU\MLOPs\dtu_mlops\data\corruptmnist\train_3.npz")
    train4 = np.load(r"E:\DTU\MLOPs\dtu_mlops\data\corruptmnist\train_4.npz")
    train_images = torch.tensor(np.concatenate([train0['images'], train1['images'], train2['images'], train3['images'], train4['images']]))
    train_labels = torch.tensor(np.concatenate([train0['labels'], train1['labels'], train2['labels'], train3['labels'], train4['labels']]))

    test = np.load(r"E:\DTU\MLOPs\dtu_mlops\data\corruptmnist\test.npz")
    test_images = torch.tensor(test['images'])
    test_labels = torch.tensor(test['labels'])

    test_images = torch.flatten(test_images.to(torch.float32),start_dim=1)
    train_images = torch.flatten(train_images.to(torch.float32),start_dim=1)
    # print(test_images.shape)

    train_set = TensorDataset(train_images,train_labels)
    test_set = TensorDataset(test_images, test_labels)

    
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784) 
    return train_set, test_set

# mnist()